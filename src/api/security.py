"""Authentication, role enforcement, and API audit helpers.

The API accepts either a signed HS256 JWT or an API key configured through the
environment.  Credentials are deliberately not stored in YAML: YAML describes
the role names and the policy surface, while secrets come from the deployment
environment.

Roles are loaded from ``configs/auth.yaml``.  Every state-changing route is
authorized twice:

* the HTTP middleware rejects missing credentials before request-body parsing,
  so a malformed body can never bypass authentication; and
* route-level ``require_role`` dependencies make the policy explicit in the
  FastAPI dependency graph.

Both checks use the same role tuples in this module, which keeps the policy in
one place rather than duplicating role names across every decorator.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fastapi import HTTPException, Request, WebSocket, status

from src.common.state import state_db_path
from src.common.utils import load_yaml

# ---------------------------------------------------------------------------
# Role names and route policy
# ---------------------------------------------------------------------------

ADMIN_ONLY: Tuple[str, ...] = ("admin",)
REPLAY_ROLES: Tuple[str, ...] = ("admin", "operator")
WORK_ORDER_ROLES: Tuple[str, ...] = ("admin", "operator", "technician")
SIMULATION_ROLES: Tuple[str, ...] = ("admin", "operator", "technician")
RETRAIN_ROLES: Tuple[str, ...] = ("admin", "operator")
MODEL_ADMIN_ROLES: Tuple[str, ...] = ("admin", "operator")
PIPELINE_ROLES: Tuple[str, ...] = ("admin", "operator")
SELF_HEALING_ROLES: Tuple[str, ...] = ("admin", "operator", "technician")
WEBHOOK_ROLES: Tuple[str, ...] = ("admin",)

# Path templates, not concrete URLs.  The middleware expands ``{param}``
# segments before matching, so aliases and versioned paths share one policy.
ROUTE_ROLES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("POST", "/api/v1/replay/start"): REPLAY_ROLES,
    ("POST", "/api/v1/replay/pause"): REPLAY_ROLES,
    ("POST", "/api/v1/replay/reset"): REPLAY_ROLES,
    ("POST", "/api/v1/work-orders/{work_order_id}/action"): WORK_ORDER_ROLES,
    ("POST", "/api/v1/work-orders/{work_order_id}/advance"): WORK_ORDER_ROLES,
    ("POST", "/api/v1/rules/simulate"): SIMULATION_ROLES,
    ("POST", "/api/v1/economics/simulate"): SIMULATION_ROLES,
    ("POST", "/api/v1/mlops/retrain"): RETRAIN_ROLES,
    ("POST", "/api/v1/mlops/{name}/promote"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/mlops/promote"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/models/{name}/promote"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/models/promote"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/mlops/{name}/rollback"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/mlops/rollback"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/models/{name}/rollback"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/models/rollback"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/models/reload"): MODEL_ADMIN_ROLES,
    ("POST", "/api/v1/pipeline/run"): PIPELINE_ROLES,
    ("POST", "/api/v1/data/upload"): PIPELINE_ROLES,
    ("POST", "/api/v1/predict/batch"): WORK_ORDER_ROLES,
    ("POST", "/api/v1/monitoring/drift/compute"): RETRAIN_ROLES,
    ("POST", "/api/v1/monitoring/webhooks/test"): WEBHOOK_ROLES,
    ("POST", "/api/v1/self-healing/execute"): SELF_HEALING_ROLES,
    ("POST", "/api/v1/self-healing/actions/{action_id}/complete"): SELF_HEALING_ROLES,
}

MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity recorded in audit events."""

    actor: str
    role: str
    method: str


def _auth_config() -> Dict[str, Any]:
    return load_yaml("configs/auth.yaml") or {}


@lru_cache(maxsize=1)
def get_roles() -> Tuple[str, ...]:
    """Return the configured role names in configuration order."""
    roles = tuple(((_auth_config().get("roles") or {}).keys()))
    if not roles:
        raise RuntimeError("configs/auth.yaml must define at least one role")
    unknown_route_roles = {
        role
        for allowed in ROUTE_ROLES.values()
        for role in allowed
        if role not in roles
    }
    if unknown_route_roles:
        raise RuntimeError(
            "configs/auth.yaml is missing roles referenced by the API policy: "
            f"{sorted(unknown_route_roles)}"
        )
    return roles


def _api_key_header_name() -> str:
    return str((_auth_config().get("api_key") or {}).get("header") or "X-API-Key")


def _api_key_json_env() -> str:
    return str(
        (_auth_config().get("api_key") or {}).get("json_env")
        or "AETURNUS_API_KEYS"
    )


def _api_key_env_prefix() -> str:
    return str(
        (_auth_config().get("api_key") or {}).get("env_prefix")
        or "AETURNUS_API_KEY_"
    )


def _jwt_secret() -> str:
    return os.environ.get(
        str((_auth_config().get("jwt") or {}).get("secret_env") or "AETURNUS_JWT_SECRET"),
        "",
    )


def _jwt_issuer() -> Optional[str]:
    env_name = str(
        (_auth_config().get("jwt") or {}).get("issuer_env") or "AETURNUS_JWT_ISSUER"
    )
    return os.environ.get(env_name) or None


def _jwt_audience() -> Optional[str]:
    env_name = str(
        (_auth_config().get("jwt") or {}).get("audience_env") or "AETURNUS_JWT_AUDIENCE"
    )
    return os.environ.get(env_name) or None


def _path_regex(pattern: str) -> re.Pattern[str]:
    parts = re.split(r"(\{[^}]+\})", pattern)
    return re.compile(
        "^"
        + "".join("[^/]+" if part.startswith("{") else re.escape(part) for part in parts)
        + "$"
    )


@lru_cache(maxsize=None)
def _compiled_path_regex(pattern: str) -> re.Pattern[str]:
    return _path_regex(pattern)


def roles_for_mutation(method: str, path: str) -> Tuple[str, ...]:
    """Return the configured roles allowed for a mutating request.

    Unknown mutating routes default to admin-only.  This fail-closed default
    means adding a new POST endpoint cannot accidentally leave it public.
    """
    method = method.upper()
    for (route_method, template), roles in ROUTE_ROLES.items():
        if route_method == method and _compiled_path_regex(template).match(path):
            return roles
    return ADMIN_ONLY


# ---------------------------------------------------------------------------
# Credential parsing
# ---------------------------------------------------------------------------


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _decode_jwt(token: str) -> Optional[Principal]:
    secret = _jwt_secret()
    if not secret:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, signature_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, UnicodeDecodeError):
        return None
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        return None
    if not isinstance(payload, dict):
        return None

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    try:
        supplied_signature = _b64url_decode(signature_b64)
    except (ValueError, TypeError):
        return None
    expected_signature = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected_signature, supplied_signature):
        return None

    now = time.time()
    exp = payload.get("exp")
    nbf = payload.get("nbf")
    if exp is not None and (not isinstance(exp, (int, float)) or now >= float(exp)):
        return None
    if nbf is not None and (not isinstance(nbf, (int, float)) or now < float(nbf)):
        return None

    issuer = _jwt_issuer()
    if issuer and payload.get("iss") not in (None, issuer):
        return None
    audience = _jwt_audience()
    if audience:
        aud = payload.get("aud")
        audiences = aud if isinstance(aud, list) else [aud]
        if audience not in audiences:
            return None

    role = payload.get("role")
    if role is None and isinstance(payload.get("roles"), list):
        role = next(
            (candidate for candidate in payload["roles"] if candidate in get_roles()),
            None,
        )
    if role not in get_roles():
        return None
    actor = (
        payload.get("sub")
        or payload.get("actor")
        or payload.get("preferred_username")
        or payload.get("email")
    )
    if not isinstance(actor, str) or not actor.strip():
        return None
    return Principal(actor=actor.strip(), role=str(role), method="jwt")


def _configured_api_keys() -> Mapping[str, Tuple[str, str]]:
    """Build key -> (role, actor) from environment variables.

    Supported forms:
    * ``AETURNUS_API_KEYS`` as JSON: ``{"key": "admin"}`` or
      ``{"key": {"role": "admin", "actor": "alice"}}``.
    * ``AETURNUS_API_KEY_ADMIN``, ``..._OPERATOR``, etc. with an optional
      ``..._ACTOR`` companion.
    """
    keys: Dict[str, Tuple[str, str]] = {}
    configured_roles = get_roles()

    raw = os.environ.get(_api_key_json_env(), "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{_api_key_json_env()} must contain valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{_api_key_json_env()} must be a JSON object")
        for key, value in parsed.items():
            if not isinstance(key, str) or not key:
                continue
            if isinstance(value, str):
                role, actor = value, f"{value}@api-key"
            elif isinstance(value, dict):
                role = value.get("role")
                actor = value.get("actor") or f"{role}@api-key"
            else:
                continue
            if role in configured_roles:
                keys[key] = (str(role), str(actor))

    prefix = _api_key_env_prefix()
    for role in configured_roles:
        env_name = f"{prefix}{role.upper()}"
        key = os.environ.get(env_name)
        if not key:
            continue
        actor = os.environ.get(f"{env_name}_ACTOR") or f"{role}@api-key"
        keys[key] = (role, actor)
    return keys


def _principal_from_api_key(key: str) -> Optional[Principal]:
    if not key:
        return None
    credential = _configured_api_keys().get(key)
    if credential is None:
        return None
    role, actor = credential
    return Principal(actor=actor, role=role, method="api_key")


def _principal_from_token(token: str) -> Optional[Principal]:
    token = (token or "").strip()
    if not token:
        return None
    return _decode_jwt(token) or _principal_from_api_key(token)


def _authorization_token(headers: Mapping[str, str]) -> Optional[str]:
    value = headers.get("authorization") or headers.get("Authorization") or ""
    lower = value.lower()
    if lower.startswith("bearer "):
        return value[7:].strip()
    if lower.startswith("apikey "):
        return value[7:].strip()
    return None


def _principal_from_headers(headers: Mapping[str, str]) -> Optional[Principal]:
    api_key = headers.get(_api_key_header_name()) or headers.get(
        _api_key_header_name().lower()
    )
    if api_key:
        principal = _principal_from_api_key(api_key)
        if principal:
            return principal
    token = _authorization_token(headers)
    if token:
        return _principal_from_token(token)
    return None


def authenticate_request(request: Request) -> Principal:
    """Authenticate an HTTP request or raise 401."""
    principal = getattr(request.state, "principal", None)
    if isinstance(principal, Principal):
        return principal

    # HTTP clients may also pass ?token= for parity with the WebSocket.
    query_token = request.query_params.get("token") or request.query_params.get("api_key")
    principal = _principal_from_token(query_token) if query_token else None
    if principal is None:
        principal = _principal_from_headers(request.headers)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer JWT or X-API-Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def _role_allowed(principal: Principal, allowed_roles: Iterable[str]) -> bool:
    allowed = tuple(allowed_roles)
    # Admin is an explicit superuser. Other roles must be listed exactly.
    return principal.role == "admin" or principal.role in allowed


def require_role(*roles: str):
    """FastAPI dependency factory enforcing one or more configured roles.

    ``admin`` is always allowed.  Unknown role names fail at import time so a
    typo cannot silently create an unreachable or over-permissive route.
    """
    configured_roles = get_roles()
    unknown = [role for role in roles if role not in configured_roles]
    if unknown:
        raise ValueError(f"Unknown role(s) in require_role: {unknown}")
    allowed = tuple(roles)

    def dependency(request: Request) -> Principal:
        principal = authenticate_request(request)
        if not _role_allowed(principal, allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{principal.role}' is not permitted for this endpoint; "
                    f"allowed roles: {list(allowed)}"
                ),
            )
        request.state.principal = principal
        request.state.actor = principal.actor
        return principal

    return dependency


async def authenticate_websocket(
    websocket: WebSocket,
    allowed_roles: Optional[Sequence[str]] = None,
) -> Optional[Principal]:
    """Authenticate a WebSocket connection without accepting an anonymous peer.

    Tokens are accepted from ``?token=``/``?api_key=``, the ``X-API-Key``
    header, or a subprotocol such as ``aeturnus.token.<jwt>``.  The latter keeps
    browser clients from putting long-lived credentials in URLs if they prefer.
    Returns ``None`` after closing the socket with policy-violation code 1008.
    """
    token = (
        websocket.query_params.get("token")
        or websocket.query_params.get("access_token")
        or websocket.query_params.get("api_key")
    )
    principal = _principal_from_token(token) if token else None
    if principal is None:
        principal = _principal_from_headers(websocket.headers)

    if principal is None:
        protocol_header = websocket.headers.get("sec-websocket-protocol", "")
        protocols = [p.strip() for p in protocol_header.split(",") if p.strip()]
        for index, protocol in enumerate(protocols):
            for prefix in ("aeturnus.token.", "aeturnus.auth.", "bearer.", "token."):
                if protocol.startswith(prefix):
                    principal = _principal_from_token(protocol[len(prefix) :])
                    break
            if principal is not None:
                break
            # Also accept a two-token form such as
            # ``["aeturnus.auth", "<jwt>"]`` emitted by some browser clients.
            if (
                protocol.lower()
                in {"aeturnus.auth", "aeturnus.token", "bearer", "token"}
                and index + 1 < len(protocols)
            ):
                principal = _principal_from_token(protocols[index + 1])
                if principal is not None:
                    break
            # Some non-browser clients offer the HTTP-style value as a
            # subprotocol, e.g. ``Bearer <jwt>``.
            lower_protocol = protocol.lower()
            if lower_protocol.startswith("bearer "):
                principal = _principal_from_token(protocol[7:])
            elif lower_protocol.startswith("apikey "):
                principal = _principal_from_token(protocol[7:])
            else:
                # A bare JWT or configured API key is unambiguous enough.
                principal = _principal_from_token(protocol)
            if principal is not None:
                break

    if principal is None:
        await websocket.close(code=1008, reason="Authentication required")
        return None

    allowed = tuple(allowed_roles) if allowed_roles is not None else get_roles()
    if not _role_allowed(principal, allowed):
        await websocket.close(code=1008, reason="Forbidden")
        return None

    return principal


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

_audit_lock = threading.Lock()


def _ensure_audit_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id TEXT PRIMARY KEY,
            work_order_id TEXT,
            engine_id TEXT NOT NULL,
            cycle INTEGER NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            trace TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )


def record_audit_event(
    actor: str,
    action: str,
    *,
    engine_id: str = "api",
    work_order_id: Optional[str] = None,
    cycle: int = 0,
    reason: str = "",
    trace: Optional[Sequence[str]] = None,
) -> None:
    """Append an actor-attributed event to the canonical work-order audit log."""
    try:
        with _audit_lock:
            conn = sqlite3.connect(str(state_db_path("work_orders.db")), timeout=5.0)
            try:
                _ensure_audit_table(conn)
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, work_order_id, engine_id, cycle, action,
                        actor, reason, trace, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"EVT-{uuid.uuid4()}",
                        work_order_id,
                        engine_id,
                        int(cycle),
                        action,
                        actor,
                        reason,
                        json.dumps(list(trace or [])),
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        # Auditing must never turn a successful mutating request into a 500.
        # The domain-specific audit rows (triage/self-healing) still persist.
        pass


def record_api_audit(
    request: Request,
    principal: Optional[Principal],
    status_code: int,
) -> None:
    """Record an API mutation with the authenticated actor."""
    actor = principal.actor if principal is not None else "anonymous"
    path = request.url.path
    match = re.search(r"/work-orders/([^/]+)", path)
    work_order_id = match.group(1) if match else None
    role = principal.role if principal is not None else "unauthenticated"
    record_audit_event(
        actor=actor,
        action=f"API_{request.method.upper()}_{path}",
        work_order_id=work_order_id,
        reason=f"HTTP {status_code}",
        trace=[f"role:{role}", f"method:{principal.method if principal else 'none'}"],
    )


def cors_origins() -> List[str]:
    """Return CORS origins from the environment, never a credentialed wildcard."""
    raw = os.environ.get(
        "AETURNUS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8000,http://127.0.0.1:8000",
    )
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError(
            "AETURNUS_CORS_ORIGINS must list explicit origins; '*' cannot be "
            "combined with credentialed CORS."
        )
    return origins
