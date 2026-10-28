import base64
import hashlib
import hmac
import json
import os
import re
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.api.server as server
from src.api.security import MUTATING_METHODS, cors_origins
from tests.auth_helpers import OPERATOR_HEADERS, VIEWER_HEADERS


def _mutating_routes():
    seen = set()
    for route in server.app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        if not path:
            continue
        for method in methods:
            if method in MUTATING_METHODS:
                key = (method, path)
                if key not in seen:
                    seen.add(key)
                    yield key


def _concrete_path(template: str) -> str:
    return re.sub(r"\{[^}]+\}", "test-id", template)


def _jwt(role: str, actor: str = "jwt-user") -> str:
    secret = os.environ.get("AETURNUS_JWT_SECRET", "test-jwt-secret")
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": actor,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
    }

    def enc(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    signing_input = f"{enc(header)}.{enc(payload)}"
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def test_every_mutating_route_rejects_missing_credentials():
    unauth = TestClient(server.app)
    routes = list(_mutating_routes())
    assert routes, "Expected at least one mutating route"

    # Static guard: a newly added mutation must also appear in the dependency
    # graph, not only rely on the middleware fallback.
    for route in server.app.routes:
        methods = getattr(route, "methods", None) or set()
        if methods & MUTATING_METHODS:
            assert getattr(route, "dependencies", None), (
                f"{sorted(methods & MUTATING_METHODS)} {route.path} has no route-level "
                "require_role dependency"
            )

    for method, template in routes:
        response = unauth.request(method, _concrete_path(template))
        assert response.status_code == 401, (
            f"{method} {template} returned {response.status_code} without credentials"
        )


def test_every_mutating_route_rejects_wrong_role():
    viewer = TestClient(server.app, headers=VIEWER_HEADERS)
    for method, template in _mutating_routes():
        response = viewer.request(method, _concrete_path(template))
        assert response.status_code == 403, (
            f"{method} {template} returned {response.status_code} for viewer role"
        )


def test_jwt_authentication_and_role_rejection():
    client = TestClient(server.app)
    admin_response = client.post(
        "/api/v1/replay/reset",
        headers={"Authorization": f"Bearer {_jwt('admin', 'jwt-admin')}"},
    )
    assert admin_response.status_code == 200

    viewer_response = client.post(
        "/api/v1/replay/reset",
        headers={"Authorization": f"Bearer {_jwt('viewer', 'jwt-viewer')}"},
    )
    assert viewer_response.status_code == 403


def test_cors_origins_never_use_credentialed_wildcard(monkeypatch):
    assert "*" not in cors_origins()
    monkeypatch.setenv("AETURNUS_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError):
        cors_origins()


def test_websocket_requires_authentication():
    unauth = TestClient(server.app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with unauth.websocket_connect("/ws") as websocket:
            websocket.receive_text()
    assert exc_info.value.code == 1008


def test_websocket_accepts_authenticated_operator():
    authenticated = TestClient(server.app, headers=OPERATOR_HEADERS)
    with authenticated.websocket_connect("/ws") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "connection_established"


def test_websocket_accepts_query_token():
    client = TestClient(server.app)
    with client.websocket_connect("/ws?token=test-operator-key") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "connection_established"


def test_websocket_accepts_subprotocol_token():
    client = TestClient(server.app)
    with client.websocket_connect(
        "/ws", subprotocols=["aeturnus.auth", "test-operator-key"]
    ) as websocket:
        message = websocket.receive_json()
    assert message["type"] == "connection_established"
