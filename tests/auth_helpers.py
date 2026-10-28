"""Shared credentials for tests that exercise state-changing API routes."""

ROLE_KEYS = {
    "admin": "test-admin-key",
    "operator": "test-operator-key",
    "technician": "test-technician-key",
    "viewer": "test-viewer-key",
}


def headers_for(role: str = "admin") -> dict[str, str]:
    return {"X-API-Key": ROLE_KEYS[role]}


ADMIN_HEADERS = headers_for("admin")
OPERATOR_HEADERS = headers_for("operator")
TECHNICIAN_HEADERS = headers_for("technician")
VIEWER_HEADERS = headers_for("viewer")
