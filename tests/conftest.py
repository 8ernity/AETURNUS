import os

import pytest

# The API rejects unauthenticated mutations by design.  Tests use these
# ephemeral keys; no credential is committed to configs/auth.yaml.
os.environ.setdefault("AETURNUS_API_KEY_ADMIN", "test-admin-key")
os.environ.setdefault("AETURNUS_API_KEY_ADMIN_ACTOR", "test-admin")
os.environ.setdefault("AETURNUS_API_KEY_OPERATOR", "test-operator-key")
os.environ.setdefault("AETURNUS_API_KEY_OPERATOR_ACTOR", "test-operator")
os.environ.setdefault("AETURNUS_API_KEY_TECHNICIAN", "test-technician-key")
os.environ.setdefault("AETURNUS_API_KEY_TECHNICIAN_ACTOR", "test-technician")
os.environ.setdefault("AETURNUS_API_KEY_VIEWER", "test-viewer-key")
os.environ.setdefault("AETURNUS_API_KEY_VIEWER_ACTOR", "test-viewer")
os.environ.setdefault("AETURNUS_JWT_SECRET", "test-jwt-secret")


@pytest.fixture(autouse=True)
def reset_replay_window():
    from src.api.server import replay_state

    replay_state.recent_replay_window = []
    yield
    replay_state.recent_replay_window = []
