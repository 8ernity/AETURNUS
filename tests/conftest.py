import pytest


@pytest.fixture(autouse=True)
def reset_replay_window():
    from src.api.server import replay_state

    replay_state.recent_replay_window = []
    yield
    replay_state.recent_replay_window = []
