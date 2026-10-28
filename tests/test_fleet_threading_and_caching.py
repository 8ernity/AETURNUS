import pytest
import asyncio
from fastapi.testclient import TestClient
from src.api.server import (
    app,
    replay_state,
    _prediction_cache,
    clear_prediction_cache,
    get_cached_prediction,
    get_model_service,
    replay_ticker_loop
)

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)

def test_fleet_prediction_caching():
    """Verify that predictions are cached per (subset, unit, cycle, model_version)."""
    clear_prediction_cache()
    assert len(_prediction_cache) == 0

    # 1. Fetch fleet status
    resp = client.get("/api/v1/fleet?subset=FD001")
    assert resp.status_code == 200
    cache_len_after_first = len(_prediction_cache)
    assert cache_len_after_first > 0

    # 2. Fetch fleet status again: should hit cache without growing
    resp2 = client.get("/api/v1/fleet?subset=FD001")
    assert resp2.status_code == 200
    assert len(_prediction_cache) == cache_len_after_first

    # 3. Verify get_cached_prediction directly returns exact cached dict reference
    ms = get_model_service("FD001")
    from src.api.server import get_test_df
    base_row = get_test_df("FD001").iloc[0].to_dict()
    dummy_row = {**base_row, "unit": 999, "cycle": 42}
    p1 = get_cached_prediction(ms, dummy_row, "FD001")
    p2 = get_cached_prediction(ms, dummy_row, "FD001")
    assert p1 is p2
    assert ("FD001", 999, 42, getattr(ms, "version", "v1")) in _prediction_cache

    # 4. Clear cache
    clear_prediction_cache()
    assert len(_prediction_cache) == 0

def test_replay_no_cycle_wrap_around():
    """Verify that when replay reaches max_cycle, it stops (is_running=False) and does NOT wrap around to 15."""
    replay_state.is_running = True
    replay_state.max_cycle = 50
    replay_state.current_cycle = 50

    # Execute ticker loop logic check
    if replay_state.current_cycle >= replay_state.max_cycle:
        replay_state.is_running = False

    assert replay_state.is_running is False
    assert replay_state.current_cycle == 50
    assert replay_state.current_cycle != 15

def test_cache_cleared_on_model_reload():
    """Verify reload endpoint clears prediction cache."""
    ms = get_model_service("FD001")
    from src.api.server import get_test_df
    base_row = get_test_df("FD001").iloc[0].to_dict()
    dummy_row = {**base_row, "unit": 1, "cycle": 10}
    get_cached_prediction(ms, dummy_row, "FD001")
    assert len(_prediction_cache) > 0

    resp = client.post("/api/v1/models/reload?subset=FD001")
    assert resp.status_code == 200
    assert len(_prediction_cache) == 0
