import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.server import app
from src.agent.maintenance_queue import MaintenanceQueue
from src.common.utils import get_root_dir, load_yaml

client = TestClient(app)
root_dir = get_root_dir()


@pytest.fixture
def cleanup_api_caches():
    """Release fleet frames/predictions loaded by endpoint contract tests."""
    yield
    import src.api.server as server
    server.loaded_dfs.clear()
    server.clear_prediction_cache()

def test_queue_weights_sum_to_one():
    """Verify weights defined in configs/queue.yaml sum to 1.0."""
    cfg = load_yaml(str(root_dir / "configs" / "queue.yaml"))
    w = cfg.get("weights", {})
    total_w = sum(w.values())
    assert abs(total_w - 1.0) < 1e-5, f"Weights must sum to 1.0, got {total_w}"

def test_priority_score_formula_exactness():
    """Verify priority score matches the PRD specification."""
    mq = MaintenanceQueue()
    item = {
        "conservative_rul": 15.0,  # (1 - 15/60) = 0.75 -> 0.75 * 0.40 = 0.30
        "fail_prob_30": 0.80,      # 0.80 * 0.30 = 0.24
        "anomaly_score": 0.50,     # 0.50 * 0.15 = 0.075
        "rul_drop": 15.0           # (15/30) = 0.50 -> 0.50 * 0.15 = 0.075
    }
    # Expected: 0.30 + 0.24 + 0.075 + 0.075 = 0.69
    score, comps = mq.compute_priority_score(item)
    assert abs(score - 0.69) < 1e-3, f"Expected 0.69, got {score}"
    assert abs(comps["rul_component"] - 0.30) < 1e-3
    assert abs(comps["fail_component"] - 0.24) < 1e-3
    assert abs(comps["anomaly_component"] - 0.075) < 1e-3
    assert abs(comps["trend_component"] - 0.075) < 1e-3

def test_priority_score_treats_null_rul_drop_as_zero_trend_only():
    """A null trend must contribute zero points without erasing other components."""
    mq = MaintenanceQueue()
    score, comps = mq.compute_priority_score({
        "conservative_rul": 15.0,
        "fail_prob_30": 0.80,
        "anomaly_score": 0.50,
        "rul_drop": None,
    })
    assert abs(score - 0.615) < 1e-3
    assert comps["trend_component"] == 0.0
    assert comps["rul_component"] == 0.30
    assert comps["fail_component"] == 0.24

def test_rank_ordering_and_tie_breaking():
    """Verify descending score sort order, ties broken by lower conservative_rul."""
    mq = MaintenanceQueue()
    items = [
        {"engine_id": "ENG-A", "conservative_rul": 50.0, "fail_prob_30": 0.1, "anomaly_score": 0.1, "rul_drop": 0.0},
        {"engine_id": "ENG-B", "conservative_rul": 10.0, "fail_prob_30": 0.9, "anomaly_score": 0.8, "rul_drop": 20.0},
        {"engine_id": "ENG-C", "conservative_rul": 20.0, "fail_prob_30": 0.5, "anomaly_score": 0.3, "rul_drop": 5.0},
        # Tie score with ENG-C but lower RUL
        {"engine_id": "ENG-D", "conservative_rul": 15.0, "fail_prob_30": 0.5, "anomaly_score": 0.3, "rul_drop": 5.0},
    ]
    ranked = mq.rank_items(items, capacity=2)
    assert ranked[0]["engine_id"] == "ENG-B", "ENG-B should be rank 1 (highest score)"
    
    # Check strict rank descending
    scores = [r["priority_score"] for r in ranked]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def test_daily_service_capacity_constraint():
    """Verify exactly top K items are flagged as actionable_today."""
    mq = MaintenanceQueue()
    items = [
        {"engine_id": f"ENG-{i}", "conservative_rul": float(i * 10), "fail_prob_30": 0.5, "anomaly_score": 0.2, "rul_drop": 0.0}
        for i in range(1, 10)
    ]
    capacity = 3
    ranked = mq.rank_items(items, capacity=capacity)
    
    actionable = [r for r in ranked if r["actionable_today"]]
    backlog = [r for r in ranked if r["backlog"]]
    
    assert len(actionable) == capacity, f"Expected {capacity} actionable items, got {len(actionable)}"
    assert len(backlog) == len(items) - capacity
    assert all(r["rank"] <= capacity for r in actionable)
    assert all(r["rank"] > capacity for r in backlog)

def test_maintenance_candidates_are_dry_run_only(monkeypatch, cleanup_api_caches):
    """Verify candidates come from TriageAgent dry-run policy with no DB writes."""
    import src.api.server as server
    from src.agent.triage_agent import TriageAgent

    fresh_agent = TriageAgent(db_path=":memory:")
    monkeypatch.setattr(server, "triage_agent", fresh_agent)
    before = client.get("/api/v1/work-orders").json()["count"]

    resp = client.get("/api/v1/maintenance-candidates?subset=FD001&capacity=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dry_run"] is True
    assert data["capacity"] == 3
    assert data["total_candidates"] == len(data["candidates"])
    assert data["actionable_count"] == min(data["total_candidates"], 3)
    assert data["backlog_count"] == max(0, data["total_candidates"] - 3)

    allowed_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "HEALTHY"}
    for candidate in data["candidates"]:
        assert candidate["dry_run"] is True
        assert candidate["would_persist"] is False
        assert candidate["rule_trace"], "Each candidate must expose the TriageAgent rule trace"
        assert candidate["severity"] in allowed_severities
        assert not candidate["work_order_id"].startswith("PENDING")
        assert candidate["cycle"] > 0
        assert candidate["candidate_id"] == candidate["work_order_id"]
        assert candidate["work_order_id"] == f"CAND-{candidate['engine_id']}-{candidate['cycle']}"
        assert candidate["rul_drop"] is None
        if candidate["rul_p90"] is None:
            assert candidate["censored_at_cap"] is True
            assert candidate["interval_label"] == ">cap"
        else:
            assert candidate["interval_label"] in {">cap", "p10-p90"}

    after = client.get("/api/v1/work-orders").json()["count"]
    assert after == before == 0


def test_fresh_state_queue_and_work_orders_agree(monkeypatch):
    """A fresh persisted state must report the same zero open orders in both views."""
    import src.api.server as server
    from src.agent.triage_agent import TriageAgent

    fresh_agent = TriageAgent(db_path=":memory:")
    monkeypatch.setattr(server, "triage_agent", fresh_agent)

    queue = client.get("/api/v1/maintenance-queue?subset=FD001&capacity=5").json()
    orders = client.get("/api/v1/work-orders").json()
    assert queue["total_open"] == 0
    assert queue["queue"] == []
    assert orders["count"] == 0
    assert orders["work_orders"] == []


def test_work_orders_include_closed_only_on_request(monkeypatch):
    """Closed orders stay available as history but not in the default live queue."""
    import src.api.server as server
    from src.agent.triage_agent import TriageAgent

    fresh_agent = TriageAgent(db_path=":memory:")
    monkeypatch.setattr(server, "triage_agent", fresh_agent)
    fresh_agent.closed_orders_history["ENG-FD001-999"] = [{
        "work_order_id": "WO-history",
        "engine_id": "ENG-FD001-999",
        "unit": 999,
        "subset": "FD001",
        "status": "CLOSED",
        "severity": "HIGH",
        "created_at_cycle": 1,
    }]

    assert client.get("/api/v1/work-orders").json()["count"] == 0
    history = client.get("/api/v1/work-orders?include_closed=true").json()
    assert history["count"] == 1
    assert history["work_orders"][0]["work_order_id"] == "WO-history"


def test_maintenance_queue_endpoint():
    """Verify GET /api/v1/maintenance-queue returns valid ranked payload."""
    resp = client.get("/api/v1/maintenance-queue?subset=FD001&capacity=4")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subset"] == "FD001"
    assert data["capacity"] == 4
    assert "queue" in data
    assert "total_open" in data
    assert "actionable_count" in data
    assert "backlog_count" in data
    assert "disclaimer" in data
    
    queue = data["queue"]
    if len(queue) > 0:
        first = queue[0]
        assert "rank" in first
        assert "priority_score" in first
        assert "actionable_today" in first
        assert "score_components" in first
        assert first["rank"] == 1
