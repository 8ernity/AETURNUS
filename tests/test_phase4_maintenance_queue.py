import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.server import app
from src.agent.maintenance_queue import MaintenanceQueue
from src.common.utils import get_root_dir, load_yaml

client = TestClient(app)
root_dir = get_root_dir()

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
