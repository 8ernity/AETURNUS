import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.economics.cost_simulator import FleetCostSimulator, DEFAULT_COST_CONFIG

client = TestClient(app)

def test_simulator_initialization():
    sim = FleetCostSimulator("FD001")
    assert sim.subset == "FD001"
    assert len(sim.eval_data) > 0
    assert "cost_catastrophic" in sim.cost_cfg

def test_simulate_fleet_metrics():
    sim = FleetCostSimulator("FD001")
    res = sim.simulate_fleet(threshold=25)
    assert "summary" in res
    assert "policies" in res
    assert "breakdown_chart" in res
    assert "ledger" in res
    
    # Net savings must be positive
    assert res["summary"]["net_savings_vs_reactive"] > 0
    assert res["summary"]["cost_reduction_percent"] > 50.0
    assert res["policies"]["aeturnus"]["total_cost"] < res["policies"]["reactive"]["total_cost"]
    assert res["summary"]["avoided_catastrophic_failures"] > 0

def test_sweep_cost_curve_convexity():
    sim = FleetCostSimulator("FD001")
    curve = sim.sweep_cost_curve(min_threshold=5, max_threshold=40, step=2)
    assert "optimal_threshold" in curve
    assert "curve" in curve
    assert len(curve["curve"]) > 10
    assert 5 <= curve["optimal_threshold"] <= 40
    assert curve["minimum_cost"] > 0

def test_economics_api_endpoints():
    # GET simulate
    resp1 = client.get("/api/v1/economics/simulate?subset=FD001&threshold=25")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["summary"]["net_savings_vs_reactive"] > 0

    # GET curve
    resp2 = client.get("/api/v1/economics/curve?subset=FD001")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "optimal_threshold" in data2
    assert len(data2["curve"]) > 0

    # POST simulate with custom financial parameters
    payload = {
        "subset": "FD001",
        "threshold": 30.0,
        "cost_catastrophic": 3500000.0,
        "cost_overhaul": 250000.0,
        "cost_wasted_cycle": 1500.0
    }
    resp3 = client.post("/api/v1/economics/simulate", json=payload)
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["policies"]["reactive"]["total_cost"] == 3500000.0 * data3["fleet_size"]
