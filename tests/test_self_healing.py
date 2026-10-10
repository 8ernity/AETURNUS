import pytest
from src.agent.self_healing_engine import SelfHealingEngine

def test_self_healing_prescriptions_generation():
    engine = SelfHealingEngine()
    status = {
        "engine_id": "test_unit_1",
        "cycle": 80,
        "conservative_rul": 14.5,
        "severity": "CRITICAL",
        "suspected_subsystem": "HPC"
    }
    prescriptions = engine.generate_prescriptions(status)
    assert len(prescriptions) >= 3
    derate_rx = next(p for p in prescriptions if p["action_type"] == "DYNAMIC_DERATE")
    assert derate_rx["derate_pct"] == 20.0
    assert derate_rx["extended_rul"] > 14.5
    assert derate_rx["safety_level"] == "AUTONOMOUS_APPROVED"

def test_self_healing_execution_and_ledger():
    engine = SelfHealingEngine()
    res = engine.execute_action(
        engine_id="test_unit_42",
        action_type="DYNAMIC_DERATE",
        cycle=45,
        cons_rul=25.0,
        subsystem="HPT"
    )
    assert res["status"] == "ACTIVE"
    assert res["extended_rul"] > 25.0
    assert res["life_extension_pct"] > 0
    assert "action_id" in res
    assert res["action_id"].startswith("ACT-")

    # Verify history ledger
    hist = engine.get_history(engine_id="test_unit_42")
    assert len(hist) >= 1
    latest = hist[0]
    assert latest["engine_id"] == "test_unit_42"
    assert latest["subsystem"] == "HPT"
    assert latest["action_type"] == "DYNAMIC_DERATE"
