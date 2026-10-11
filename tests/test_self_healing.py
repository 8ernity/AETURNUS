import pytest
from src.agent.self_healing_engine import SelfHealingEngine

@pytest.fixture
def engine(tmp_path):
    db_file = tmp_path / "test_self_healing.db"
    return SelfHealingEngine(db_path=db_file)

def test_self_healing_prescriptions_rule_based_only(engine):
    status = {
        "engine_id": "test_unit_1",
        "cycle": 80,
        "conservative_rul": 14.5,
        "severity": "CRITICAL",
        "suspected_subsystem": "HPC"
    }
    prescriptions = engine.generate_prescriptions(status)
    assert len(prescriptions) >= 2

    for rx in prescriptions:
        # Strict assertion: no invented numbers or synthetic physics parameters
        assert "life_extension_pct" not in rx
        assert "derate_pct" not in rx
        assert "extended_rul" not in rx
        assert "parameters" not in rx
        assert "safety_level" not in rx
        assert rx.get("status") == "RECOMMENDED"
        assert "HPC" in rx["description"]

    maint_rx = next(p for p in prescriptions if p["action_type"] == "MAINTENANCE_INSPECTION")
    assert "Immediate shutdown" in maint_rx["description"]
    assert maint_rx["severity"] == "CRITICAL"

def test_self_healing_execution_and_ledger(engine):
    res = engine.execute_action(
        engine_id="test_unit_42",
        action_type="MAINTENANCE_INSPECTION",
        cycle=45,
        cons_rul=25.0,
        subsystem="HPT"
    )
    assert res["status"] == "ACTIVE"
    assert "life_extension_pct" not in res
    assert "derate_pct" not in res
    assert "extended_rul" not in res
    assert "action_id" in res
    assert res["action_id"].startswith("ACT-")
    assert "recommended_action" in res
    assert "HPT" in res["recommended_action"]

    # Verify history ledger
    hist = engine.get_history(engine_id="test_unit_42")
    assert len(hist) >= 1
    latest = hist[0]
    assert latest["engine_id"] == "test_unit_42"
    assert latest["subsystem"] == "HPT"
    assert latest["action_type"] == "MAINTENANCE_INSPECTION"
    assert "recommended_action" in latest
