import pytest
import sqlite3
from pathlib import Path
from src.agent.triage_agent import TriageAgent

@pytest.fixture
def agent(tmp_path):
    db_file = tmp_path / "test_agent_autoheal.db"
    return TriageAgent(db_path=db_file)

def test_auto_close_order_on_sustained_recovery(agent):
    """Verify open orders automatically close when engine sustains healthy operating state for >= deescalate cycles."""
    engine_id = "ENG-AUTO-001"
    # 1. Trigger critical order
    crit_pred = {
        "engine_id": engine_id, "unit": 1, "subset": "FD001", "cycle": 40,
        "conservative_rul": 12.0, "rul_p10": 10.0, "rul_p50": 15.0, "rul_p90": 20.0,
        "fail_prob_30": 0.92, "anomaly_score": 0.88, "model_version": "rul-fd001@champion"
    }
    ord1, tr1 = agent.evaluate_engine_cycle(crit_pred)
    assert ord1 is not None
    assert ord1["severity"] == "CRITICAL"
    assert ord1["status"] in ["AUTO_CREATED", "NEEDS_REVIEW"]
    assert engine_id in agent.open_orders

    # 2. Machine undergoes cyber-physical restoration/maintenance: healthy conditions for cycles 41..44
    for c in range(41, 45):
        healthy_pred = {
            "engine_id": engine_id, "unit": 1, "subset": "FD001", "cycle": c,
            "conservative_rul": 115.0, "rul_p10": 110.0, "rul_p50": 120.0, "rul_p90": 130.0,
            "fail_prob_30": 0.02, "anomaly_score": 0.10, "model_version": "rul-fd001@champion"
        }
        ord_c, tr_c = agent.evaluate_engine_cycle(healthy_pred)
        assert ord_c is not None
        assert ord_c["status"] != "CLOSED"
        assert engine_id in agent.open_orders

    # 3. Cycle 45 (5th consecutive healthy cycle): sustained recovery triggers AUTO-CLOSE
    healthy_pred_5 = {
        "engine_id": engine_id, "unit": 1, "subset": "FD001", "cycle": 45,
        "conservative_rul": 115.0, "rul_p10": 110.0, "rul_p50": 120.0, "rul_p90": 130.0,
        "fail_prob_30": 0.02, "anomaly_score": 0.10, "model_version": "rul-fd001@champion"
    }
    ord_closed, tr_closed = agent.evaluate_engine_cycle(healthy_pred_5)
    assert ord_closed["status"] == "CLOSED"
    assert ord_closed["closed_at_cycle"] == 45
    assert engine_id not in agent.open_orders
    assert any("auto_close:sustained_recovery" in t for t in tr_closed)

    # Verify audit event in DB
    with agent._get_db() as conn:
        evt = conn.execute("SELECT * FROM audit_events WHERE engine_id = ? AND action = 'AUTO_CLOSE'", (engine_id,)).fetchone()
        assert evt is not None
        assert evt["cycle"] == 45

def test_downgrade_order_on_sustained_partial_recovery(agent):
    """Verify open orders downgrade severity (e.g. CRITICAL -> MEDIUM) on sustained lower severity."""
    engine_id = "ENG-AUTO-002"
    crit_pred = {
        "engine_id": engine_id, "unit": 2, "subset": "FD001", "cycle": 50,
        "conservative_rul": 10.0, "rul_p10": 10.0, "rul_p50": 15.0, "rul_p90": 20.0,
        "fail_prob_30": 0.90, "anomaly_score": 0.85, "model_version": "rul-fd001@champion"
    }
    ord_crit, _ = agent.evaluate_engine_cycle(crit_pred)
    assert ord_crit["severity"] == "CRITICAL"

    # Sustained conditions in MEDIUM band (RUL = 50, cycles 51..55)
    for c in range(51, 56):
        med_pred = {
            "engine_id": engine_id, "unit": 2, "subset": "FD001", "cycle": c,
            "conservative_rul": 50.0, "rul_p10": 45.0, "rul_p50": 55.0, "rul_p90": 65.0,
            "fail_prob_30": 0.15, "anomaly_score": 0.25, "model_version": "rul-fd001@champion"
        }
        ord_med, tr_med = agent.evaluate_engine_cycle(med_pred)

    assert ord_med["severity"] == "MEDIUM"
    assert any("dedupe:downgraded_to_MEDIUM" in t for t in tr_med)
    assert engine_id in agent.open_orders

def test_persistence_of_hysteresis_and_history_across_reloads(tmp_path):
    """Verify hysteresis and engine state history persist in SQLite across TriageAgent instances."""
    db_file = tmp_path / "shared_agent.db"
    agent1 = TriageAgent(db_path=db_file)
    engine_id = "ENG-PERSIST-001"

    pred1 = {
        "engine_id": engine_id, "unit": 10, "subset": "FD001", "cycle": 20,
        "conservative_rul": 25.0, "rul_p10": 20.0, "rul_p50": 28.0, "rul_p90": 35.0,
        "fail_prob_30": 0.65, "anomaly_score": 0.55, "model_version": "rul-fd001@champion"
    }
    agent1.evaluate_engine_cycle(pred1)

    assert engine_id in agent1.engine_hysteresis
    assert len(agent1.engine_state_history[engine_id]) == 1

    # Instantiate fresh Agent 2 pointing to same database
    agent2 = TriageAgent(db_path=db_file)
    assert engine_id in agent2.engine_hysteresis
    assert agent2.engine_hysteresis[engine_id]["held_severity"] == agent1.engine_hysteresis[engine_id]["held_severity"]
    assert len(agent2.engine_state_history[engine_id]) == 1
    assert agent2.engine_state_history[engine_id][0]["cycle"] == 20

def test_x7_real_exceptions(agent):
    """Verify comprehensive X7 triggers: telemetry data gaps, out-of-order time regressions, and missing model."""
    engine_id = "ENG-X7-TEST"

    # 1. Normal evaluation at cycle 10
    pred_c10 = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 10,
        "conservative_rul": 90.0, "rul_p10": 85.0, "rul_p50": 95.0, "rul_p90": 105.0,
        "fail_prob_30": 0.05, "anomaly_score": 0.15, "model_version": "rul-fd001@champion"
    }
    agent.evaluate_engine_cycle(pred_c10)

    # 2. X7: Telemetry data gap (cycle jumps from 10 to 25, gap = 15 > 5)
    pred_gap = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 25,
        "conservative_rul": 85.0, "rul_p10": 80.0, "rul_p50": 90.0, "rul_p90": 100.0,
        "fail_prob_30": 0.08, "anomaly_score": 0.18, "model_version": "rul-fd001@champion"
    }
    ord_gap, tr_gap = agent.evaluate_engine_cycle(pred_gap)
    assert ord_gap is not None
    assert ord_gap["status"] == "NEEDS_REVIEW"
    assert "exception:X7" in tr_gap
    assert "X7: Telemetry data gap detected" in ord_gap["exception"]

    # 3. X7: Out of order cycle regression (cycle 20 received after cycle 25)
    pred_regress = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 20,
        "conservative_rul": 85.0, "rul_p10": 80.0, "rul_p50": 90.0, "rul_p90": 100.0,
        "fail_prob_30": 0.08, "anomaly_score": 0.18, "model_version": "rul-fd001@champion"
    }
    ord_reg, tr_reg = agent.evaluate_engine_cycle(pred_regress)
    assert ord_reg["status"] == "NEEDS_REVIEW"
    assert "exception:X7" in tr_reg
    assert "time regression" in ord_reg["exception"]

    # 4. X7: Data quality flag reporting gap
    pred_flag_gap = {
        "engine_id": "ENG-X7-FLAG", "unit": 8, "subset": "FD001", "cycle": 30,
        "conservative_rul": 80.0, "rul_p10": 75.0, "rul_p50": 85.0, "rul_p90": 95.0,
        "fail_prob_30": 0.05, "anomaly_score": 0.15, "model_version": "rul-fd001@champion"
    }
    ord_flag, tr_flag = agent.evaluate_engine_cycle(pred_flag_gap, data_quality_flags={"has_gap": True, "cycle_gap": 8})
    assert ord_flag["status"] == "NEEDS_REVIEW"
    assert "exception:X7" in tr_flag
