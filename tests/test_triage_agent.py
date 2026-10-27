import pytest
import sqlite3
import uuid
from pathlib import Path
from src.agent.triage_agent import TriageAgent
from src.common.utils import get_root_dir

root_dir = get_root_dir()

@pytest.fixture
def agent(tmp_path):
    db_file = tmp_path / "test_work_orders.db"
    return TriageAgent(db_path=db_file)

def test_handle_none_and_invalid_input(agent):
    order, trace = agent.evaluate_engine_cycle(None)
    assert order is None
    assert "X0:prediction_is_none" in trace

    incomplete_pred = {"engine_id": "ENG-FD001-001", "unit": 1, "cycle": 10}
    order2, trace2 = agent.evaluate_engine_cycle(incomplete_pred)
    assert order2 is not None
    assert order2["status"] == "NEEDS_REVIEW"
    assert "X0:missing_conservative_rul->exception" in trace2

    with pytest.raises(ValueError, match="unit"):
        agent.evaluate_engine_cycle({"engine_id": "ENG-FD001-001", "cycle": 10})
    with pytest.raises(ValueError, match="cycle"):
        agent.evaluate_engine_cycle({"engine_id": "ENG-FD001-001", "unit": 1})


def test_dry_run_does_not_persist_or_mutate_state(agent):
    pred = {
        "engine_id": "ENG-FD001-DRYRUN",
        "unit": 8,
        "subset": "FD001",
        "cycle": 40,
        "conservative_rul": 20.0,
        "rul_p10": 20.0,
        "rul_p50": 28.0,
        "rul_p90": 35.0,
        "fail_prob_30": 0.85,
        "anomaly_score": 0.60,
        "model_version": "rul-fd001@champion",
    }
    order, trace = agent.evaluate_engine_cycle(pred, dry_run=True)
    assert order is not None
    assert order["work_order_id"].startswith("WO-")
    assert "dry_run:not_persisted" in trace
    assert pred["engine_id"] not in agent.open_orders
    with agent._get_db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 0

def test_uuid_order_ids(agent):
    pred = {
        "engine_id": "ENG-FD001-002",
        "unit": 2,
        "subset": "FD001",
        "cycle": 50,
        "conservative_rul": 20.0,
        "rul_p10": 20.0,
        "rul_p50": 28.0,
        "rul_p90": 35.0,
        "fail_prob_30": 0.85,
        "anomaly_score": 0.60,
        "model_version": "rul-fd001@champion"
    }
    order, trace = agent.evaluate_engine_cycle(pred)
    assert order is not None
    wo_id = order["work_order_id"]
    assert wo_id.startswith("WO-")
    # Verify UUID format in ID
    uuid_str = wo_id[3:]
    assert uuid.UUID(uuid_str)

def test_hysteresis_and_deescalation(agent):
    engine_id = "ENG-FD001-003"
    # Escalate to CRITICAL (RUL = 10)
    crit_pred = {
        "engine_id": engine_id, "unit": 3, "subset": "FD001", "cycle": 50,
        "conservative_rul": 10.0, "rul_p10": 10.0, "rul_p50": 15.0, "rul_p90": 25.0,
        "fail_prob_30": 0.90, "anomaly_score": 0.85, "model_version": "rul-fd001@champion"
    }
    order, trace = agent.evaluate_engine_cycle(crit_pred)
    assert order["severity"] == "CRITICAL"

    # Now sensor conditions improve (raw severity LOW, RUL = 85)
    # Cycles 1 to 4: held at CRITICAL due to hysteresis
    for c in range(51, 55):
        low_pred = {
            "engine_id": engine_id, "unit": 3, "subset": "FD001", "cycle": c,
            "conservative_rul": 85.0, "rul_p10": 80.0, "rul_p50": 90.0, "rul_p90": 100.0,
            "fail_prob_30": 0.10, "anomaly_score": 0.20, "model_version": "rul-fd001@champion"
        }
        ord_c, tr_c = agent.evaluate_engine_cycle(low_pred)
        assert ord_c["severity"] == "CRITICAL"
        assert any("hysteresis:held_at_CRITICAL" in t for t in tr_c)

    # Cycle 5 (c = 55): 5th consecutive cycle in lower band -> de-escalate!
    low_pred_5 = {
        "engine_id": engine_id, "unit": 3, "subset": "FD001", "cycle": 55,
        "conservative_rul": 85.0, "rul_p10": 80.0, "rul_p50": 90.0, "rul_p90": 100.0,
        "fail_prob_30": 0.10, "anomaly_score": 0.20, "model_version": "rul-fd001@champion"
    }
    ord_5, tr_5 = agent.evaluate_engine_cycle(low_pred_5)
    assert any("hysteresis:de_escalated" in t for t in tr_5)

def test_reject_suppression_and_audit(agent):
    engine_id = "ENG-FD001-004"
    pred = {
        "engine_id": engine_id, "unit": 4, "subset": "FD001", "cycle": 30,
        "conservative_rul": 25.0, "rul_p10": 25.0, "rul_p50": 30.0, "rul_p90": 40.0,
        "fail_prob_30": 0.60, "anomaly_score": 0.40, "model_version": "rul-fd001@champion"
    }
    order, _ = agent.evaluate_engine_cycle(pred)
    wo_id = order["work_order_id"]

    # Operator rejects the order at cycle 30
    success = agent.human_action(wo_id, "REJECT", reason="Known sensor glitch", current_cycle=30)
    assert success is True
    assert engine_id not in agent.open_orders

    # Verify audit event in SQLite
    with agent._get_db() as conn:
        row = conn.execute("SELECT * FROM audit_events WHERE work_order_id = ? AND action = 'REJECT'", (wo_id,)).fetchone()
        assert row is not None
        assert row["action"] == "REJECT"
        assert row["reason"] == "Known sensor glitch"

    # Subsequent non-critical cycles (e.g. cycle 35) are suppressed (window = 10 cycles, until cycle 40)
    pred_cycle_35 = {
        "engine_id": engine_id, "unit": 4, "subset": "FD001", "cycle": 35,
        "conservative_rul": 28.0, "rul_p10": 28.0, "rul_p50": 35.0, "rul_p90": 45.0,
        "fail_prob_30": 0.50, "anomaly_score": 0.30, "model_version": "rul-fd001@champion"
    }
    ord_suppressed, tr_suppressed = agent.evaluate_engine_cycle(pred_cycle_35)
    assert ord_suppressed is None
    assert any("suppression:active" in t for t in tr_suppressed)

def test_never_overwrite_human_status(agent):
    engine_id = "ENG-FD001-005"
    pred = {
        "engine_id": engine_id, "unit": 5, "subset": "FD001", "cycle": 40,
        "conservative_rul": 20.0, "rul_p10": 20.0, "rul_p50": 25.0, "rul_p90": 35.0,
        "fail_prob_30": 0.80, "anomaly_score": 0.60, "model_version": "rul-fd001@champion"
    }
    order, _ = agent.evaluate_engine_cycle(pred)
    wo_id = order["work_order_id"]

    # Human approves
    agent.human_action(wo_id, "APPROVE", reason="Maintenance crew dispatched")
    assert agent.open_orders[engine_id]["status"] == "APPROVED"

    # Recurring tick with an exception trigger
    pred2 = {
        "engine_id": engine_id, "unit": 5, "subset": "FD001", "cycle": 41,
        "conservative_rul": 18.0, "rul_p10": 5.0, "rul_p50": 25.0, "rul_p90": 70.0, # X1 wide uncertainty
        "fail_prob_30": 0.85, "anomaly_score": 0.65, "model_version": "rul-fd001@champion"
    }
    order2, trace2 = agent.evaluate_engine_cycle(pred2)
    assert order2["status"] == "APPROVED"
    assert any("human_status_preserved:APPROVED" in t for t in trace2)

def test_refresh_rul_on_open_orders(agent):
    engine_id = "ENG-FD001-006"
    pred1 = {
        "engine_id": engine_id, "unit": 6, "subset": "FD001", "cycle": 40,
        "conservative_rul": 55.0, "rul_p10": 55.0, "rul_p50": 65.0, "rul_p90": 80.0,
        "fail_prob_30": 0.20, "anomaly_score": 0.85, # M2 fires
        "model_version": "rul-fd001@champion"
    }
    order1, _ = agent.evaluate_engine_cycle(pred1)
    assert order1["conservative_rul"] == 55.0

    pred2 = {
        "engine_id": engine_id, "unit": 6, "subset": "FD001", "cycle": 45,
        "conservative_rul": 48.0, "rul_p10": 48.0, "rul_p50": 58.0, "rul_p90": 72.0,
        "fail_prob_30": 0.25, "anomaly_score": 0.85,
        "model_version": "rul-fd001@champion"
    }
    order2, _ = agent.evaluate_engine_cycle(pred2)
    assert order2["conservative_rul"] == 48.0
    assert order2["current_cycle"] == 45

def test_cooldown_from_real_closure_cycle(agent):
    engine_id = "ENG-FD001-007"
    pred = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 50,
        "conservative_rul": 25.0, "rul_p10": 25.0, "rul_p50": 30.0, "rul_p90": 40.0,
        "fail_prob_30": 0.60, "anomaly_score": 0.50, "model_version": "rul-fd001@champion"
    }
    order, _ = agent.evaluate_engine_cycle(pred)
    wo_id = order["work_order_id"]

    # Close order at cycle 50
    agent.human_action(wo_id, "CLOSE", reason="Part replaced", current_cycle=50)

    # Within cooldown (cycle 60 < 50 + 20)
    pred_c60 = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 60,
        "conservative_rul": 28.0, "rul_p10": 28.0, "rul_p50": 35.0, "rul_p90": 45.0,
        "fail_prob_30": 0.50, "anomaly_score": 0.40, "model_version": "rul-fd001@champion"
    }
    ord_c60, tr_c60 = agent.evaluate_engine_cycle(pred_c60)
    assert ord_c60 is None
    assert any("cooldown:active" in t for t in tr_c60)

    # After cooldown (cycle 75 > 50 + 20)
    pred_c75 = {
        "engine_id": engine_id, "unit": 7, "subset": "FD001", "cycle": 75,
        "conservative_rul": 22.0, "rul_p10": 22.0, "rul_p50": 28.0, "rul_p90": 38.0,
        "fail_prob_30": 0.70, "anomaly_score": 0.60, "model_version": "rul-fd001@champion"
    }
    ord_c75, tr_c75 = agent.evaluate_engine_cycle(pred_c75)
    assert ord_c75 is not None
    assert any("dedupe:new_order_created" in t for t in tr_c75)

def test_independent_exception_checks(agent):
    # Test X1 & X2 independently
    pred_x1 = {
        "engine_id": "ENG-FD001-X01", "unit": 10, "subset": "FD001", "cycle": 30,
        "conservative_rul": 50.0, "rul_p10": 10.0, "rul_p50": 60.0, "rul_p90": 95.0, # width = 85 > 40
        "fail_prob_30": 0.30, "anomaly_score": 0.10, "model_version": "rul-fd001@champion"
    }
    order_x1, trace_x1 = agent.evaluate_engine_cycle(pred_x1)
    assert order_x1["status"] == "NEEDS_REVIEW"
    assert "exception:X1" in trace_x1

    # Test X3: Flatline
    pred_x3 = {
        "engine_id": "ENG-FD001-X03", "unit": 11, "subset": "FD001", "cycle": 30,
        "conservative_rul": 40.0, "rul_p10": 35.0, "rul_p50": 45.0, "rul_p90": 55.0,
        "fail_prob_30": 0.30, "anomaly_score": 0.30, "model_version": "rul-fd001@champion"
    }
    order_x3, trace_x3 = agent.evaluate_engine_cycle(pred_x3, data_quality_flags={"flatlines_count": 4})
    assert order_x3["status"] == "NEEDS_REVIEW"
    assert "exception:X3" in trace_x3

    # Test X7: Missing model version
    pred_x7 = {
        "engine_id": "ENG-FD001-X07", "unit": 12, "subset": "FD001", "cycle": 30,
        "conservative_rul": 45.0, "rul_p10": 40.0, "rul_p50": 50.0, "rul_p90": 60.0,
        "fail_prob_30": 0.20, "anomaly_score": 0.30, "model_version": "unknown"
    }
    order_x7, trace_x7 = agent.evaluate_engine_cycle(pred_x7)
    assert order_x7["status"] == "NEEDS_REVIEW"
    assert "exception:X7" in trace_x7

def test_sqlite_persistence(agent):
    pred = {
        "engine_id": "ENG-FD001-099", "unit": 99, "subset": "FD001", "cycle": 25,
        "conservative_rul": 12.0, "rul_p10": 12.0, "rul_p50": 18.0, "rul_p90": 26.0,
        "fail_prob_30": 0.95, "anomaly_score": 0.90, "model_version": "rul-fd001@champion"
    }
    order, _ = agent.evaluate_engine_cycle(pred)
    wo_id = order["work_order_id"]

    # Verify directly from SQLite
    with agent._get_db() as conn:
        row = conn.execute("SELECT * FROM work_orders WHERE work_order_id = ?", (wo_id,)).fetchone()
        assert row is not None
        assert row["engine_id"] == "ENG-FD001-099"
        assert row["severity"] == "CRITICAL"
        assert row["conservative_rul"] == 12.0

def test_agent_idempotency(agent):
    """Calling evaluate_engine_cycle multiple times for same engine state does not create redundant orders."""
    pred = {
        "engine_id": "ENG-FD001-IDEMPOTENT", "unit": 77, "subset": "FD001", "cycle": 40,
        "conservative_rul": 15.0, "rul_p10": 15.0, "rul_p50": 20.0, "rul_p90": 28.0,
        "fail_prob_30": 0.90, "anomaly_score": 0.85, "model_version": "rul-fd001@champion"
    }
    order1, trace1 = agent.evaluate_engine_cycle(pred)
    assert order1 is not None
    order_id1 = order1["work_order_id"]

    # Re-evaluate with exact same prediction
    order2, trace2 = agent.evaluate_engine_cycle(pred)
    assert order2 is not None
    # Must update or maintain existing order without spawning a new work order ID
    assert order2["work_order_id"] == order_id1
    assert "dedupe:open_order_refreshed" in trace2

