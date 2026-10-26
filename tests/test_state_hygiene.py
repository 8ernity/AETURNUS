import sqlite3
from pathlib import Path

from src.agent.self_healing_engine import SelfHealingEngine
from src.agent.triage_agent import TriageAgent
from src.common.state import get_state_dir, state_db_path
from src.mlops.drift_and_retrain import MLOpsManager
from scripts.seed_demo_state import seed_demo_state


def test_single_state_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AETURNUS_STATE_DIR", str(tmp_path / "runtime_state"))
    state_dir = get_state_dir()
    assert state_dir == tmp_path / "runtime_state"
    assert state_dir.exists()
    assert state_db_path("work_orders.db") == state_dir / "work_orders.db"

    agent = TriageAgent()
    assert agent.db_path == state_dir / "work_orders.db"

    healing = SelfHealingEngine()
    assert healing.db_path == state_dir / "self_healing.db"

    mgr = MLOpsManager("FD001")
    assert mgr.drift_db_path == state_dir / "drift_monitoring.db"


def test_state_db_path_rejects_paths(tmp_path):
    try:
        state_db_path(str(tmp_path / "nested" / "x.db"))
    except ValueError:
        return
    raise AssertionError("state_db_path must reject nested paths")


def test_seed_demo_state_has_zero_open_orders(tmp_path, monkeypatch):
    monkeypatch.setenv("AETURNUS_STATE_DIR", str(tmp_path / "demo_state"))
    result = seed_demo_state()
    assert result["open_work_orders"] == 0
    assert len(result["seeded_closed_orders"]) == 2

    db_path = Path(result["state_dir"]) / "work_orders.db"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM work_orders WHERE status != 'CLOSED'").fetchone()[0] == 0
    finally:
        conn.close()

    # Idempotent: seeding twice still leaves zero open orders.
    result2 = seed_demo_state()
    assert result2["open_work_orders"] == 0
