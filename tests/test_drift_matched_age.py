import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.api.server import app, replay_state
from src.mlops.drift_and_retrain import MLOpsManager
from src.common.utils import get_root_dir

client = TestClient(app)
root_dir = get_root_dir()

def test_default_state_must_not_alarm():
    """Item 5: Default monitoring state must not trigger a drift alarm."""
    # Ensure replay buffer is clean/empty
    replay_state.recent_replay_window = []
    # The drift DB persists across tests/processes; clear it so this test observes the
    # default (no persisted drift run) state instead of a row written by an earlier run.
    from src.mlops.drift_and_retrain import get_drift_db
    conn = get_drift_db(root_dir / "artifacts" / "drift_monitoring.db")
    try:
        conn.execute("DELETE FROM drift_runs")
        conn.commit()
    finally:
        conn.close()

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        resp = client.get(f"/api/v1/monitoring/drift?subset={sub}")
        assert resp.status_code == 200
        data = resp.json()

        assert "drift_alert" in data
        assert "share_drifting_features" in data
        # CRITICAL: Default state must not alarm
        assert data["drift_alert"] is False, f"Default state alarmed on {sub}: {data['share_drifting_features']}"
        assert data["share_drifting_features"] < 0.30

def test_drift_returns_no_data_without_monitored_features():
    """No monitorable features must be reported as no_data rather than a fake zero share."""
    from src.mlops.drift_and_retrain import get_drift_db

    subset = "FD001"
    mgr = MLOpsManager(subset)
    ref_df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2]})
    curr_df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2]})

    conn = get_drift_db(root_dir / "artifacts" / "drift_monitoring.db")
    try:
        before = conn.execute("SELECT COUNT(*) FROM drift_runs WHERE subset = ?", (subset,)).fetchone()[0]
    finally:
        conn.close()

    report = mgr.calculate_drift_report(ref_df, curr_df, persist=True)
    assert report["status"] == "no_data"
    assert report["monitored_features_count"] == 0
    assert report["drift_alert"] is False
    assert report["share_drifting_features"] is None

    conn = get_drift_db(root_dir / "artifacts" / "drift_monitoring.db")
    try:
        after = conn.execute("SELECT COUNT(*) FROM drift_runs WHERE subset = ?", (subset,)).fetchone()[0]
    finally:
        conn.close()
    assert after == before, "no_data reports must not be persisted as drift runs"

def test_drift_alarms_on_real_distribution_shift():
    """Verify that when actual operational shift occurs, drift alert is triggered."""
    subset = "FD001"
    mgr = MLOpsManager(subset)
    train_df = pd.read_parquet(root_dir / "data" / "processed" / subset / "train.parquet")
    test_df = pd.read_parquet(root_dir / "data" / "processed" / subset / "test.parquet")

    ref_df = train_df[(train_df["cycle"] >= 1) & (train_df["cycle"] <= 50)].copy()
    curr_df = test_df[(test_df["cycle"] >= 1) & (test_df["cycle"] <= 50)].copy()

    # Inject significant operational drift
    for col in curr_df.columns:
        if col.endswith("_dev_base") or col.endswith("_ewma"):
            curr_df[col] = curr_df[col] * 2.5 + 2.0

    report = mgr.calculate_drift_report(ref_df, curr_df)
    assert report["drift_alert"] is True
    assert report["share_drifting_features"] >= 0.30
