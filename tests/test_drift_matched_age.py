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

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        resp = client.get(f"/api/v1/monitoring/drift?subset={sub}")
        assert resp.status_code == 200
        data = resp.json()

        assert "drift_alert" in data
        assert "share_drifting_features" in data
        # CRITICAL: Default state must not alarm
        assert data["drift_alert"] is False, f"Default state alarmed on {sub}: {data['share_drifting_features']}"
        assert data["share_drifting_features"] < 0.30

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
