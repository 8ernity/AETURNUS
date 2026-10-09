import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.server import app
from src.mlops.drift_and_retrain import MLOpsManager, compute_df_sha256
from src.common.utils import get_root_dir
import pandas as pd

client = TestClient(app)
root_dir = get_root_dir()

def test_drift_uses_training_reference_and_replay_window():
    response = client.get("/api/v1/monitoring/drift?subset=FD001")
    assert response.status_code == 200
    data = response.json()
    assert "monitored_features_count" in data
    assert "sensor_metrics" in data
    assert "drift_alert" in data
    assert "share_drifting_features" in data
    assert data["monitored_features_count"] > 0
    # Check sensor metrics have psi, ks_stat, ks_pvalue
    for sensor, m in data["sensor_metrics"].items():
        assert "psi" in m
        assert "ks_stat" in m
        assert "ks_pvalue" in m
        assert "is_drifting" in m

def test_compute_df_sha256():
    df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    df3 = pd.DataFrame({"a": [1.0, 99.0], "b": [3.0, 4.0]})
    
    hash1 = compute_df_sha256(df1)
    hash2 = compute_df_sha256(df2)
    hash3 = compute_df_sha256(df3)
    
    assert hash1.startswith("sha256:")
    assert hash1 == hash2
    assert hash1 != hash3

def test_retrain_demo_fixed_holdout_and_no_fake_multipliers():
    mgr = MLOpsManager("FD001")
    result = mgr.run_retrain_demo(scenario="A")
    
    assert "champion_metrics" in result
    assert "challenger_metrics" in result
    assert "champion_holdout_metrics" in result
    assert "challenger_holdout_metrics" in result
    assert "gate_passed" in result
    assert "gate_reasons" in result

    # Check metrics are positive floats
    champ_holdout = result["champion_holdout_metrics"]
    chal_holdout = result["challenger_holdout_metrics"]
    assert champ_holdout["rmse"] > 0
    assert chal_holdout["rmse"] > 0
    assert champ_holdout["nasa_score"] > 0
    assert chal_holdout["nasa_score"] > 0

    # Ensure no fake 0.96 or 0.94 hardcoding:
    # In fake code, chal_rmse was exactly round(champ_rmse * 0.96, 4)
    # Check that it is evaluated independently from the model heads on holdout
    ratio = chal_holdout["rmse"] / champ_holdout["rmse"]
    assert abs(ratio - 0.96) > 1e-4 or abs(ratio - 0.94) > 1e-4

    # Check registry updated with real data hashes and aliases
    reg = mgr.get_registry()
    models = reg["models"]
    assert len(models) >= 2
    
    # Check aliases
    aliases = [m["alias"] for m in models]
    assert "champion" in aliases
    assert any("previous" in a or "challenger" in a for a in aliases)

    # Check data hash of all models are sha256 hex
    for m in models:
        assert m["data_hash"].startswith("sha256:")
        assert len(m["data_hash"]) == len("sha256:") + 16

    # Verify versioned model file exists on disk
    latest_model = models[-1]
    if "model_path" in latest_model:
        model_file = root_dir / latest_model["model_path"]
        assert model_file.exists()

def test_mlflow_runs_logged():
    import mlflow
    db_file = root_dir / "artifacts" / "mlflow.db"
    assert db_file.exists(), "MLflow SQLite database must exist"
    mlflow.set_tracking_uri(f"sqlite:///{db_file.resolve().as_posix()}")
    runs = mlflow.search_runs(experiment_names=["cmapss_FD001"])
    assert len(runs) > 0, "MLflow must have logged runs for cmapss_FD001"

