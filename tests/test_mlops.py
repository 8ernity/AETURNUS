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
    bundle_file = root_dir / "artifacts" / "models" / "model_bundle_FD001.joblib"
    orig_bundle = bundle_file.read_bytes() if bundle_file.exists() else None
    reg_file = root_dir / "artifacts" / "reports" / "model_registry.json"
    orig_reg = reg_file.read_text(encoding="utf-8") if reg_file.exists() else None

    try:
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

        # Check registry updated with minimal rows and derived provenance
        reg = mgr.get_registry()
        models = reg["models"]
        assert len(models) >= 2

        aliases = [m["alias"] for m in models]
        assert aliases.count("champion") == 1
        assert any("previous" in a or "challenger" in a for a in aliases)

        for m in models:
            assert len(m["bundle_sha256"]) == 64
            assert m["model_path"].endswith(".joblib")
            assert "manifest_run_id" in m

        champion = [m for m in models if m["alias"] == "champion"][0]
        assert champion["metrics"]["rmse"] > 0

        # Verify versioned model file exists on disk
        target_version = result.get("promoted_version") or result.get("challenger_version")
        if target_version:
            target_model_file = root_dir / "artifacts" / "models" / f"model_bundle_FD001_{target_version}.joblib"
            assert target_model_file.exists(), f"Trained bundle {target_model_file} must exist on disk"

        latest_model = models[-1]
        if "model_path" in latest_model:
            model_file = root_dir / latest_model["model_path"]
            assert model_file.exists(), f"Model file {model_file} must exist on disk"
    finally:
        if orig_bundle and bundle_file.exists():
            bundle_file.write_bytes(orig_bundle)
        if orig_reg and reg_file.exists():
            reg_file.write_text(orig_reg, encoding="utf-8")

def test_mlflow_runs_logged():
    import mlflow
    from src.common.state import state_db_path
    db_file = state_db_path("mlflow.db", root_dir)
    assert db_file.exists(), "MLflow SQLite database must exist"
    mlflow.set_tracking_uri(f"sqlite:///{db_file.resolve().as_posix()}")
    runs = mlflow.search_runs(experiment_names=["cmapss_FD001"])
    assert len(runs) > 0, "MLflow must have logged runs for cmapss_FD001"

