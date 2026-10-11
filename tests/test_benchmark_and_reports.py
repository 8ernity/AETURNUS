import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.server import app
from src.mlops.drift_and_retrain import MLOpsManager
from src.common.utils import get_root_dir

client = TestClient(app)
root_dir = get_root_dir()

def test_benchmark_endpoint_reads_disk_reports():
    response = client.get("/api/v1/models/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data
    benchmarks = data["benchmarks"]
    assert len(benchmarks) == 4
    
    subsets = [b["subset"] for b in benchmarks]
    assert subsets == ["FD001", "FD002", "FD003", "FD004"]
    
    # Verify each benchmark matches the report file on disk
    for b in benchmarks:
        sub = b["subset"]
        report_file = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        assert report_file.exists(), f"Evaluation report for {sub} must exist on disk"
        with open(report_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        
        assert b["rmse"] == disk_data["rmse"]
        assert b["nasa_score"] == disk_data["nasa_score"]
        assert b["pr_auc"] == disk_data["pr_auc"]
        assert b["status"] == disk_data["status"]
        assert b["condition"] == disk_data["condition"]
        assert b["faults"] == disk_data["faults"]

    # Verify honest reporting of targets matching disk reports
    status_map = {b["subset"]: b["status"] for b in benchmarks}
    assert status_map["FD001"] == "Target Met"
    assert status_map["FD002"] == "Target Missed"
    assert status_map["FD003"] in ["Target Met", "Target Missed"]
    assert status_map["FD004"] == "Target Missed"

def test_model_details_endpoint_returns_real_metrics():
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        response = client.get(f"/api/v1/models/{sub}/details")
        assert response.status_code == 200
        data = response.json()
        assert data["subset"] == sub
        assert "bucket_errors" in data
        assert len(data["bucket_errors"]) == 3
        for b in data["bucket_errors"]:
            assert "bucket" in b
            assert "rmse" in b
            assert "mae" in b
            assert "sample_count" in b
            assert b["rmse"] >= 0
            assert b["sample_count"] > 0
        
        assert "feature_importance" in data
        assert len(data["feature_importance"]) > 0
        for feat in data["feature_importance"]:
            assert "feat" in feat
            assert "weight" in feat

def test_data_quality_endpoint_returns_real_audit():
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        response = client.get(f"/api/v1/data/quality?subset={sub}")
        assert response.status_code == 200
        data = response.json()
        assert data["subset"] == sub
        assert "train_validation" in data
        assert "train_cleaning" in data
        assert data["train_validation"]["valid_rows"] > 0
        assert data["kept_sensors_count"] > 0
        assert data["feature_count"] > 0
        assert "kept_sensors" in data
        # Flat sensors dropped are tracked
        dropped = data["train_cleaning"]["sensors_dropped"]
        assert isinstance(dropped, list)
        if sub in ["FD001", "FD003"]:
            assert len(dropped) > 0

def test_registry_has_real_hash_and_metrics():
    mgr = MLOpsManager("FD001")
    reg = mgr.get_registry()
    assert "models" in reg
    champion = [m for m in reg["models"] if m.get("alias") == "champion"][0]
    
    # Real SHA-256 data hash
    data_hash = champion["data_hash"]
    assert data_hash.startswith("sha256:")
    assert len(data_hash) == len("sha256:") + 16
    assert data_hash != "sha256-c-mapss-fd001-train-v1"
    
    # Real metrics matching evaluation report for initial v1.0 baseline
    v1_model = [m for m in reg["models"] if m.get("version") == "v1.0"][0]
    report_file = root_dir / "artifacts" / "reports" / "evaluation_FD001.json"
    with open(report_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert v1_model["metrics"]["rmse"] == disk_data["rmse"]
    assert v1_model["metrics"]["nasa_score"] == disk_data["nasa_score"]

def test_bundle_rmse_within_threshold_of_evaluation_json():
    """
    Evaluates the shipped model bundles and asserts that their test RMSE
    is within 0.05 of the values saved in evaluation_*.json.
    """
    from src.models.model_service import ModelService
    from src.ingest.loader import CMAPSSLoader
    import pandas as pd

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        eval_path = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        assert eval_path.exists(), f"Evaluation report missing for {sub}"
        with open(eval_path, "r", encoding="utf-8") as f:
            disk_eval = json.load(f)
        disk_rmse = disk_eval["rmse"]

        # Evaluate the shipped active bundle
        ms = ModelService(subset=sub).load()
        assert hasattr(ms, "sklearn_version")
        test_path = root_dir / "data" / "processed" / sub / "test.parquet"
        if not test_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset=sub).run()
            dp = DataPipeline(subset=sub, inject_faults=False)
            dp.run()
        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        bundle_eval = ms.evaluate_test(test_df, rul_truth, save_report=False)
        bundle_rmse = bundle_eval["rmse"]

        diff = abs(bundle_rmse - disk_rmse)
        assert diff <= 0.05, f"[{sub}] Shipped bundle RMSE ({bundle_rmse:.4f}) differs from report ({disk_rmse:.4f}) by {diff:.4f} > 0.05"

