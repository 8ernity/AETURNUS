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
    
    # Verify each benchmark matches the manifest
    from tests.helpers import load_manifest
    manifest = load_manifest()
    for b in benchmarks:
        sub = b["subset"]
        sub_manifest = manifest["subsets"][sub]
        sub_metrics = sub_manifest["metrics"]
        assert b["rmse"] == pytest.approx(sub_metrics["rmse"], rel=1e-4)
        assert b["nasa_score"] == pytest.approx(sub_metrics["nasa_score"], rel=1e-4)
        assert b["pr_auc"] == pytest.approx(sub_metrics["pr_auc"], rel=1e-4)
        assert b["target_met"] == sub_manifest["target_met"]

    # Verify honest reporting of targets matching manifest
    status_map = {b["subset"]: b["target_met"] for b in benchmarks}
    assert isinstance(status_map["FD001"], bool)

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
    
    from tests.helpers import load_manifest
    manifest = load_manifest()
    sub_data = manifest["subsets"]["FD001"]

    bundle_sha256 = champion["bundle_sha256"]
    assert len(bundle_sha256) == 64
    assert bundle_sha256 == sub_data["bundle_sha256"]

    # Real metrics matching manifest
    v1_model = [m for m in reg["models"] if m.get("version") == "v1.0"][0]
    assert v1_model["metrics"]["rmse"] == pytest.approx(sub_data["metrics"]["rmse"], rel=1e-4)
    assert v1_model["metrics"]["nasa_score"] == pytest.approx(sub_data["metrics"]["nasa_score"], rel=1e-4)

def test_bundle_rmse_within_threshold_of_evaluation_json():
    """
    Evaluates the shipped model bundles and asserts that their test RMSE
    is within 0.05 of the values saved in run_manifest.json.
    """
    from src.models.model_service import ModelService
    from src.ingest.loader import CMAPSSLoader
    from tests.helpers import load_manifest
    import pandas as pd

    manifest = load_manifest()
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        manifest_rmse = manifest["subsets"][sub]["metrics"]["rmse"]

        # Evaluate the shipped active bundle
        ms = ModelService(subset=sub).load()
        assert hasattr(ms, "sklearn_version")
        test_path = root_dir / "data" / "processed" / sub / "test.parquet"
        if not test_path.exists():
            from src.features.pipeline import DataPipeline
            dp = DataPipeline(subset=sub, inject_faults=False)
            dp.run()
        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        bundle_eval = ms.evaluate_test(test_df, rul_truth, save_report=False)
        bundle_rmse = bundle_eval["rmse"]

        diff = abs(bundle_rmse - manifest_rmse)
        assert diff <= 0.05, f"[{sub}] Shipped bundle RMSE ({bundle_rmse:.4f}) differs from manifest ({manifest_rmse:.4f}) by {diff:.4f} > 0.05"

