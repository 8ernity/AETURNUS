import os
import json
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.server import app, model_services, get_model_service
from src.mlops.drift_and_retrain import MLOpsManager
from src.models.model_service import ModelService
from src.common.utils import get_root_dir
import joblib
import pandas as pd

client = TestClient(app)
root_dir = get_root_dir()

def test_get_registry_is_strictly_read_only():
    mgr = MLOpsManager("FD001")
    reg_file = mgr.registry_file
    assert reg_file.exists()

    # Read original contents and mtime
    with open(reg_file, "r", encoding="utf-8") as f:
        original_content = f.read()
    original_mtime = os.path.getmtime(reg_file)

    # Allow slight time delta if any write would happen
    time.sleep(0.05)

    # Call GET /api/v1/mlops/registry and GET /api/v1/registry
    resp1 = client.get("/api/v1/mlops/registry?subset=FD001")
    assert resp1.status_code == 200
    data1 = resp1.json()

    resp2 = client.get("/api/v1/registry?subset=FD001")
    assert resp2.status_code == 200
    data2 = resp2.json()

    # File on disk must NOT be modified
    assert os.path.getmtime(reg_file) == original_mtime
    with open(reg_file, "r", encoding="utf-8") as f:
        current_content = f.read()
    assert current_content == original_content

    # Every registered version must have real metrics and model_path
    assert len(data1.get("models", [])) > 0
    for m in data1["models"]:
        assert "model_path" in m, f"Model version {m.get('version')} missing model_path"
        assert m["model_path"].endswith(".joblib")
        assert "metrics" in m, f"Model version {m.get('version')} missing metrics"
        assert m["metrics"]["rmse"] > 0
        assert m["metrics"]["mae"] > 0


def test_rollback_and_promote_swaps_bundle_and_reloads_cached_modelservice(tmp_path, monkeypatch):
    subset = "FD001"
    import shutil

    # Isolate test artifacts in tmp_path
    tmp_artifacts = tmp_path / "artifacts"
    tmp_models = tmp_artifacts / "models"
    tmp_reports = tmp_artifacts / "reports"
    tmp_models.mkdir(parents=True, exist_ok=True)
    tmp_reports.mkdir(parents=True, exist_ok=True)

    src_active = root_dir / "artifacts" / "models" / f"model_bundle_{subset}.joblib"
    src_reg = root_dir / "artifacts" / "reports" / "model_registry.json"
    shutil.copyfile(src_active, tmp_models / f"model_bundle_{subset}.joblib")
    shutil.copyfile(src_reg, tmp_reports / "model_registry.json")

    # Copy configs into tmp_path so ModelService can load models.yaml, etc.
    shutil.copytree(root_dir / "configs", tmp_path / "configs")

    # Monkeypatch ROOT_DIR / get_root_dir to isolate test operations completely
    monkeypatch.setattr("src.mlops.drift_and_retrain.get_root_dir", lambda: tmp_path)
    monkeypatch.setattr("src.models.model_service.get_root_dir", lambda: tmp_path)
    monkeypatch.setattr("src.api.server.ROOT_DIR", tmp_path)

    from src.api.server import mlops_managers
    mlops_managers[subset] = MLOpsManager(subset)
    mgr = mlops_managers[subset]
    active_bundle_path = tmp_models / f"model_bundle_{subset}.joblib"
    assert active_bundle_path.exists()

    # Create two distinct test model versions to test swapping
    v_a = "v_test_a"
    v_b = "v_test_b"

    path_a = tmp_models / f"model_bundle_{subset}_{v_a}.joblib"
    path_b = tmp_models / f"model_bundle_{subset}_{v_b}.joblib"

    try:
        ms_a = ModelService(subset)
        ms_a.load()
        ms_a.custom_test_marker = "MARKER_A"
        bundle_a = ms_a._to_bundle(version=v_a)
        bundle_a["test_marker"] = "MARKER_A"
        joblib.dump(bundle_a, path_a)

        ms_b = ModelService(subset)
        ms_b.load()
        bundle_b = ms_b._to_bundle(version=v_b)
        bundle_b["test_marker"] = "MARKER_B"
        joblib.dump(bundle_b, path_b)

        # Register both in registry
        mgr.register_model(version=v_a, alias="challenger")
        mgr.register_model(version=v_b, alias="challenger")

        # Step 1: Promote v_a
        resp_promote_a = client.post(f"/api/v1/models/{v_a}/promote?subset={subset}")
        assert resp_promote_a.status_code == 200
        assert resp_promote_a.json()["status"] == "promoted"
        assert resp_promote_a.json()["version"] == v_a

        # Verify active bundle on disk now has MARKER_A
        loaded_active_a = joblib.load(active_bundle_path)
        assert loaded_active_a.get("test_marker") == "MARKER_A"

        # Verify cached model_service was reloaded in server
        assert subset in model_services
        cached_bundle = model_services[subset]._to_bundle()
        assert cached_bundle.get("test_marker") == "MARKER_A"

        # Step 2: Promote v_b
        resp_promote_b = client.post(f"/api/v1/models/{v_b}/promote?subset={subset}")
        assert resp_promote_b.status_code == 200
        assert resp_promote_b.json()["status"] == "promoted"
        assert resp_promote_b.json()["version"] == v_b

        # Verify active bundle on disk now has MARKER_B
        loaded_active_b = joblib.load(active_bundle_path)
        assert loaded_active_b.get("test_marker") == "MARKER_B"

        # Verify cached model_service in server has MARKER_B
        cached_bundle_b = model_services[subset]._to_bundle()
        assert cached_bundle_b.get("test_marker") == "MARKER_B"

        # Step 3: Rollback to previous (which should be v_a)
        resp_rollback = client.post(f"/api/v1/models/rollback?subset={subset}&target_name={v_a}")
        assert resp_rollback.status_code == 200
        assert resp_rollback.json()["status"] == "rolled_back"
        assert resp_rollback.json()["version"] == v_a

        # Verify active bundle swapped back to MARKER_A
        loaded_active_rolled = joblib.load(active_bundle_path)
        assert loaded_active_rolled.get("test_marker") == "MARKER_A"

        # Verify cached model_service reloaded with MARKER_A
        cached_rolled = model_services[subset]._to_bundle()
        assert cached_rolled.get("test_marker") == "MARKER_A"

    finally:
        # Restore server caches
        from src.api.server import ROOT_DIR as PROD_ROOT
        mlops_managers[subset] = MLOpsManager(subset)
        model_services[subset] = ModelService(subset).load()

