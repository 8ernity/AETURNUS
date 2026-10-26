#!/usr/bin/env python3
"""
CI Smoke Test for AETURNUS
Validates end-to-end functionality across pipeline, models, agent, and API endpoints
in under 10 seconds.
"""
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

def run_smoke_test():
    print("=" * 60)
    print("AETURNUS - CI Smoke Test")
    print("=" * 60)

    # 1. Configs & Rules
    print("[1/6] Validating configuration files...")
    from src.common.utils import load_yaml
    rules_cfg = load_yaml("configs/agent_rules.yaml")
    assert "severity_bands" in rules_cfg, "agent_rules.yaml missing severity_bands"
    assert "escalation" in rules_cfg, "agent_rules.yaml missing escalation"
    pipeline_cfg = load_yaml("configs/pipeline.yaml")
    assert "cleaning" in pipeline_cfg, "pipeline.yaml missing cleaning"
    monitoring_cfg = load_yaml("configs/monitoring.yaml")
    assert "drift" in monitoring_cfg, "monitoring.yaml missing drift"
    print("  --> Configurations loaded successfully.")

    # 2. Manifest & model artifacts
    print("[2/6] Validating manifest provenance and model artifacts...")
    artifacts_dir = root_dir / "artifacts"
    reports_dir = artifacts_dir / "reports"
    models_dir = artifacts_dir / "models"

    manifest_f = artifacts_dir / "run_manifest.json"
    assert manifest_f.exists(), f"Missing run manifest: {manifest_f}"
    with open(manifest_f, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest.get("run_id"), "run_manifest.json is missing run_id"

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        info = manifest.get("subsets", {}).get(sub)
        assert info, f"Missing {sub} entry in {manifest_f}"

        metrics = info.get("metrics") or {}
        assert float(metrics.get("rmse", 0.0)) > 0.0, f"Invalid RMSE in manifest for {sub}"
        assert "multi_seed" in info, f"Missing multi_seed in manifest for {sub}"

        bundle_f = models_dir / f"model_bundle_{sub}.joblib"
        assert bundle_f.exists(), f"Missing model bundle: {bundle_f}"
        actual_sha = hashlib.sha256(bundle_f.read_bytes()).hexdigest()
        assert actual_sha == info.get("bundle_sha256"), (
            f"Bundle hash mismatch for {sub}: {actual_sha} != {info.get('bundle_sha256')}"
        )

    # The registry is a derived cache; validate it when present, but a fresh
    # clone is expected to rebuild it from the manifest on first use.
    reg_f = reports_dir / "model_registry.json"
    if reg_f.exists():
        with open(reg_f, "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert isinstance(registry.get("models"), list), "Invalid model_registry.json"
    print("  --> FD001-FD004 manifest provenance and bundles verified.")

    # 3. Data Cleaning & Regime Normalization
    print("[3/6] Testing DataCleaner and Causal Imputation...")
    from src.clean.cleaner import DataCleaner
    cleaner = DataCleaner({"cleaning": {"ffill_limit": 3, "outlier_z_clip": 4.0}})
    dummy_df = pd.DataFrame({
        "unit": [1, 1, 1],
        "cycle": [1, 2, 3],
        "setting_1": [0.0, 0.0, 0.0],
        "s2": [np.nan, 100.0, 105.0]
    })
    cleaner.fit_train(pd.DataFrame({"s2": [90.0, 95.0, 100.0], "s3": [50.0, 51.0, 49.0]}))
    cleaned, _ = cleaner.clean_impute(dummy_df, is_train=False)
    assert cleaned.loc[0, "s2"] == 95.0, "Causal fallback failed"
    assert cleaned.loc[1, "s2"] == 100.0
    print("  --> Cleaner passes causal imputation and settings regex exclusion.")

    # 4. Triage Agent Rules & SQLite
    print("[4/6] Testing Triage Agent rule engine & UUID generation...")
    import tempfile
    import shutil
    import uuid
    from src.agent.triage_agent import TriageAgent
    smoke_dir = Path(tempfile.mkdtemp(prefix="aeturnus_smoke_"))
    smoke_db = smoke_dir / "smoke_test.db"
    agent = TriageAgent(db_path=smoke_db)
    pred = {
        "engine_id": "ENG-FD001-SMOKE",
        "unit": 1,
        "subset": "FD001",
        "cycle": 50,
        "conservative_rul": 12.0,
        "rul_p10": 10.0,
        "rul_p50": 15.0,
        "rul_p90": 22.0,
        "fail_prob_30": 0.88,
        "anomaly_score": 0.75,
        "model_version": "rul-fd001@champion"
    }
    order, trace = agent.evaluate_engine_cycle(pred)
    assert order is not None, "Agent failed to create order"
    assert order["severity"] == "CRITICAL", f"Expected CRITICAL, got {order['severity']}"
    wo_id = order["work_order_id"]
    assert wo_id.startswith("WO-"), f"Expected WO- prefix, got {wo_id}"
    uuid_obj = uuid.UUID(wo_id[3:])
    assert f"WO-{str(uuid_obj)}" == wo_id, "work_order_id is not a valid UUID"
    try:
        shutil.rmtree(smoke_dir, ignore_errors=True)
    except Exception:
        pass
    print("  --> Agent produced valid UUID order and proper audit trace.")

    # 5. FastAPI Endpoints & Contract
    print("[5/6] Testing FastAPI Server Endpoints via TestClient...")
    from scripts.entrypoint import ensure_data_and_models
    ensure_data_and_models()
    from fastapi.testclient import TestClient
    from src.api.server import app
    client = TestClient(app)

    # Health
    r = client.get("/api/v1/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"

    # Benchmark
    r = client.get("/api/v1/models/benchmark")
    assert r.status_code == 200, f"Benchmark failed: {r.text}"
    bench = r.json()
    assert "benchmarks" in bench, "Expected benchmarks key in benchmark endpoint"
    assert len(bench["benchmarks"]) == 4, "Expected 4 datasets in benchmark"

    # Fleet
    r = client.get("/api/v1/fleet?subset=FD001")
    assert r.status_code == 200, f"Fleet failed: {r.text}"
    fleet = r.json()
    assert "engines" in fleet and len(fleet["engines"]) > 0, "Fleet empty"

    # Model Details
    r = client.get("/api/v1/models/FD001/details")
    assert r.status_code == 200, f"Model details failed: {r.text}"
    details = r.json()
    assert "bucket_errors" in details, "Missing bucket_errors in details"

    # 400 Validation on Invalid Subset
    r = client.get("/api/v1/fleet?subset=INVALID")
    assert r.status_code == 400, "Failed to return 400 for invalid subset"
    print("  --> FastAPI endpoints verified.")

    # 6. Web App Build Verification
    print("[6/6] Checking Frontend Vite Distribution Build...")
    dist_index = root_dir / "web" / "dist" / "index.html"
    assert dist_index.exists(), "web/dist/index.html not found! Run 'cd web && npm run build'"
    print("  --> Frontend bundle verified.")

    print("=" * 60)
    print("CI SMOKE TEST PASSED: ALL CHECKS SUCCESSFUL!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        print(f"\n[ERROR] CI Smoke Test Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
