import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pytest

from src.common.utils import get_root_dir
from src.ingest.loader import CMAPSSLoader

ROOT_DIR = get_root_dir()

def test_manifest_and_provenance():
    manifest_path = ROOT_DIR / "artifacts" / "run_manifest.json"
    assert manifest_path.exists(), "artifacts/run_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Required top-level fields
    for field in ["run_id", "git_sha", "config_hash", "data_sha256", "subsets"]:
        assert field in manifest, f"Manifest missing top-level field: {field}"

    # Verify data_sha256 is full 64-char sha256 hash (not partial)
    data_hashes = manifest["data_sha256"]
    assert isinstance(data_hashes, dict) and len(data_hashes) > 0
    for k, h in data_hashes.items():
        assert len(h) == 64, f"Hash for {k} must be full 64-char SHA256, got: {h}"

    from src.eval.protocol import (
        calculate_metrics,
        get_protocol_id,
        sample_truncation,
    )

    subsets = manifest["subsets"]
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        assert sub in subsets, f"Manifest missing subset {sub}"
        sub_info = subsets[sub]
        
        # Verify subset fields
        for field in ["bundle_sha256", "algorithm", "protocol_id", "n_engines", "metrics", "cv", "comparison_rows", "champion"]:
            assert field in sub_info, f"Subset {sub} missing field: {field}"

        bundle_path = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        assert bundle_path.exists(), f"Bundle {bundle_path} must exist"

        # Check bundle SHA256 equality
        bundle_bytes = bundle_path.read_bytes()
        actual_bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
        assert sub_info["bundle_sha256"] == actual_bundle_sha, (
            f"Bundle SHA mismatch for {sub}: manifest={sub_info['bundle_sha256']}, disk={actual_bundle_sha}"
        )

        bundle = joblib.load(bundle_path)
        served_algo = bundle.get("algorithm", bundle.get("model_name", ""))
        champ_algo = sub_info["champion"]["selected_algorithm"]
        assert champ_algo == served_algo, (
            f"Champion in manifest ({champ_algo}) != served bundle algorithm ({served_algo}) for {sub}"
        )

        # Assert cv and test protocols resolve to the same protocol_id
        cv_protocol_id = sub_info["cv"].get("protocol_id")
        test_protocol_id = sub_info.get("protocol_id")
        assert cv_protocol_id == test_protocol_id, (
            f"CV protocol_id ({cv_protocol_id}) != test protocol_id ({test_protocol_id}) for {sub}"
        )

        # Recompute test metrics from processed data using protocol.py and assert equality (atol 1e-6)
        test_path = ROOT_DIR / "data" / "processed" / sub / "test.parquet"
        assert test_path.exists(), f"Missing processed test data for {sub}"
        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        eval_rows = sample_truncation(test_df, mode="last")
        feature_cols = bundle["feature_cols"]
        X_test = eval_rows[feature_cols].values
        
        # Predict using bundle model
        p50_model = bundle.get("rul_regressor") or bundle.get("rul_p50_model") or bundle.get("model")
        y_pred = np.maximum(0.0, p50_model.predict(X_test))
        recomputed = calculate_metrics(rul_truth, y_pred)

        manifest_metrics = sub_info["metrics"]
        for m_key in ["rmse", "mae", "nasa_score"]:
            assert m_key in manifest_metrics, f"Metric {m_key} missing from manifest for {sub}"
            diff = abs(recomputed[m_key] - manifest_metrics[m_key])
            assert diff <= 1e-6, (
                f"Metric {m_key} mismatch for {sub}: recomputed={recomputed[m_key]}, manifest={manifest_metrics[m_key]}, diff={diff}"
            )
