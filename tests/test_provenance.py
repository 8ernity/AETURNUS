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
        DEFAULT_SEED,
    )
    from src.models.model_service import (
        algorithm_for_p50_model,
        load_cv_protocol,
    )

    cv_protocol = load_cv_protocol()
    expected_cv_protocol_id = get_protocol_id(
        mode=cv_protocol["mode"],
        seed=DEFAULT_SEED,
        n_draws=cv_protocol["n_draws"],
    )

    subsets = manifest["subsets"]
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        assert sub in subsets, f"Manifest missing subset {sub}"
        sub_info = subsets[sub]
        
        # Verify subset fields
        for field in [
            "bundle_sha256", "algorithm", "p50_model_class", "protocol_id",
            "cv_protocol_id", "n_engines", "metrics", "cv", "comparison_rows", "champion",
        ]:
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
        p50_model = bundle["rul_p50_model"]
        served_algo = algorithm_for_p50_model(p50_model)
        champ_algo = sub_info["champion"]["selected_algorithm"]
        assert champ_algo == served_algo == sub_info["algorithm"], (
            f"Champion ({champ_algo}), manifest algorithm ({sub_info['algorithm']}), and "
            f"served bundle class ({type(p50_model).__name__} -> {served_algo}) disagree for {sub}"
        )
        assert sub_info["p50_model_class"] == type(p50_model).__name__

        # CV uses the censored test-like protocol; test metrics use the canonical last-cycle protocol.
        cv_protocol_id = sub_info["cv"]["protocol_id"]
        test_protocol_id = sub_info["protocol_id"]
        assert cv_protocol_id == sub_info["cv_protocol_id"] == expected_cv_protocol_id, (
            f"CV protocol_id {cv_protocol_id} does not match the configured censored protocol "
            f"{expected_cv_protocol_id} for {sub}"
        )
        assert sub_info["cv"]["evaluation_mode"] == cv_protocol["mode"]
        assert sub_info["cv"]["n_draws"] == cv_protocol["n_draws"]
        assert sub_info["cv"]["min_cycles"] == cv_protocol["min_cycles"]
        assert cv_protocol_id != test_protocol_id, (
            f"CV and test protocols must differ for {sub}; CV was not censored"
        )

        # Censored CV must be in the same plausible band as the held-out test RMSE.
        cv_rmse = float(sub_info["cv"]["mean_rmse"])
        test_rmse = float(sub_info["metrics"]["rmse"])
        assert 0.4 <= cv_rmse / test_rmse <= 2.5, (
            f"CV RMSE {cv_rmse:.2f} is implausibly far from test RMSE {test_rmse:.2f} for {sub}"
        )

        # Champion must be the lowest-CV servable candidate.
        servable_cv = {
            row["model_name"]: float(row["cv_mean_rmse"])
            for row in sub_info["comparison_rows"]
            if row.get("servable", True) and row.get("cv_mean_rmse") is not None
        }
        selected_name = sub_info["champion"]["model_name"]
        assert selected_name in servable_cv
        assert servable_cv[selected_name] == pytest.approx(min(servable_cv.values())), (
            f"Champion {selected_name} has CV {servable_cv[selected_name]}, "
            f"not the minimum servable CV {min(servable_cv.values())} for {sub}"
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
        
        # Predict using the served p50 bundle model
        y_pred = np.maximum(0.0, p50_model.predict(X_test))
        recomputed = calculate_metrics(rul_truth, y_pred)

        manifest_metrics = sub_info["metrics"]
        for m_key in ["rmse", "mae", "nasa_score"]:
            assert m_key in manifest_metrics, f"Metric {m_key} missing from manifest for {sub}"
            diff = abs(recomputed[m_key] - manifest_metrics[m_key])
            assert diff <= 1e-6, (
                f"Metric {m_key} mismatch for {sub}: recomputed={recomputed[m_key]}, manifest={manifest_metrics[m_key]}, diff={diff}"
            )
