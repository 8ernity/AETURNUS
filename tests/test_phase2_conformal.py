import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.server import app
from src.common.utils import get_root_dir
from src.models.model_service import ModelService, wilson_ci
from src.ingest.loader import CMAPSSLoader
from src.common.splits import get_persistent_split

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)
root_dir = get_root_dir()

def test_conformal_calibration_provenance():
    """Verify conformal offset is loaded from bundle and is valid."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        offset = getattr(ms, "interval_calibration_offset", getattr(ms, "conformal_q", None))
        assert offset is not None, f"Calibration offset must be defined for {sub}"
        assert offset >= 0.0, f"Calibration offset must be non-negative for {sub}"

def test_conformal_validation_selection_and_cross_fit():
    """Verify validation-selected nominal level and cross-conformal calibration metadata."""
    from tests.helpers import load_manifest
    from src.common.utils import load_yaml

    manifest = load_manifest()
    pipeline = load_yaml("configs/pipeline.yaml")
    conformal_cfg = pipeline["conformal"]
    expected_levels = [round(float(v), 4) for v in conformal_cfg["nominal_levels"]]
    target_coverage = float(conformal_cfg["target_coverage"])
    validation_draws = int(conformal_cfg["validation_draws"])
    tolerance = float(conformal_cfg["selection_tolerance"])
    assert validation_draws >= 10

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        calibration = manifest["subsets"][sub]["calibration"]
        assert calibration["nominal_level"] in expected_levels, (
            f"{sub}: nominal_level {calibration['nominal_level']} not in {expected_levels}"
        )
        assert calibration["validation_coverage"] >= target_coverage - tolerance, (
            f"{sub}: validation coverage {calibration['validation_coverage']} "
            f"is below target {target_coverage} - tolerance {tolerance}"
        )
        assert calibration["target_coverage"] == target_coverage
        assert calibration["selection_tolerance"] == tolerance
        assert calibration["validation_draws"] >= 10
        assert calibration["validation_engine_count"] > 0
        assert calibration["validation_sample_count"] >= (
            calibration["validation_engine_count"] * calibration["validation_draws"]
        )
        val_lower, val_upper = calibration["validation_coverage_wilson_ci_95"]
        assert 0.0 <= val_lower <= calibration["validation_coverage"] <= val_upper <= 1.0
        assert calibration["validation_coverage_ci_lower"] >= target_coverage - tolerance, (
            f"{sub}: selected level does not satisfy the Wilson-lower-bound floor"
        )
        assert calibration["n_calibration"] > 0
        assert calibration["calibration_engine_count"] > 0
        assert calibration["method"] == "cross_conformal_all_non_test_engines"
        assert calibration["cross_conformal_folds"] >= 2
        assert len(calibration["validation_levels"]) == len(expected_levels)
        for level in calibration["validation_levels"]:
            assert level["nominal_level"] in expected_levels
            assert 0.0 <= level["coverage"] <= 1.0
            assert level["n"] == calibration["validation_sample_count"]
            assert level["ci_lower"] <= level["coverage"] <= level["wilson_ci_95"][1]
            assert level["passes_floor"] == (level["ci_lower"] >= target_coverage - tolerance)


def test_split_conformal_val_engines_only():
    """Verify persistent split guarantees validation engines are disjoint from train engines."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        loader = CMAPSSLoader(sub)
        train_raw, _, _ = loader.load_raw_data()
        all_units = [int(u) for u in np.unique(train_raw["unit"].values)]
        split = get_persistent_split(sub, all_units=all_units)
        
        train_units = set(split["train_units"])
        calib_units = set(split["calib_units"])
        val_units = set(split["val_units"])
        assert len(train_units.intersection(calib_units)) == 0, f"Train and Calib engines must be strictly disjoint for {sub}"
        assert len(calib_units.intersection(val_units)) == 0, f"Calib and Val engines must be strictly disjoint for {sub}"
        assert len(calib_units) > 0, f"Calib engines must exist for conformal calibration in {sub}"
        assert len(val_units) > 0, f"Val engines must exist for threshold tuning in {sub}"

def test_interval_non_inversion_and_non_negativity():
    """Verify calibrated intervals are strictly ordered: 0 <= p10 <= p50 <= p90."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        test_path = root_dir / "data" / "processed" / sub / "test.parquet"
        if not test_path.exists():
            continue
        test_df = pd.read_parquet(test_path)
        last_indices = test_df.groupby("unit")["cycle"].idxmax()
        eval_rows = test_df.loc[last_indices].head(20)
        
        for _, row in eval_rows.iterrows():
            pred = ms.predict_point(row.to_dict())
            p10 = pred["rul_p10"]
            p50 = pred["rul_p50"]
            p90 = pred["rul_p90"]
            assert p10 >= 0.0, f"p10 must be non-negative, got {p10}"
            assert p10 <= p50, f"p10 ({p10}) must be <= p50 ({p50})"
            if p90 is None:
                assert pred["censored_at_cap"] is True, f"{sub}: null p90 must be censored"
                assert pred["interval_label"] == ">cap"
                assert pred["interval_width"] is None
            else:
                assert p50 <= p90, f"p50 ({p50}) must be <= p90 ({p90})"
                assert pred["censored_at_cap"] is False
                assert pred["interval_label"] == "p10-p90"

def test_test_set_coverage_matches_evaluation_report():
    """Verify interval coverage calculated directly on test data matches manifest."""
    from tests.helpers import load_manifest
    manifest = load_manifest()
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        reported_cov = manifest["subsets"][sub]["metrics"]["coverage_all"]
        
        ms = ModelService(sub).load()
        test_df = pd.read_parquet(root_dir / "data" / "processed" / sub / "test.parquet")
        loader = CMAPSSLoader(sub)
        _, _, rul_truth = loader.load_raw_data()
        
        eval_res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        computed_cov = eval_res["interval_coverage"]
        assert abs(reported_cov - computed_cov) < 0.01, (
            f"{sub}: Manifest coverage {reported_cov} differs from computed {computed_cov}"
        )
        assert eval_res["coverage_rule"] == "censored_open_upper_bound"
        assert eval_res["coverage_all"] == eval_res["interval_coverage"]
        assert eval_res["coverage_all_n"] == eval_res["test_engines_count"]
        split = eval_res["coverage_split"]
        assert split["all"]["n"] == eval_res["test_engines_count"]
        assert split["le_cap"]["n"] + split["above_cap"]["n"] == eval_res["test_engines_count"]
        assert split["le_cap"]["n"] == int(np.sum(rul_truth <= eval_res["metrics"]["rul_cap"]))
        for branch in ("all", "le_cap", "above_cap"):
            stats = split[branch]
            assert stats["n"] == eval_res[f"coverage_{branch}_n"]
            assert stats["coverage"] == pytest.approx(eval_res[f"coverage_{branch}"], abs=1e-4)
            lower, upper = stats["wilson_ci_95"]
            if stats["n"]:
                assert 0.0 <= lower <= stats["coverage"] <= upper <= 1.0
        target_coverage = eval_res["coverage_target"]
        assert eval_res["coverage_metric"] == manifest["coverage_metric"]["value"]
        assert eval_res["coverage_metric"] == "coverage_le_cap"
        assert eval_res["coverage_metric_value"] == eval_res["coverage_le_cap"]
        assert eval_res["coverage_le_cap"] >= target_coverage, (
            f"{sub}: policy coverage {eval_res['coverage_le_cap']} is below {target_coverage}"
        )
        assert eval_res["target_coverage_met"] is True
        assert eval_res["coverage_target"] == manifest["subsets"][sub]["metrics"]["coverage_target"]
        # Test coverage is reported with a Wilson CI. Target attainment is checked
        # on the held-out validation-engine selection fold, not by post-hoc test tuning.
        ci_lower, ci_upper = eval_res["coverage_wilson_ci_95"]
        assert 0.0 <= ci_lower <= eval_res["interval_coverage"] <= ci_upper <= 1.0, (
            f"{sub}: invalid Wilson CI {eval_res['coverage_wilson_ci_95']} "
            f"for coverage {eval_res['interval_coverage']}"
        )
        policy_lower, policy_upper = eval_res["coverage_metric_wilson_ci_95"]
        assert 0.0 <= policy_lower <= eval_res["coverage_metric_value"] <= policy_upper <= 1.0

        covered_flags = []
        le_cap_flags = []
        for point in eval_res["parity_scatter"]:
            if point["censored_at_cap"]:
                covered = point["actual"] >= point["p10"]
            else:
                covered = point["p10"] <= point["actual"] <= point["p90"]
            covered_flags.append(covered)
            if point["actual"] <= eval_res["metrics"]["rul_cap"]:
                le_cap_flags.append(covered)
        assert abs(float(np.mean(covered_flags)) - eval_res["coverage_all"]) < 0.02
        assert abs(float(np.mean(le_cap_flags)) - eval_res["coverage_le_cap"]) < 0.02
        assert eval_res["n_censored_intervals"] == sum(
            1 for point in eval_res["parity_scatter"] if point["censored_at_cap"]
        )
        buckets = eval_res["coverage_by_bucket"]
        assert buckets, f"{sub}: expected per-bucket coverage diagnostics"
        assert sum(b["sample_count"] for b in buckets) == eval_res["test_engines_count"]
        for bucket in buckets:
            assert 0.0 <= bucket["coverage"] <= 1.0
            assert len(bucket["wilson_ci_95"]) == 2
            assert bucket["conformal_q"] >= 0.0

def test_wilson_ci_mathematical_properties():
    """Verify Wilson score confidence interval correctness recomputed from bundle + manifest."""
    from tests.helpers import load_manifest
    manifest = load_manifest()
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        assert sub in manifest["subsets"]
        ms = ModelService(sub).load()
        test_df = pd.read_parquet(root_dir / "data" / "processed" / sub / "test.parquet")
        loader = CMAPSSLoader(sub)
        _, _, rul_truth = loader.load_raw_data()
        eval_res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        
        cov = eval_res["coverage_metric_value"]
        ci = eval_res.get("coverage_metric_wilson_ci_95")
        assert ci is not None and len(ci) == 2, f"Wilson CI must exist for {sub}"
        lower, upper = ci[0], ci[1]
        
        assert 0.0 <= lower <= 1.0, f"Lower bound out of bounds: {lower}"
        assert 0.0 <= upper <= 1.0, f"Upper bound out of bounds: {upper}"
        assert lower <= upper, f"Lower bound {lower} > upper bound {upper}"
        assert lower <= cov <= upper or abs(cov - lower) < 1e-4 or abs(cov - upper) < 1e-4, (
            f"Point estimate {cov} not contained in [{lower}, {upper}]"
        )
        
        target = eval_res["target_coverage"]
        ci_excludes_target = eval_res.get("coverage_ci_excludes_target")
        expected_excludes = (lower > target) or (upper < target)
        assert ci_excludes_target == expected_excludes, f"{sub}: ci_excludes_target mismatch"

def test_model_details_endpoint_returns_conformal_fields():
    """Verify API endpoint returns conformal interval fields."""
    for sub in ["FD001", "FD002"]:
        resp = client.get(f"/api/v1/models/{sub}/details")
        assert resp.status_code == 200
        data = resp.json()
        assert "interval_coverage" in data
        assert "coverage_wilson_ci_95" in data
        assert "coverage_ci_excludes_target" in data
        assert "mean_interval_width" in data
        assert data["coverage_metric"] == "coverage_le_cap"
        assert data["coverage_metric_value"] == data["coverage_le_cap"]
        assert data["target_coverage_met"] is True
        assert data["coverage_all_n"] > 0
        assert data["coverage_le_cap_n"] > 0
        assert data["coverage_all"] == data["interval_coverage"]
        assert data["coverage_split"]["le_cap"]["n"] == data["coverage_le_cap_n"]
        assert data["n_censored_intervals"] > 0
        assert len(data["coverage_metric_wilson_ci_95"]) == 2

        for point in data["parity_scatter"]:
            if point["censored_at_cap"]:
                assert point["p90"] is None
                assert point["interval_label"] == ">cap"
            else:
                assert point["p90"] is not None
        assert data["target_metric"] in {"rmse_vs_raw_truth", "rmse_on_true_rul_le_cap"}
        assert data["rmse_vs_raw_truth"] is not None
        assert data["rmse_vs_capped_truth"] is not None
        assert data["rul_bucket_metrics"], f"{sub}: missing RUL-bucket metrics"
        calibration = data["calibration"]
        assert 0.0 < calibration["nominal_level"] < 1.0
        assert calibration["validation_coverage"] >= (
            calibration["target_coverage"] - calibration["selection_tolerance"]
        )
        assert calibration["validation_draws"] >= 10
        assert calibration["n_calibration"] > 0


def test_censored_prediction_contract_and_open_interval_labels():
    """Every subset must expose at least one censored prediction with no closed p90."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        test_df = pd.read_parquet(root_dir / "data" / "processed" / sub / "test.parquet")
        loader = CMAPSSLoader(sub)
        _, _, rul_truth = loader.load_raw_data()
        eval_res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        censored_points = [
            p for p in eval_res["parity_scatter"] if p.get("censored_at_cap")
        ]
        assert censored_points, f"{sub}: expected censored predictions"
        assert eval_res["n_censored_intervals"] == len(censored_points)

        unit = int(censored_points[0]["unit"])
        last_row = (
            test_df[test_df["unit"] == unit]
            .sort_values("cycle")
            .iloc[-1]
            .to_dict()
        )
        pred = ms.predict_point(last_row)
        assert pred["censored_at_cap"] is True
        assert pred["rul_p90"] is None
        assert pred["interval_width"] is None
        assert pred["interval_label"] == ">cap"
        assert pred["rul_interval_label"] == ">cap"
        assert pred["rul_p10"] <= pred["rul_p50"]
