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

client = TestClient(app)
root_dir = get_root_dir()

def test_conformal_calibration_provenance():
    """Verify conformal offset is loaded from bundle and is valid."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        offset = getattr(ms, "interval_calibration_offset", getattr(ms, "conformal_q", None))
        assert offset is not None, f"Calibration offset must be defined for {sub}"
        assert offset >= 0.0, f"Calibration offset must be non-negative for {sub}"

def test_split_conformal_val_engines_only():
    """Verify persistent split guarantees validation engines are disjoint from train engines."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        loader = CMAPSSLoader(sub)
        train_raw, _, _ = loader.load_raw_data()
        all_units = [int(u) for u in np.unique(train_raw["unit"].values)]
        split = get_persistent_split(sub, all_units=all_units)
        
        train_units = set(split["train_units"])
        val_units = set(split["val_units"])
        assert len(train_units.intersection(val_units)) == 0, f"Train and Val engines must be strictly disjoint for {sub}"
        assert len(val_units) > 0, f"Val engines must exist for calibration in {sub}"

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
            assert p50 <= p90, f"p50 ({p50}) must be <= p90 ({p90})"

def test_test_set_coverage_matches_evaluation_report():
    """Verify interval coverage calculated directly on test data matches evaluation report."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        report_file = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        assert report_file.exists(), f"Evaluation report for {sub} must exist"
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        ms = ModelService(sub).load()
        test_df = pd.read_parquet(root_dir / "data" / "processed" / sub / "test.parquet")
        loader = CMAPSSLoader(sub)
        _, _, rul_truth = loader.load_raw_data()
        
        eval_res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        reported_cov = report["interval_coverage"]
        computed_cov = eval_res["interval_coverage"]
        assert abs(reported_cov - computed_cov) < 0.01, (
            f"{sub}: Reported coverage {reported_cov} differs from computed {computed_cov}"
        )

def test_wilson_ci_mathematical_properties():
    """Verify Wilson score confidence interval correctness."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        report_file = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        cov = report["interval_coverage"]
        ci = report.get("coverage_wilson_ci_95")
        assert ci is not None and len(ci) == 2, f"Wilson CI must exist for {sub}"
        lower, upper = ci[0], ci[1]
        
        assert 0.0 <= lower <= 1.0, f"Lower bound out of bounds: {lower}"
        assert 0.0 <= upper <= 1.0, f"Upper bound out of bounds: {upper}"
        assert lower <= upper, f"Lower bound {lower} > upper bound {upper}"
        assert lower <= cov <= upper or abs(cov - lower) < 1e-4 or abs(cov - upper) < 1e-4, (
            f"Point estimate {cov} not contained in [{lower}, {upper}]"
        )
        
        ci_excludes_80 = report.get("coverage_ci_excludes_80")
        expected_excludes = (lower > 0.80) or (upper < 0.80)
        assert ci_excludes_80 == expected_excludes, f"{sub}: ci_excludes_80 mismatch"

def test_model_details_endpoint_returns_conformal_fields():
    """Verify API endpoint returns conformal interval fields."""
    for sub in ["FD001", "FD002"]:
        resp = client.get(f"/api/v1/models/{sub}/details")
        assert resp.status_code == 200
        data = resp.json()
        assert "interval_coverage" in data
        assert "coverage_wilson_ci_95" in data
        assert "coverage_ci_excludes_80" in data
        assert "mean_interval_width" in data
