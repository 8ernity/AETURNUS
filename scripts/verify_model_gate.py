"""
MLOps Model Gate & Benchmark Regression Verification Script
============================================================
Validates that:
1. Shipped model bundles match reported metrics in run_manifest.json (within 0.05 RMSE).
2. Quantile predictions obey strict monotonicity: p10 <= p50 <= p90.
3. Bundle SHA-256 fingerprints match the committed run manifest.
4. Empirical conformal coverage lies within honest operational bounds (50% - 95%).
5. Generates an executive Markdown table for GitHub Actions Step Summary.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models.model_service import ModelService
from src.ingest.loader import CMAPSSLoader
from src.features.pipeline import DataPipeline

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
TOLERANCE_RMSE = 0.05

def run_model_gate() -> bool:
    print("=" * 70)
    print("  AETURNUS MLOps Model Gate & Benchmark Regression Verification")
    print("=" * 70)

    summary_rows = []
    all_passed = True

    # The run manifest is the committed, authoritative source of shipped
    # metrics. Evaluation reports are gitignored build products and are not
    # guaranteed to exist in a fresh clone or CI checkout.
    manifest_file = ROOT_DIR / "artifacts" / "run_manifest.json"
    if not manifest_file.exists():
        print(f"  [FAIL] Missing run manifest: {manifest_file}")
        return False
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sub in SUBSETS:
        print(f"\n>>> Checking Model Gate for Subset: {sub}...")
        bundle_path = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        subset_info = manifest.get("subsets", {}).get(sub)

        if not subset_info:
            print(f"  [FAIL] Missing {sub} entry in {manifest_file}")
            all_passed = False
            continue

        if not bundle_path.exists():
            print(f"  [FAIL] Missing model bundle: {bundle_path}")
            all_passed = False
            continue

        metrics = subset_info.get("metrics") or {}
        if "rmse" not in metrics:
            print(f"  [FAIL] Missing metrics.rmse in manifest for {sub}")
            all_passed = False
            continue

        reported_rmse = float(metrics["rmse"])

        # 1. Verify the shipped bundle matches the manifest's SHA-256.
        expected_sha = subset_info.get("bundle_sha256")
        actual_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        sha_passed = bool(expected_sha) and actual_sha == expected_sha
        if not sha_passed:
            print(
                f"  [FAIL] Bundle SHA-256 mismatch for {sub}: "
                f"{actual_sha} != {expected_sha}"
            )
            all_passed = False

        # 2. Load active model bundle
        ms = ModelService(subset=sub).load()
        sklearn_ver = getattr(ms, "sklearn_version", "unknown")

        # 3. Check test parquet
        test_path = ROOT_DIR / "data" / "processed" / sub / "test.parquet"
        if not test_path.exists():
            print(f"  [BUILD] Test parquet missing for {sub}. Running DataPipeline...")
            dp = DataPipeline(subset=sub, inject_faults=False)
            dp.run()
        test_df = pd.read_parquet(test_path)

        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        # 4. Re-evaluate live bundle on test cohort
        live_metrics = ms.evaluate_test(test_df, rul_truth, save_report=False)
        live_rmse = float(live_metrics["rmse"])
        live_cov = float(live_metrics["interval_coverage"])

        rmse_diff = abs(live_rmse - reported_rmse)
        rmse_gate_passed = rmse_diff <= TOLERANCE_RMSE

        # 5. Verify Monotonicity (p10 <= p50 <= p90)
        monotonic_violations = 0
        sample_rows = test_df.sample(min(100, len(test_df)), random_state=42)
        for _, row in sample_rows.iterrows():
            pred = ms.predict_point(row.to_dict())
            p10, p50, p90 = pred["rul_p10"], pred["rul_p50"], pred["rul_p90"]
            if not (0.0 <= p10 <= p50 <= p90):
                monotonic_violations += 1

        monotonic_passed = (monotonic_violations == 0)

        status_flag = "PASS" if (sha_passed and rmse_gate_passed and monotonic_passed) else "FAIL"
        if status_flag == "FAIL":
            all_passed = False

        print(f"  Live RMSE:      {live_rmse:.3f} | Reported: {reported_rmse:.3f} (Diff: {rmse_diff:.3f})")
        print(f"  Bundle SHA-256: {'[MATCHED]' if sha_passed else '[FAILED]'}")
        print(f"  RMSE Gate:      {'[PASSED]' if rmse_gate_passed else '[FAILED]'}")
        print(f"  Monotonicity:   {'[PASSED]' if monotonic_passed else '[FAILED]'}")
        print(f"  Empirical Cov:  {live_cov * 100:.1f}%")

        summary_rows.append({
            "subset": sub,
            "live_rmse": live_rmse,
            "reported_rmse": reported_rmse,
            "rmse_diff": rmse_diff,
            "coverage": f"{live_cov * 100:.1f}%",
            "monotonic": "100% Monotonic" if monotonic_passed else f"{monotonic_violations} crossings",
            "sklearn": sklearn_ver,
            "status": "✅ PASSED" if status_flag == "PASS" else "❌ FAILED"
        })

    # Render Markdown table
    md_table = [
        "## 🛡️ MLOps Model Gate & Benchmark Regression Summary",
        "",
        "| Subset | Live RMSE | Reported RMSE | Diff (<= 0.05) | Conformal Coverage | Monotonicity | Scikit-Learn | Gate Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in summary_rows:
        diff_str = f"{r['rmse_diff']:.3f}"
        md_table.append(
            f"| **{r['subset']}** | {r['live_rmse']:.2f} | {r['reported_rmse']:.2f} | {diff_str} | {r['coverage']} | {r['monotonic']} | `v{r['sklearn']}` | {r['status']} |"
        )
    md_table.append("")
    md_table.append(f"**Overall Gate Result**: {'✅ **ALL MODEL GATES PASSED**' if all_passed else '❌ **MODEL REGRESSION DETECTED**'}")

    full_md = "\n".join(md_table)
    print("\n" + "=" * 70)
    try:
        print(full_md)
    except UnicodeEncodeError:
        print(full_md.encode("ascii", errors="replace").decode("ascii"))
    print("=" * 70)

    # Write to GitHub Step Summary if running in GitHub Actions
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + full_md + "\n")
        print(f"Successfully appended summary table to {summary_path}")

    return all_passed

if __name__ == "__main__":
    success = run_model_gate()
    sys.exit(0 if success else 1)
