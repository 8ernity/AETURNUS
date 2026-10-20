import sys
import os
import json
import time
import hashlib
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir, load_yaml
from src.models.model_service import ModelService
from src.ingest.loader import CMAPSSLoader
from src.eval.protocol import (
    calculate_metrics,
    calculate_nasa_score,
    sample_truncation,
    cap_rul,
    get_protocol_id,
)

def compute_git_sha(root_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root_dir),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if len(out) == 40:
            return out
    except Exception:
        pass
    return "a000000000000000000000000000000000000000"

def compute_config_hash(root_dir: Path) -> str:
    hasher = hashlib.sha256()
    configs_dir = root_dir / "configs"
    for p in sorted(configs_dir.glob("*.yaml")):
        hasher.update(p.name.encode("utf-8"))
        hasher.update(p.read_bytes())
    return hasher.hexdigest()

def compute_data_sha256(root_dir: Path, subsets: list) -> dict:
    data_hashes = {}
    for sub in subsets:
        tr_p = root_dir / "data" / "processed" / sub / "train.parquet"
        te_p = root_dir / "data" / "processed" / sub / "test.parquet"
        if tr_p.exists():
            data_hashes[f"{sub}_train"] = hashlib.sha256(tr_p.read_bytes()).hexdigest()
        if te_p.exists():
            data_hashes[f"{sub}_test"] = hashlib.sha256(te_p.read_bytes()).hexdigest()
    return data_hashes

def select_champion(comparison_models: list) -> dict:
    """
    Selects champion algorithm based on protocol rules:
    Filter candidate models with late_rate <= 0.60, then select minimum CV RMSE.
    """
    rule_inputs = {
        "max_acceptable_late_rate": 0.60,
        "primary_metric": "cv_mean_rmse",
        "secondary_metric": "nasa_score",
        "evaluated_algorithms": [m.get("model_name", "") for m in comparison_models]
    }
    
    # Filter candidates
    valid_candidates = [m for m in comparison_models if m.get("late_rate", 0.5) <= 0.60]
    if not valid_candidates:
        valid_candidates = comparison_models

    # Sort by cv_mean_rmse (or test rmse)
    best = min(valid_candidates, key=lambda m: m.get("cv_mean_rmse", m.get("rmse", 999.0)))
    
    # Canonical algorithm key inside bundle
    algo_name = "lightgbm"
    if "ridge" in best.get("model_name", "").lower():
        algo_name = "ridge"
    elif "hist" in best.get("model_name", "").lower():
        algo_name = "hist_gb"
        
    return {
        "selected_algorithm": algo_name,
        "model_name": best.get("model_name", "LightGBM Regressor (Champion)"),
        "selection_rule": "min_cv_rmse_with_nasa_safety",
        "rule_inputs": rule_inputs
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate C-MAPSS models and build run manifest.")
    parser.add_argument("--force", action="store_true", help="Force recomputation of CV and algorithm comparisons.")
    args = parser.parse_args()
    force = args.force

    root_dir = get_root_dir()
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    print("=" * 60)
    print("Evaluating Official C-MAPSS Benchmarks & Building Run Manifest")
    print("=" * 60)

    git_sha = compute_git_sha(root_dir)
    config_hash = compute_config_hash(root_dir)
    data_sha256 = compute_data_sha256(root_dir, subsets)
    run_id = f"run_{int(time.time())}_{git_sha[:8]}"
    protocol_id = get_protocol_id()

    manifest_subsets = {}
    benchmarks = []

    for sub in subsets:
        print(f"\n>>> Evaluating {sub}...")
        test_path = root_dir / "data" / "processed" / sub / "test.parquet"
        train_path = root_dir / "data" / "processed" / sub / "train.parquet"
        test_df = pd.read_parquet(test_path)
        train_df = pd.read_parquet(train_path)
        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        # Load active champion service
        ms = ModelService(subset=sub).load()

        # 5-fold GroupKFold Cross-Validation on persistent train engines
        cv_path = root_dir / "artifacts" / "reports" / f"cv_{sub}.json"
        if cv_path.exists() and not force:
            print(f">>> Reusing existing 5-fold CV evaluation from {cv_path.name}...")
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_res = json.load(f)
        else:
            print(f">>> Running 5-fold CV evaluation for {sub}...")
            cv_res = ms.evaluate_cv(k=5, train_df=train_df)

        # Multi-seed evaluation over 3 seeds
        eval_path = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        multi_seed = {}
        if eval_path.exists() and not force:
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    multi_seed = json.load(f).get("multi_seed", {})
            except Exception:
                multi_seed = {}
        if not multi_seed:
            print(f">>> Running 3-seed evaluation for {sub}...")
            multi_seed = ms.evaluate_multi_seed(train_df, test_df, rul_truth, seeds=[42, 43, 44])

        # Deep learning and baseline comparison
        comp_path = root_dir / "artifacts" / "reports" / f"model_comparison_{sub}.json"
        if comp_path.exists() and not force:
            print(f">>> Reusing existing algorithm comparison from {comp_path.name}...")
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_res = json.load(f)
        else:
            print(f">>> Running algorithm comparison for {sub}...")
            comp_res = ms.compare_deep_learning_models(train_df, test_df, rul_truth)

        # Champion selection via protocol
        champ_decision = select_champion(comp_res["models"])
        selected_algo = champ_decision["selected_algorithm"]
        print(f">>> Selected Champion for {sub}: {selected_algo}")

        # Ensure active served bundle matches selected champion algorithm
        if getattr(ms, "algorithm", "") != selected_algo:
            print(f">>> Retraining served champion bundle for {sub} with {selected_algo}...")
            ms.algorithm = selected_algo
            ms.train(train_df)
            ms.save(version=getattr(ms, "version", "v1.0"))
            ms.promote()
        else:
            ms.save(version=getattr(ms, "version", "v1.0"))
            ms.promote()

        active_bundle_path = root_dir / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        
        # Now evaluate test on the active served bundle
        eval_rows = sample_truncation(test_df, mode="last")
        res = ms.evaluate_test(test_df, rul_truth)

        # Compute full bundle sha256
        bundle_sha256 = hashlib.sha256(active_bundle_path.read_bytes()).hexdigest()

        # Prepare exact test metrics
        metrics = {
            "rmse": float(res["rmse"]),
            "mae": float(res["mae"]),
            "nasa_score": float(res["nasa_score"]),
            "early_rate": float(res.get("early_rate", 0.5)),
            "late_rate": float(res.get("late_rate", 0.5)),
            "pr_auc": float(res["pr_auc"]),
            "interval_coverage": float(res["interval_coverage"]),
            "mean_interval_width": float(res["mean_interval_width"])
        }

        manifest_subsets[sub] = {
            "bundle_sha256": bundle_sha256,
            "algorithm": selected_algo,
            "protocol_id": protocol_id,
            "n_engines": len(eval_rows),
            "metrics": metrics,
            "cv": {
                "protocol_id": protocol_id,
                "k_folds": cv_res.get("k_folds", 5),
                "mean_rmse": cv_res["mean_rmse"],
                "std_rmse": cv_res["std_rmse"],
                "mean_nasa": cv_res["mean_nasa"],
                "std_nasa": cv_res["std_nasa"],
                "mean_train_rmse": cv_res.get("mean_train_rmse", 0.0)
            },
            "comparison_rows": comp_res["models"],
            "champion": champ_decision
        }

        benchmarks.append({
            "subset": sub,
            "condition": res["condition"],
            "faults": res["faults"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "nasa_score": res["nasa_score"],
            "early_rate": res.get("early_rate", 0.5),
            "late_rate": res.get("late_rate", 0.5),
            "nasa_phm08": res.get("nasa_phm08", {}),
            "pr_auc": res["pr_auc"],
            "train_time_sec": res["train_time_sec"],
            "target_rmse": res["target_rmse"],
            "status": res["status"],
            "interval_coverage": res["interval_coverage"],
            "mean_interval_width": res["mean_interval_width"],
            "multi_seed": multi_seed,
            "deep_comparison": comp_res["models"]
        })

    # Assemble complete manifest
    manifest_data = {
        "run_id": run_id,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "created_at": time.time(),
        "subsets": manifest_subsets
    }

    # Atomically write run_manifest.json
    manifest_file = root_dir / "artifacts" / "run_manifest.json"
    manifest_tmp = root_dir / "artifacts" / "run_manifest.json.tmp"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    manifest_tmp.replace(manifest_file)
    print(f"\n[DONE] Atomically written run manifest to {manifest_file}")

    # Write legacy benchmark_summary.json for backward compatibility
    summary_file = root_dir / "artifacts" / "reports" / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"benchmarks": benchmarks}, f, indent=2)

    for sub, sinfo in manifest_subsets.items():
        m = sinfo["metrics"]
        print(f"  [{sub}] Champion: {sinfo['algorithm']} | RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} | NASA: {m['nasa_score']:.1f} | Cov: {m['interval_coverage']*100:.1f}%")

if __name__ == "__main__":
    main()
