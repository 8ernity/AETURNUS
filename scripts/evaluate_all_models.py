import sys
import os
import gc
import json
import time
import hashlib
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir, load_yaml
from src.models.model_service import (
    ModelService,
    algorithm_for_p50_model,
    load_cv_protocol,
    resolve_coverage_policy,
    resolve_target_policy,
)
from src.ingest.loader import CMAPSSLoader
from src.eval.protocol import (
    calculate_metrics,
    calculate_nasa_score,
    sample_truncation,
    cap_rul,
    get_protocol_id,
    eval_frames,
    DEFAULT_SEED,
)
from src.eval.regime import regime_mix_gap

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

ARCHITECTURE_TO_ALGORITHM = {
    "linear": "ridge",
    "ridge": "ridge",
    "gbdt": "hist_gb",
    "hist_gradient_boosting": "hist_gb",
    "hist_gb": "hist_gb",
    "lightgbm": "lightgbm",
}

def _algorithm_for(candidate: dict) -> str:
    """Maps a comparison row's architecture onto the canonical bundle algorithm key."""
    arch = str(candidate["architecture"]).strip().lower()
    if arch not in ARCHITECTURE_TO_ALGORITHM:
        raise ValueError(
            f"Champion candidate '{candidate['model_name']}' has architecture '{candidate['architecture']}' "
            f"which ModelService cannot serve; add it to ARCHITECTURE_TO_ALGORITHM and ModelService.train."
        )
    return ARCHITECTURE_TO_ALGORITHM[arch]

def select_champion(comparison_models: list) -> dict:
    """
    Selects the champion strictly from cross-validation scores (no test leakage):
    - Candidates missing CV mean or std are recorded as ineligible, never defaulted.
    - Primary criterion: lowest cv_mean_rmse.
    - If the margin to the runner-up is below the leader's CV std, the tie is resolved
      by the tie_break rule (lowest cv_mean_nasa, i.e. PHM'08 penalty) and recorded.
    """
    per_candidate_cv = {}
    ineligible = {}
    valid_candidates = []
    for m in comparison_models:
        per_candidate_cv[m["model_name"]] = {
            "mean_rmse": m.get("cv_mean_rmse"),
            "std_rmse": m.get("cv_std_rmse"),
            "mean_nasa": m.get("cv_mean_nasa"),
            "std_nasa": m.get("cv_std_nasa"),
        }
        if m.get("servable", True) is False:
            ineligible[m["model_name"]] = "not servable by ModelService; excluded from champion selection"
        elif m.get("cv_mean_rmse") is None or m.get("cv_std_rmse") is None:
            ineligible[m["model_name"]] = "missing CV mean/std; excluded from champion selection"
        else:
            valid_candidates.append(m)

    if not valid_candidates:
        raise ValueError("No candidate carries CV mean/std; cannot select champion without CV scores.")

    ranked = sorted(valid_candidates, key=lambda m: float(m["cv_mean_rmse"]))
    leader = ranked[0]
    leader_rmse = float(leader["cv_mean_rmse"])
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = float(runner_up["cv_mean_rmse"]) - leader_rmse if runner_up is not None else 0.0

    rule_inputs = {
        "primary_metric": "cv_mean_rmse",
        "secondary_metric": "cv_mean_nasa",
        "evaluated_algorithms": [m["model_name"] for m in comparison_models],
        "ineligible_candidates": ineligible,
    }

    if runner_up is not None and margin <= 1e-9:
        tie_break_rule = "cv_mean_nasa_phm08"
        tied = [m for m in ranked if float(m["cv_mean_rmse"]) - leader_rmse <= 1e-9]
        missing_nasa = [m["model_name"] for m in tied if m.get("cv_mean_nasa") is None]
        if missing_nasa:
            raise ValueError(f"Tie-break requires cv_mean_nasa for tied candidates: {missing_nasa}")
        best = min(tied, key=lambda m: float(m["cv_mean_nasa"]))
        selection_rule = f"tied_min_cv_rmse_tiebreak_{tie_break_rule}"
    else:
        tie_break_rule = None
        best = leader
        selection_rule = "min_cv_rmse"

    return {
        "selected_algorithm": _algorithm_for(best),
        "model_name": best["model_name"],
        "selection_rule": selection_rule,
        "selected_cv_rmse": round(float(best["cv_mean_rmse"]), 4),
        "selected_cv_std": round(float(best["cv_std_rmse"]), 4),
        "margin": round(margin, 3),
        "tie_break": tie_break_rule,
        "per_candidate_cv": per_candidate_cv,
        "rule_inputs": rule_inputs
    }

def build_manifest_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """Projects evaluation results onto the exact metrics persisted in the manifest."""
    return {
        "rmse": float(res["rmse"]),
        "mae": float(res["mae"]),
        "nasa_score": float(res["nasa_score"]),
        "early_rate": float(res["early_rate"]),
        "late_rate": float(res["late_rate"]),
        "pr_auc": float(res["pr_auc"]),
        "interval_coverage": float(res["interval_coverage"]),
        "mean_interval_width": (
            float(res["mean_interval_width"])
            if res.get("mean_interval_width") is not None
            else None
        ),
        "coverage_target": float(res["coverage_target"]),
        "coverage_wilson_ci_95": res["coverage_wilson_ci_95"],
        "coverage_wilson_ci_contains_target": bool(res["coverage_wilson_ci_contains_target"]),
        "coverage_ci_excludes_target": bool(res["coverage_ci_excludes_target"]),
        "coverage_ci_contains_target": bool(res["coverage_ci_contains_target"]),
        "coverage_by_bucket": res["coverage_by_bucket"],
        "coverage_metric": res["coverage_metric"],
        "coverage_metric_value": float(res["coverage_metric_value"]),
        "coverage_metric_wilson_ci_95": res["coverage_metric_wilson_ci_95"],
        "coverage_metric_ci_lower": res["coverage_metric_ci_lower"],
        "coverage_metric_ci_upper": res["coverage_metric_ci_upper"],
        "target_coverage": float(res["target_coverage"]),
        "target_coverage_met": bool(res["target_coverage_met"]),
        "coverage_ci_lower_meets_target": bool(res["coverage_ci_lower_meets_target"]),
        "coverage_all": res["coverage_all"],
        "coverage_all_n": int(res["coverage_all_n"]),
        "coverage_all_wilson_ci_95": res["coverage_all_wilson_ci_95"],
        "coverage_le_cap": res["coverage_le_cap"],
        "coverage_le_cap_n": int(res["coverage_le_cap_n"]),
        "coverage_le_cap_wilson_ci_95": res["coverage_le_cap_wilson_ci_95"],
        "coverage_above_cap": res["coverage_above_cap"],
        "coverage_above_cap_n": int(res["coverage_above_cap_n"]),
        "coverage_above_cap_wilson_ci_95": res["coverage_above_cap_wilson_ci_95"],
        "coverage_split": res["coverage_split"],
        "coverage_rule": res["coverage_rule"],
        "n_censored_intervals": int(res["n_censored_intervals"]),
        "n_closed_intervals": int(res["n_closed_intervals"]),
        "censored_interval_rate": float(res["censored_interval_rate"]),
        "rul_cap": res["metrics"]["rul_cap"],
        "rmse_vs_raw_truth": res["metrics"]["rmse_vs_raw_truth"],
        "rmse_vs_capped_truth": res["metrics"]["rmse_vs_capped_truth"],
        "share_true_rul_above_cap": res["metrics"]["share_true_rul_above_cap"],
        "rmse_on_true_rul_le_cap": res["metrics"]["rmse_on_true_rul_le_cap"],
        "n_true_rul_le_cap": res["metrics"]["n_true_rul_le_cap"],
        "n_true_rul_above_cap": res["metrics"]["n_true_rul_above_cap"],
        "rul_bucket_metrics": res["metrics"]["rul_bucket_metrics"],
    }


def evaluate_subset(
    sub: str, root_dir: Path, force: bool = False, persist_active: bool = True
) -> Dict[str, Any]:
    test_path = root_dir / "data" / "processed" / sub / "test.parquet"
    train_path = root_dir / "data" / "processed" / sub / "train.parquet"
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    loader = CMAPSSLoader(subset=sub)
    _, _, rul_truth = loader.load_raw_data()
    test_protocol_id = get_protocol_id(mode="last", seed=DEFAULT_SEED, n_draws=1)
    cv_proto = load_cv_protocol()
    expected_cv_protocol_id = get_protocol_id(
        mode=cv_proto["mode"], seed=DEFAULT_SEED, n_draws=cv_proto["n_draws"]
    )

    # Load the shipped service only to select features for the algorithm comparison.
    ms = ModelService(subset=sub).load()

    # Deep learning and baseline comparison (cached only for the canonical CV protocol)
    comp_path = root_dir / "artifacts" / "reports" / f"model_comparison_{sub}.json"
    comp_res = None
    if comp_path.exists() and not force:
        with open(comp_path, "r", encoding="utf-8") as f:
            cached_comp = json.load(f)
        if (
            cached_comp.get("cv_protocol_id") == expected_cv_protocol_id
            and cached_comp.get("cv_min_cycles") == cv_proto["min_cycles"]
            and cached_comp.get("models")
        ):
            print(f">>> Reusing existing algorithm comparison from {comp_path.name}...")
            comp_res = cached_comp
    if comp_res is None:
        print(f">>> Running algorithm comparison for {sub}...")
        comp_res = ms.compare_deep_learning_models(train_df, test_df, rul_truth)

    # Champion selection via protocol (servable candidates with complete censored CV)
    champ_decision = select_champion(comp_res["models"])
    selected_algo = champ_decision["selected_algorithm"]
    print(f">>> Selected Champion for {sub}: {selected_algo}")

    # Drop the benchmark-loaded service and any torch state before retraining.
    del ms
    gc.collect()

    # Always retrain the selected algorithm through the canonical train path, then
    # promote the saved bundle so the served artifact is the champion.
    print(f">>> Retraining served champion bundle for {sub} with {selected_algo}...")
    ms = ModelService(subset=sub)
    ms.algorithm = selected_algo
    ms.train(train_df)
    version = getattr(ms, "version", "v1.0")
    versioned_bundle_path = ms.save(version=version)
    if persist_active:
        ms.promote(version=version)
        ms.load()
        served_bundle_path = (
            root_dir / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        )
    else:
        # Evaluation-only callers must not rewrite the tracked serving artifact.
        ms.load(version=version)
        served_bundle_path = versioned_bundle_path

    bundle = joblib.load(served_bundle_path)
    p50_model_class = type(bundle["rul_p50_model"]).__name__
    served_algo = algorithm_for_p50_model(bundle["rul_p50_model"])
    if served_algo != selected_algo:
        raise ValueError(
            f"[{sub}] Champion selection chose '{selected_algo}', but the promoted bundle "
            f"serves p50 class '{p50_model_class}' mapped to '{served_algo}'."
        )

    # 5-fold GroupKFold CV on persistent train engines using the promoted champion.
    cv_path = root_dir / "artifacts" / "reports" / f"cv_{sub}.json"
    cv_res = None
    if cv_path.exists() and not force:
        with open(cv_path, "r", encoding="utf-8") as f:
            cached_cv = json.load(f)
        if (
            cached_cv.get("protocol_id") == expected_cv_protocol_id
            and cached_cv.get("algorithm") == served_algo
            and cached_cv.get("evaluation_mode") == cv_proto["mode"]
            and cached_cv.get("n_draws") == cv_proto["n_draws"]
            and cached_cv.get("min_cycles") == cv_proto["min_cycles"]
            and cached_cv.get("regime_mix")
            and cached_cv.get("by_regime")
        ):
            print(f">>> Reusing existing censored 5-fold CV evaluation from {cv_path.name}...")
            cv_res = cached_cv
    if cv_res is None:
        print(f">>> Running censored 5-fold CV evaluation for {sub}...")
        cv_res = ms.evaluate_cv(train_df=train_df)

    # Multi-seed evaluation over 3 seeds for the served champion algorithm.
    eval_path = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
    multi_seed = {}
    if eval_path.exists() and not force:
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                cached_eval = json.load(f)
            if cached_eval.get("algorithm") == served_algo:
                multi_seed = cached_eval.get("multi_seed", {})
        except (json.JSONDecodeError, OSError):
            multi_seed = {}
    if not multi_seed:
        print(f">>> Running 3-seed evaluation for {sub}...")
        base_seed = int(load_yaml(root_dir / "configs" / "pipeline.yaml")["splits"]["random_seed"])
        multi_seed = ms.evaluate_multi_seed(
            train_df, test_df, rul_truth, seeds=[base_seed, base_seed + 1, base_seed + 2]
        )

    # Evaluate test on the promoted served bundle using the canonical last-cycle frame.
    eval_rows = eval_frames(test_df, mode="last", seed=DEFAULT_SEED, n_draws=1)[0]
    res = ms.evaluate_test(test_df, rul_truth)

    # Compute full bundle sha256
    bundle_sha256 = hashlib.sha256(served_bundle_path.read_bytes()).hexdigest()

    # Resolve target_met from the named metric policies in configs/targets.yaml.
    target_policy = resolve_target_policy(sub, res["metrics"])
    target_val = target_policy["target_rmse"]
    target_metric = target_policy["target_metric"]
    target_metric_value = target_policy["target_metric_value"]
    target_met = target_policy["target_met"]
    coverage_policy = resolve_coverage_policy(sub, res["metrics"])

    # Prepare exact test metrics without fallback defaults
    metrics = build_manifest_metrics(res)
    calibration = bundle.get("conformal_buckets") or res.get("conformal_buckets") or {}

    regime_gap = regime_mix_gap(
        cv_res["regime_mix"],
        res["test_by_regime"],
    )

    sub_dict = {
        "bundle_sha256": bundle_sha256,
        "algorithm": served_algo,
        "p50_model_class": p50_model_class,
        "protocol_id": test_protocol_id,
        "cv_protocol_id": cv_res["protocol_id"],
        "n_engines": len(eval_rows),
        "target_rmse": target_val,
        "target_metric": target_metric,
        "target_metric_value": round(float(target_metric_value), 4),
        "target_met": target_met,
        "target_metric_provenance": target_policy["target_metric_provenance"],
        "target_provenance": {
            **target_policy["target_provenance"],
            "config": "configs/targets.yaml",
        },
        "coverage_metric": coverage_policy["coverage_metric"],
        "coverage_metric_value": round(float(coverage_policy["coverage_metric_value"]), 4),
        "coverage_metric_provenance": coverage_policy["coverage_metric_provenance"],
        "target_coverage": round(float(coverage_policy["target_coverage"]), 4),
        "target_coverage_provenance": coverage_policy["target_coverage_provenance"],
        "target_coverage_met": coverage_policy["target_coverage_met"],
        "coverage_status": (
            "Coverage Target Met"
            if coverage_policy["target_coverage_met"]
            else "Coverage Target Missed"
        ),
        "metrics": metrics,
        "calibration": calibration,
        "cv": {
            "protocol_id": cv_res["protocol_id"],
            "evaluation_mode": cv_res["evaluation_mode"],
            "n_draws": cv_res["n_draws"],
            "min_cycles": cv_res["min_cycles"],
            "k_folds": cv_res["k_folds"],
            "mean_rmse": cv_res["mean_rmse"],
            "std_rmse": cv_res["std_rmse"],
            "mean_nasa": cv_res["mean_nasa"],
            "std_nasa": cv_res["std_nasa"],
            "mean_train_rmse": cv_res["mean_train_rmse"],
            "by_regime": cv_res["by_regime"],
        },
        "cv_by_regime": cv_res["by_regime"],
        "test_by_regime": res["test_by_regime"],
        "regime_mix_gap": regime_gap,
        "comparison_rows": comp_res["models"],
        "champion": champ_decision,
        "condition": res["condition"],
        "faults": res["faults"],
        "status": "Target Met" if target_met else "Target Missed",
        "train_time_sec": res["train_time_sec"],
        "multi_seed": multi_seed
    }
    return sub_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force full recomputation of CV and comparisons")
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    force = args.force

    root_dir = get_root_dir()
    if args.subset:
        result = evaluate_subset(args.subset, root_dir, force=force)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    print("=" * 60)
    print("Evaluating Official C-MAPSS Benchmarks & Building Run Manifest")
    print("=" * 60)

    git_sha = compute_git_sha(root_dir)
    config_hash = compute_config_hash(root_dir)
    data_sha256 = compute_data_sha256(root_dir, subsets)
    run_id = f"run_{int(time.time())}_{git_sha[:8]}"

    manifest_subsets = {}
    benchmarks = []

    # Evaluate each subset in its own process so PyTorch/tree-model memory is fully
    # reclaimed between subsets on memory-constrained hosts.
    worker_dir = Path(tempfile.mkdtemp(prefix=f"aeturnus_eval_{run_id}_"))

    for sub in subsets:
        print(f"\n>>> Evaluating {sub}...")
        result_path = worker_dir / f"evaluate_{sub}.json"
        command = [
            sys.executable, "-m", "scripts.evaluate_all_models",
            "--subset", sub, "--result-json", str(result_path),
        ]
        if force:
            command.append("--force")
        subprocess.run(command, cwd=str(root_dir), check=True)
        sub_info = json.loads(result_path.read_text(encoding="utf-8"))
        manifest_subsets[sub] = sub_info
        benchmarks.append({
            "subset": sub,
            "condition": sub_info["condition"],
            "faults": sub_info["faults"],
            "rmse": sub_info["metrics"]["rmse"],
            "rmse_vs_raw_truth": sub_info["metrics"]["rmse_vs_raw_truth"],
            "rmse_vs_capped_truth": sub_info["metrics"]["rmse_vs_capped_truth"],
            "share_true_rul_above_cap": sub_info["metrics"]["share_true_rul_above_cap"],
            "rmse_on_true_rul_le_cap": sub_info["metrics"]["rmse_on_true_rul_le_cap"],
            "rul_cap": sub_info["metrics"]["rul_cap"],
            "mae": sub_info["metrics"]["mae"],
            "nasa_score": sub_info["metrics"]["nasa_score"],
            "early_rate": sub_info["metrics"]["early_rate"],
            "late_rate": sub_info["metrics"]["late_rate"],
            "nasa_phm08": sub_info.get("champion", {}),
            "pr_auc": sub_info["metrics"]["pr_auc"],
            "train_time_sec": sub_info["train_time_sec"],
            "target_rmse": sub_info["target_rmse"],
            "target_metric": sub_info["target_metric"],
            "target_metric_value": sub_info["target_metric_value"],
            "target_met": sub_info["target_met"],
            "status": sub_info["status"],
            "coverage_metric": sub_info["coverage_metric"],
            "coverage_metric_value": sub_info["coverage_metric_value"],
            "target_coverage": sub_info["target_coverage"],
            "target_coverage_met": sub_info["target_coverage_met"],
            "coverage_status": sub_info["coverage_status"],
            "interval_coverage": sub_info["metrics"]["interval_coverage"],
            "coverage_all": sub_info["metrics"]["coverage_all"],
            "coverage_all_n": sub_info["metrics"]["coverage_all_n"],
            "coverage_all_wilson_ci_95": sub_info["metrics"]["coverage_all_wilson_ci_95"],
            "coverage_le_cap": sub_info["metrics"]["coverage_le_cap"],
            "coverage_le_cap_n": sub_info["metrics"]["coverage_le_cap_n"],
            "coverage_le_cap_wilson_ci_95": sub_info["metrics"]["coverage_le_cap_wilson_ci_95"],
            "coverage_above_cap": sub_info["metrics"]["coverage_above_cap"],
            "coverage_above_cap_n": sub_info["metrics"]["coverage_above_cap_n"],
            "coverage_above_cap_wilson_ci_95": sub_info["metrics"]["coverage_above_cap_wilson_ci_95"],
            "mean_interval_width": sub_info["metrics"]["mean_interval_width"],
            "multi_seed": sub_info.get("multi_seed", {}),
            "deep_comparison": sub_info.get("comparison_rows", [])
        })

    shutil.rmtree(worker_dir, ignore_errors=True)

    flagged_regimes = [
        sub for sub, info in manifest_subsets.items()
        if info.get("regime_mix_gap", {}).get("gap_flag")
    ]

    targets_cfg = load_yaml(root_dir / "configs" / "targets.yaml", raw=True)
    target_metric_cfg = targets_cfg["target_metric"]
    coverage_metric_cfg = targets_cfg["coverage_metric"]
    target_coverage_cfg = load_yaml(
        root_dir / "configs" / "pipeline.yaml", raw=True
    )["conformal"]["target_coverage"]

    # Assemble complete manifest
    manifest_data = {
        "run_id": run_id,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "created_at": time.time(),
        "target_metric": {
            "value": target_metric_cfg["value"],
            "kind": target_metric_cfg.get("kind"),
            "source": target_metric_cfg.get("source"),
            "allowed_metrics": target_metric_cfg.get("allowed_metrics", []),
        },
        "coverage_metric": {
            "value": coverage_metric_cfg["value"],
            "kind": coverage_metric_cfg.get("kind"),
            "source": coverage_metric_cfg.get("source"),
            "allowed_metrics": coverage_metric_cfg.get("allowed_metrics", []),
        },
        "target_coverage": {
            **target_coverage_cfg,
            "config": "configs/pipeline.yaml",
        },
        "rul_buckets": targets_cfg["rul_buckets"],
        "subsets": manifest_subsets,
        "regime_mix_gap": {
            "flagged_subsets": flagged_regimes,
            "status": "material_gap" if flagged_regimes else "comparable",
            "note": "Chi-square and KS compare censored-CV sample regime mix with official test regime mix.",
        },
    }

    # Atomically write run_manifest.json
    manifest_file = root_dir / "artifacts" / "run_manifest.json"
    manifest_tmp = root_dir / "artifacts" / "run_manifest.json.tmp"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    manifest_tmp.replace(manifest_file)
    print(f"\n[DONE] Atomically written run manifest to {manifest_file}")

    # Write a minimal registry: metrics and paths are derived from this manifest at read time.
    registry_file = root_dir / "artifacts" / "reports" / "model_registry.json"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_rows = [
        {
            "version": "v1.0",
            "subset": sub,
            "alias": "champion",
            "manifest_run_id": run_id,
            "bundle_sha256": sinfo["bundle_sha256"],
        }
        for sub, sinfo in manifest_subsets.items()
    ]
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump({"models": registry_rows, "active_champion": "v1.0"}, f, indent=2)

    # Write legacy benchmark_summary.json for backward compatibility
    summary_file = root_dir / "artifacts" / "reports" / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "target_metric": manifest_data["target_metric"],
            "coverage_metric": manifest_data["coverage_metric"],
            "target_coverage": manifest_data["target_coverage"],
            "benchmarks": benchmarks,
        }, f, indent=2)

    for sub, sinfo in manifest_subsets.items():
        m = sinfo["metrics"]
        print(
            f"  [{sub}] Champion: {sinfo['algorithm']} | "
            f"{sinfo['target_metric']}: {sinfo['target_metric_value']:.2f} | "
            f"raw RMSE: {m['rmse_vs_raw_truth']:.2f} | "
            f"capped RMSE: {m['rmse_vs_capped_truth']:.2f} | "
            f"MAE: {m['mae']:.2f} | NASA: {m['nasa_score']:.1f} | "
            f"Cov(all): {m['coverage_all']*100:.1f}% | "
            f"Cov(<=cap): {m['coverage_le_cap']*100:.1f}% | "
            f"Coverage policy met: {sinfo['target_coverage_met']}"
        )

if __name__ == "__main__":
    main()
