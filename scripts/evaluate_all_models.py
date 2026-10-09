import json
import time
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir
from src.models.model_service import ModelService
from src.ingest.loader import CMAPSSLoader

def main():
    root_dir = get_root_dir()
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    print("=" * 60)
    print("Evaluating Official C-MAPSS Benchmarks (FD001 - FD004)")
    print("=" * 60)

    benchmarks = []
    for sub in subsets:
        print(f"\n>>> Evaluating {sub}...")
        test_path = root_dir / "data" / "processed" / sub / "test.parquet"
        train_path = root_dir / "data" / "processed" / sub / "train.parquet"
        test_df = pd.read_parquet(test_path)
        train_df = pd.read_parquet(train_path)
        loader = CMAPSSLoader(subset=sub)
        _, _, rul_truth = loader.load_raw_data()

        ms = ModelService(subset=sub).load()
        res = ms.evaluate_test(test_df, rul_truth)

        # Multi-seed evaluation over 3 seeds
        print(f">>> Running 3-seed evaluation for {sub}...")
        multi_seed = ms.evaluate_multi_seed(train_df, test_df, rul_truth, seeds=[42, 43, 44])

        # Deep learning benchmark comparison (1D-CNN and GRU)
        print(f">>> Running Deep Learning (CNN/GRU) comparison for {sub}...")
        deep_comp = ms.compare_deep_learning_models(train_df, test_df, rul_truth)

        benchmarks.append({
            "subset": sub,
            "condition": res["condition"],
            "faults": res["faults"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "nasa_score": res["nasa_score"],
            "pr_auc": res["pr_auc"],
            "train_time_sec": res["train_time_sec"],
            "target_rmse": res["target_rmse"],
            "status": res["status"],
            "interval_coverage": res["interval_coverage"],
            "mean_interval_width": res["mean_interval_width"],
            "multi_seed": multi_seed,
            "deep_comparison": deep_comp["models"]
        })

    summary_file = root_dir / "artifacts" / "reports" / "benchmark_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"benchmarks": benchmarks}, f, indent=2)

    print(f"\n[DONE] Saved real benchmarks summary with multi-seed and DL comparison to {summary_file}:")
    for b in benchmarks:
        ms_info = b["multi_seed"]
        print(f"  [{b['subset']}] RMSE: {b['rmse']} (mean: {ms_info['rmse_mean']} ± {ms_info['rmse_std']}) | Target: <={b['target_rmse']} | Coverage: {b['interval_coverage']*100:.1f}%")

if __name__ == "__main__":
    main()
