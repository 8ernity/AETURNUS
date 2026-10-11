import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

def ensure_data_and_models():
    """
    Ensures that raw data, processed pipelines, and trained model bundles
    exist on the machine before starting the API server.
    """
    print("=" * 60)
    print("  AETURNUS: Verifying Data, Pipeline & Model Artifacts")
    print("=" * 60)

    # 1. Download raw data if missing
    raw_fd001 = ROOT_DIR / "data" / "raw" / "train_FD001.txt"
    if not raw_fd001.exists():
        print("[ENTRYPOINT] Raw C-MAPSS dataset missing. Running download_data...")
        import download_data
        download_data.download_and_extract()
    else:
        print("[ENTRYPOINT] Raw dataset present.")

    # 2. Build feature pipelines if processed parquets are missing
    fd001_train = ROOT_DIR / "data" / "processed" / "FD001" / "train.parquet"
    fd001_test = ROOT_DIR / "data" / "processed" / "FD001" / "test.parquet"
    if not (fd001_train.exists() and fd001_test.exists()):
        print("[ENTRYPOINT] Primary dataset FD001 missing. Building FD001 pipeline...")
        from src.features.pipeline import DataPipeline
        dp = DataPipeline("FD001")
        dp.run()
        del dp
        import gc
        gc.collect()
    else:
        print("[ENTRYPOINT] Primary dataset FD001 present.")

    # Build other subsets if explicitly requested or outside memory-constrained environments
    build_all = os.environ.get("BUILD_ALL_DATASETS", "0") == "1"
    subsets = ["FD002", "FD003", "FD004"] if build_all else []
    for sub in subsets:
        sub_train = ROOT_DIR / "data" / "processed" / sub / "train.parquet"
        if not sub_train.exists():
            print(f"[ENTRYPOINT] Building pipeline for {sub}...")
            from src.features.pipeline import DataPipeline
            dp = DataPipeline(sub)
            dp.run()
            del dp
            import gc
            gc.collect()

    # 3. Train models if bundles are missing
    subsets_to_check = ["FD001", "FD002", "FD003", "FD004"]
    missing_models = [
        sub for sub in subsets_to_check
        if not (ROOT_DIR / "artifacts" / "models" / f"model_bundle_{sub}.joblib").exists()
    ]
    if missing_models:
        print(f"[ENTRYPOINT] Model bundles missing for {missing_models}. Training models...")
        import scripts.train_all_models as train_all_models
        train_all_models.main()
    else:
        print("[ENTRYPOINT] Model bundles present.")

    print("=" * 60)
    print("  AETURNUS: System Artifacts Verified.")
    print("=" * 60)

def main():
    ensure_data_and_models()
    import run_app
    run_app.main()

if __name__ == "__main__":
    main()
