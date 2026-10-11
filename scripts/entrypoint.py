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
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    missing_processed = [
        sub for sub in subsets
        if not (ROOT_DIR / "data" / "processed" / sub / "train.parquet").exists()
    ]
    if missing_processed:
        print(f"[ENTRYPOINT] Processed data missing for {missing_processed}. Building pipelines...")
        import scripts.build_all_data as build_all_data
        build_all_data.main()
    else:
        print("[ENTRYPOINT] Processed datasets present.")

    # 3. Train models if bundles are missing
    missing_models = [
        sub for sub in subsets
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
