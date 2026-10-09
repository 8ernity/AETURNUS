import time
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir
from src.models.model_service import ModelService

def main():
    root_dir = get_root_dir()
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    print("=" * 60)
    print("Training Real Models for All Subsets (FD001 - FD004)")
    print("=" * 60)

    for sub in subsets:
        t0 = time.time()
        train_path = root_dir / "data" / "processed" / sub / "train.parquet"
        if not train_path.exists():
            print(f"Processed training data missing for {sub}. Building pipeline first...")
            from src.features.pipeline import DataPipeline
            DataPipeline(sub).run()

        train_df = pd.read_parquet(train_path)
        ms = ModelService(subset=sub)
        print(f"\n>>> Training {sub} on {len(train_df)} rows...")
        ms.train(train_df)
        print(f">>> {sub} trained in {time.time() - t0:.2f}s")

    print("\n[DONE] All models successfully trained and persisted.")

if __name__ == "__main__":
    main()
