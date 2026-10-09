import sys
import time
from pathlib import Path
from src.features.pipeline import DataPipeline

def main():
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    print("=" * 60)
    print("Generating Feature & Data Pipeline Artifacts for FD001 - FD004")
    print("=" * 60)
    
    total_start = time.time()
    for sub in subsets:
        t0 = time.time()
        print(f"\n>>> Processing {sub}...")
        dp = DataPipeline(subset=sub, inject_faults=False)
        dp.run()
        print(f">>> {sub} finished in {time.time() - t0:.2f}s")
        
    print(f"\n[DONE] All 4 C-MAPSS subsets built successfully in {time.time() - total_start:.2f}s!")

if __name__ == "__main__":
    main()
