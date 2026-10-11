import os
import sys

# Constrain thread pools and glibc memory arenas to prevent memory explosion on 512MB cloud instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import uvicorn
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    from scripts.entrypoint import ensure_data_and_models
    ensure_data_and_models()

    print("=" * 60)
    print("  AETURNUS: Autonomous Equipment Tracking and Uptime Restoration")
    print("  Networked Unified Systems · NASA C-MAPSS Benchmark")
    print("=" * 60)
    print("Starting FastAPI backend and Web Dashboard...")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f"  Bound to:   http://{host}:{port}")
    print("=" * 60)
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        limit_concurrency=40,
        timeout_keep_alive=15
    )

if __name__ == "__main__":
    main()
