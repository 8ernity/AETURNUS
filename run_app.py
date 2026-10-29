import os
import subprocess
import sys
from pathlib import Path

# Constrain thread pools and glibc memory arenas to prevent memory explosion on 512MB cloud instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent


def bootstrap_artifacts() -> None:
    """Ensure data and models exist before importing the API server.

    On a 512 MB host, building the FD001 feature pipeline in the same Python
    process that later imports FastAPI, pandas, LightGBM and the model bundle
    leaves enough allocator-retained memory behind to trigger an OOM.  Running
    the bootstrap in a short-lived child process releases that memory before
    Uvicorn starts.  Set AETURNUS_SKIP_BOOTSTRAP=1 when artifacts are prepared
    by the deploy build step.
    """
    if os.environ.get("AETURNUS_SKIP_BOOTSTRAP", "0").lower() in ("1", "true", "yes"):
        print("[BOOTSTRAP] Skipped by AETURNUS_SKIP_BOOTSTRAP.", flush=True)
        return

    if os.environ.get("AETURNUS_BOOTSTRAP_IN_PROCESS", "0").lower() in ("1", "true", "yes"):
        from scripts.entrypoint import ensure_data_and_models

        ensure_data_and_models()
        return

    command = [
        sys.executable,
        "-c",
        "from scripts.entrypoint import ensure_data_and_models; ensure_data_and_models()",
    ]
    print("[BOOTSTRAP] Verifying artifacts in a child process...", flush=True)
    subprocess.run(command, cwd=str(ROOT_DIR), check=True)


def main(bootstrap: bool = True) -> None:
    if bootstrap:
        bootstrap_artifacts()

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
