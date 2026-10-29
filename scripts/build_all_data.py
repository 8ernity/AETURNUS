import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.pipeline import DataPipeline  # noqa: E402

DEFAULT_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def parse_subsets(argv=None) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Build processed C-MAPSS feature parquets for one or more subsets."
    )
    parser.add_argument(
        "subsets",
        nargs="*",
        help="Subsets to build (default: all four). Example: FD001",
    )
    args = parser.parse_args(argv)
    requested = [item.strip().upper() for item in args.subsets if item.strip()]
    if not requested:
        return list(DEFAULT_SUBSETS)
    invalid = [item for item in requested if item not in DEFAULT_SUBSETS]
    if invalid:
        parser.error(f"unknown subset(s): {', '.join(invalid)}")
    return requested


def main(argv=None):
    subsets = parse_subsets(argv)
    print("=" * 60)
    print(f"Generating Feature & Data Pipeline Artifacts for {', '.join(subsets)}")
    print("=" * 60)

    total_start = time.time()
    for sub in subsets:
        t0 = time.time()
        print(f"\n>>> Processing {sub}...")
        dp = DataPipeline(subset=sub, inject_faults=False)
        dp.run()
        print(f">>> {sub} finished in {time.time() - t0:.2f}s")

    print(
        f"\n[DONE] {len(subsets)} C-MAPSS subset(s) built successfully in "
        f"{time.time() - total_start:.2f}s!"
    )


if __name__ == "__main__":
    main()
