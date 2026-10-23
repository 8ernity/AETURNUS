"""README table generation entrypoint.

All metrics tables are rendered from ``artifacts/run_manifest.json`` by
``scripts/render_readme.py``. The legacy report-scraping generators were removed
because they duplicated operating-condition/fault-mode literals and read stale
``artifacts/reports/evaluation_*.json`` files instead of the manifest.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.render_readme import update_or_check_readme


def update_readme() -> bool:
    """Regenerates the README metrics block from the run manifest."""
    return update_or_check_readme(check_only=False)


if __name__ == "__main__":
    update_readme()
