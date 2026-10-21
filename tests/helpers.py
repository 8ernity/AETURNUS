import json
from pathlib import Path
from typing import Any, Dict

def load_manifest(manifest_path: str = "artifacts/run_manifest.json") -> Dict[str, Any]:
    """Loads and returns the run manifest JSON."""
    p = Path(manifest_path)
    if not p.is_absolute():
        # find root dir relative to this file
        root = Path(__file__).resolve().parent.parent
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
