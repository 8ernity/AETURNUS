import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from src.common.utils import get_root_dir

def get_persistent_split(
    subset: str = "FD001",
    all_units: Optional[List[int]] = None,
    split_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Returns the persistent engine-level split (train / val / lifecycle-holdout) for the subset.
    If the split file does not exist, it deterministically partitions all_units and saves it.
    """
    root_dir = get_root_dir()
    target_dir = split_dir or (root_dir / "artifacts" / "splits")
    target_dir.mkdir(parents=True, exist_ok=True)
    split_file = target_dir / f"engine_splits_{subset}.json"

    if split_file.exists():
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "train_units" in data and "val_units" in data and "lifecycle_holdout_units" in data:
                return data
        except Exception:
            pass

    # Generate deterministic split if absent
    if all_units is None:
        train_path = root_dir / "data" / "processed" / subset / "train.parquet"
        if train_path.exists():
            df = pd.read_parquet(train_path)
            all_units = sorted([int(u) for u in df["unit"].unique()])
        else:
            all_units = list(range(1, 101))

    units = sorted(list(set(int(u) for u in all_units)))
    n_units = len(units)

    if n_units <= 2:
        train_units = [units[0]]
        val_units = [units[-1]]
        lifecycle_holdout_units = [units[-1]]
    elif n_units <= 4:
        train_units = units[:2]
        val_units = [units[2]]
        lifecycle_holdout_units = [units[-1]]
    else:
        # Standard: 70% train, 15% val, 15% lifecycle holdout
        n_train = max(1, int(n_units * 0.70))
        n_val = max(1, int(n_units * 0.15))
        train_units = units[:n_train]
        val_units = units[n_train : n_train + n_val]
        lifecycle_holdout_units = units[n_train + n_val :]

    split_data = {
        "subset": subset,
        "total_units": n_units,
        "train_units": train_units,
        "val_units": val_units,
        "lifecycle_holdout_units": lifecycle_holdout_units
    }

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    return split_data
