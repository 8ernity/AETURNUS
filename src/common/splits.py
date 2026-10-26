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
    Returns the persistent engine-level split for the subset:

        train / calibration / validation / lifecycle-holdout

    Calibration engines are disjoint from validation engines: conformal interval
    offsets are learned only on ``calib_units`` while validation engines remain
    available for threshold tuning and promotion gates.  If an older split file
    lacks ``calib_units``, the existing validation group is deterministically
    divided in half and the file is rewritten in place (train and holdout engines
    are never changed).
    """
    root_dir = get_root_dir()
    target_dir = split_dir or (root_dir / "artifacts" / "splits")
    target_dir.mkdir(parents=True, exist_ok=True)
    split_file = target_dir / f"engine_splits_{subset}.json"

    if split_file.exists():
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            required = {"train_units", "val_units", "lifecycle_holdout_units"}
            if required.issubset(data):
                if "calib_units" not in data:
                    # Backward-compatible migration: carve calibration engines out
                    # of the old validation group without touching train/holdout.
                    old_val = [int(u) for u in data["val_units"]]
                    n_calib = max(1, len(old_val) // 2) if old_val else 0
                    data["calib_units"] = old_val[:n_calib]
                    data["val_units"] = old_val[n_calib:]
                    with open(split_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
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
        val_group = [units[-1]]
        lifecycle_holdout_units = [units[-1]]
    elif n_units <= 4:
        train_units = units[:2]
        val_group = [units[2]]
        lifecycle_holdout_units = [units[-1]]
    else:
        # Standard: 70% train, 15% validation group, 15% lifecycle holdout.
        # The validation group is split into disjoint calibration and validation
        # engines (calibration first, preserving deterministic ordering).
        n_train = max(1, int(n_units * 0.70))
        n_val = max(1, int(n_units * 0.15))
        train_units = units[:n_train]
        val_group = units[n_train : n_train + n_val]
        lifecycle_holdout_units = units[n_train + n_val :]

    n_calib = max(1, len(val_group) // 2) if val_group else 0
    calib_units = val_group[:n_calib]
    val_units = val_group[n_calib:]

    split_data = {
        "subset": subset,
        "total_units": n_units,
        "train_units": train_units,
        "calib_units": calib_units,
        "val_units": val_units,
        "lifecycle_holdout_units": lifecycle_holdout_units
    }

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    return split_data
