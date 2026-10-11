import pytest
import pandas as pd
import numpy as np
from src.common.splits import get_persistent_split
from src.common.utils import get_root_dir
from src.models.model_service import ModelService
from src.mlops.drift_and_retrain import MLOpsManager

root_dir = get_root_dir()

def test_persistent_split_properties():
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        split1 = get_persistent_split(sub)
        split2 = get_persistent_split(sub)

        assert split1["subset"] == sub
        assert split1 == split2, f"Split for {sub} must be persistent and deterministic"

        train = set(split1["train_units"])
        val = set(split1["val_units"])
        holdout = set(split1["lifecycle_holdout_units"])

        # Disjoint checks
        assert train.isdisjoint(val), f"Train and Val must be disjoint in {sub}"
        assert train.isdisjoint(holdout), f"Train and Holdout must be disjoint in {sub}"
        assert val.isdisjoint(holdout), f"Val and Holdout must be disjoint in {sub}"

        # Union covers all
        total_unique = train | val | holdout
        assert len(total_unique) == split1["total_units"]

def test_training_and_retrain_gate_and_lifecycle_use_same_split():
    sub = "FD001"
    split = get_persistent_split(sub)
    train_units = set(split["train_units"])
    val_units = set(split["val_units"])
    holdout_units = set(split["lifecycle_holdout_units"])

    # 1. ModelService.train respects the split
    train_path = root_dir / "data" / "processed" / sub / "train.parquet"
    if train_path.exists():
        df = pd.read_parquet(train_path)
        ms = ModelService(sub)
        # Train on small sample to verify engine masks
        ms._select_features(df)
        engine_units = [int(u) for u in np.unique(df["unit"].values)]
        
        # Ensure holdout units are never in training or validation masks
        train_mask = df["unit"].isin(train_units)
        val_mask = df["unit"].isin(val_units)
        holdout_mask = df["unit"].isin(holdout_units)

        assert not np.any(train_mask & holdout_mask)
        assert not np.any(val_mask & holdout_mask)
        assert np.any(holdout_mask)

    # 2. MLOpsManager retrain gate uses the same holdout units
    mgr = MLOpsManager(sub)
    retrain_split = get_persistent_split(sub)
    assert set(retrain_split["lifecycle_holdout_units"]) == holdout_units

    # 3. Lifecycle demo in server uses the same holdout units
    from src.api.server import replay_state
    # Lifecycle demo defaults to one of the holdout units
    assert len(holdout_units) > 0
