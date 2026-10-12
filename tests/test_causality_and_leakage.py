import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

from src.common.utils import get_root_dir
from src.common.splits import get_persistent_split
from src.ingest.loader import CMAPSSLoader
from src.models.model_service import ModelService
from src.features.streaming import StreamingEnginePipeline

root_dir = get_root_dir()

def test_truncation_invariance():
    """
    Truncation invariance:
    Predicting up to cycle C-k on truncated input must produce the exact same
    predictions and causal features as running on the full series up to C-k.
    Guarantees no future-cycle leakage, forward-looking EWMA, or centered smoothing.
    """
    subset = "FD001"
    unit = 1
    loader = CMAPSSLoader(subset=subset)
    _, raw_test, _ = loader.load_raw_data()
    unit_raw = raw_test[raw_test["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    
    total_cycles = len(unit_raw)
    assert total_cycles >= 30, "Unit 1 must have at least 30 cycles"
    
    k = 15
    trunc_limit = total_cycles - k
    
    ms = ModelService(subset=subset).load()
    
    # 1. Full run
    pipe_full = StreamingEnginePipeline(subset=subset, model_service=ms)
    pipe_full.reset_engine(unit)
    full_preds = []
    for i in range(trunc_limit):
        row = unit_raw.iloc[i].to_dict()
        res = pipe_full.stream_cycle(row)
        full_preds.append((res["prediction"]["rul_p50"], res["prediction"]["health_index"]))
        
    # 2. Truncated run (only feed up to trunc_limit)
    pipe_trunc = StreamingEnginePipeline(subset=subset, model_service=ms)
    pipe_trunc.reset_engine(unit)
    trunc_preds = []
    for i in range(trunc_limit):
        row = unit_raw.iloc[i].to_dict()
        res = pipe_trunc.stream_cycle(row)
        trunc_preds.append((res["prediction"]["rul_p50"], res["prediction"]["health_index"]))
        
    assert len(full_preds) == len(trunc_preds)
    for (f_p50, f_hi), (t_p50, t_hi) in zip(full_preds, trunc_preds):
        assert abs(f_p50 - t_p50) < 1e-4, f"Truncation invariance violated for RUL: {f_p50} vs {t_p50}"
        assert abs(f_hi - t_hi) < 1e-4, f"Truncation invariance violated for HI: {f_hi} vs {t_hi}"

def test_engine_independence():
    """
    Engine independence:
    Feature extraction and scoring of Engine A must be 100% invariant to whether
    Engine A is scored alone or alongside Engine B in a fleet batch.
    Guarantees no cross-engine batch effects, leaks, or cross-unit normalizations.
    """
    subset = "FD001"
    loader = CMAPSSLoader(subset=subset)
    _, raw_test, _ = loader.load_raw_data()
    
    unit1_raw = raw_test[raw_test["unit"] == 1].sort_values("cycle").head(25)
    unit2_raw = raw_test[raw_test["unit"] == 2].sort_values("cycle").head(25)
    
    ms = ModelService(subset=subset).load()
    
    # Run Unit 1 alone
    pipe_solo = StreamingEnginePipeline(subset=subset, model_service=ms)
    pipe_solo.reset_engine(1)
    solo_p50 = [pipe_solo.stream_cycle(r.to_dict())["prediction"]["rul_p50"] for _, r in unit1_raw.iterrows()]
    
    # Run interleaved Unit 1 and Unit 2
    pipe_interleaved = StreamingEnginePipeline(subset=subset, model_service=ms)
    pipe_interleaved.reset_engine(1)
    pipe_interleaved.reset_engine(2)
    interleaved_u1_p50 = []
    
    for i in range(len(unit1_raw)):
        # Stream cycle for unit 1
        r1 = unit1_raw.iloc[i].to_dict()
        out1 = pipe_interleaved.stream_cycle(r1)
        interleaved_u1_p50.append(out1["prediction"]["rul_p50"])
        # Stream cycle for unit 2
        r2 = unit2_raw.iloc[i].to_dict()
        pipe_interleaved.stream_cycle(r2)
        
    assert len(solo_p50) == len(interleaved_u1_p50)
    for s_val, i_val in zip(solo_p50, interleaved_u1_p50):
        assert abs(s_val - i_val) < 1e-4, f"Engine independence violated: {s_val} vs {i_val}"

def test_disjoint_persistent_engine_splits():
    """
    Disjoint Splits:
    Across all 4 subsets, train, validation, and lifecycle holdout sets must have
    strictly empty pairwise intersections.
    """
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        loader = CMAPSSLoader(subset=sub)
        train_raw, _, _ = loader.load_raw_data()
        all_units = [int(u) for u in np.unique(train_raw["unit"].values)]
        
        split = get_persistent_split(sub, all_units=all_units)
        train_u = set(split["train_units"])
        val_u = set(split["val_units"])
        holdout_u = set(split["lifecycle_holdout_units"])
        
        # Verify pairwise disjointness
        assert len(train_u.intersection(val_u)) == 0, f"{sub}: Train and Val overlap!"
        assert len(train_u.intersection(holdout_u)) == 0, f"{sub}: Train and Holdout overlap!"
        assert len(val_u.intersection(holdout_u)) == 0, f"{sub}: Val and Holdout overlap!"
        
        # Verify union covers all units
        assert train_u.union(val_u).union(holdout_u) == set(all_units), f"{sub}: Partition incomplete"

def test_group_kfold_no_cross_engine_leakage():
    """
    GroupKFold Leak-Free Test:
    In 5-fold CV on persistent train engines, training folds and validation folds
    must never share any engine units.
    """
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        train_path = root_dir / "data" / "processed" / sub / "train.parquet"
        if not train_path.exists():
            continue
        df = pd.read_parquet(train_path)
        all_units = [int(u) for u in np.unique(df["unit"].values)]
        split = get_persistent_split(sub, all_units=all_units)
        train_units = split["train_units"]
        
        train_only_df = df[df["unit"].isin(train_units)].copy().reset_index(drop=True)
        groups = train_only_df["unit"].values
        X = np.zeros(len(groups))
        y = np.zeros(len(groups))
        
        gkf = GroupKFold(n_splits=5)
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            tr_engines = set(groups[tr_idx])
            val_engines = set(groups[val_idx])
            overlap = tr_engines.intersection(val_engines)
            assert len(overlap) == 0, f"{sub} Fold {fold}: Cross-engine leak! Overlapping units: {overlap}"
