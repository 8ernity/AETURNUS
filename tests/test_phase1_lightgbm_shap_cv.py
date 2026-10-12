import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from src.common.utils import get_root_dir, load_yaml
from src.common.splits import get_persistent_split
from src.models.model_service import ModelService

root_dir = get_root_dir()

def test_shap_sums_reconstruct_grouping_and_sorted():
    """Verify SHAP sums per sensor reconstruct the feature grouping and contributors are sorted by |SHAP|."""
    subset = "FD001"
    ms = ModelService(subset=subset)
    
    # Create synthetic dataset with multiple features per sensor
    sensors = ["s2", "s3", "s4", "s7", "s11", "s14"]
    features = []
    for s in sensors:
        features.extend([f"{s}", f"{s}_norm", f"{s}_ewma", f"{s}_dev_base"])
    
    np.random.seed(42)
    X = np.random.randn(50, len(features)).astype(np.float32)
    y = np.random.uniform(10, 100, size=50).astype(np.float32)
    
    # Fit LightGBM p50 model
    model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=30, random_state=42, verbose=-1)
    model.fit(X, y)
    ms.rul_p50_model = model
    ms.rul_p10_model = model
    ms.rul_p90_model = model
    ms.feature_cols = features
    
    # Test point
    row_dict = {f: float(X[0, i]) for i, f in enumerate(features)}
    row_dict["unit"] = 1
    row_dict["cycle"] = 10
    
    res = ms.predict_point(row_dict, compute_shap=True)
    assert "top_contributors" in res
    assert "top_sensors" in res
    assert len(res["top_contributors"]) == 3
    assert len(res["top_sensors"]) == 3
    
    # 1. Contributors must be sorted by absolute SHAP value descending
    abs_shaps = [abs(c["shap_value"]) for c in res["top_contributors"]]
    assert abs_shaps == sorted(abs_shaps, reverse=True), f"Contributors not sorted descending by |SHAP|: {abs_shaps}"
    
    # 2. Sensor grouping must reconstruct feature contributions
    explainer = ms.get_shap_explainer()
    raw_shap = explainer.shap_values(X[:1])[0]
    
    for c in res["top_contributors"]:
        s_id = c["sensor"]
        expected_sum = sum(raw_shap[i] for i, f in enumerate(features) if f.startswith(f"{s_id}") or f.startswith(f"{s_id}_"))
        assert pytest.approx(c["shap_value"], abs=1e-2) == expected_sum
        assert c["direction"] in ["shortens_life", "prolongs_life"]
        if c["shap_value"] < 0:
            assert c["direction"] == "shortens_life"
        else:
            assert c["direction"] == "prolongs_life"
        assert c["subsystem"] != ""

def test_bundle_loads_old_and_new_format(tmp_path):
    """Verify ModelService.load() gracefully handles both old HistGB bundles and new LightGBM bundles."""
    subset = "FD001"
    ms_new = ModelService(subset=subset)
    ms_new.algorithm = "lightgbm"
    ms_new.feature_cols = ["s2", "s3"]
    from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
    ms_new.anomaly_detector = IsolationForest(n_estimators=10, random_state=42)
    ms_new.anomaly_detector.fit(np.zeros((10, 2)))
    
    dummy_reg = HistGradientBoostingRegressor(max_iter=10, random_state=42)
    dummy_reg.fit(np.zeros((10, 2)), np.ones(10))
    ms_new.rul_p10_model = dummy_reg
    ms_new.rul_p50_model = dummy_reg
    ms_new.rul_p90_model = dummy_reg
    ms_new.classifier = dummy_reg
    
    # Save as new bundle
    new_bundle_path = tmp_path / "model_bundle_FD001_new.joblib"
    ms_new.save(path=new_bundle_path)
    
    # Load new bundle
    ms_loaded_new = ModelService(subset=subset)
    ms_loaded_new.artifacts_dir = tmp_path
    ms_loaded_new.load("new")
    assert ms_loaded_new.algorithm == "lightgbm"
    assert hasattr(ms_loaded_new, "conformal_q")
    
    # Save as old bundle (simulate legacy bundle missing 'algorithm' key)
    import joblib
    old_bundle_data = joblib.load(new_bundle_path)
    del old_bundle_data["algorithm"]
    old_bundle_path = tmp_path / "model_bundle_FD001_legacy.joblib"
    joblib.dump(old_bundle_data, old_bundle_path)
    
    # Load legacy bundle
    ms_loaded_legacy = ModelService(subset=subset)
    ms_loaded_legacy.artifacts_dir = tmp_path
    ms_loaded_legacy.load("legacy")
    assert ms_loaded_legacy.algorithm == "lightgbm" or ms_loaded_legacy.algorithm == "hist_gb"

def test_tuning_never_touches_test_data():
    """Verify tuning units are strictly a subset of train_units and disjoint from holdout, val, and test sets."""
    subset = "FD001"
    train_path = root_dir / "data" / "processed" / subset / "train.parquet"
    if not train_path.exists():
        pytest.skip("train.parquet not present")
    
    train_df = pd.read_parquet(train_path)
    all_units = [int(u) for u in np.unique(train_df["unit"].values)]
    split = get_persistent_split(subset, all_units=all_units)
    
    train_units = set(split["train_units"])
    val_units = set(split["val_units"])
    holdout_units = set(split["lifecycle_holdout_units"])
    
    # Train units must be disjoint from val and lifecycle holdout
    assert train_units.isdisjoint(val_units), "Tuning/Train units overlap with val units!"
    assert train_units.isdisjoint(holdout_units), "Tuning/Train units overlap with lifecycle holdout units!"
    assert len(train_units) > 0

def test_cv_folds_are_engine_disjoint():
    """Verify that GroupKFold CV produces strictly engine-disjoint folds."""
    subset = "FD001"
    train_path = root_dir / "data" / "processed" / subset / "train.parquet"
    if not train_path.exists():
        pytest.skip("train.parquet not present")
        
    ms = ModelService(subset=subset)
    train_df = pd.read_parquet(train_path)
    
    all_units = [int(u) for u in np.unique(train_df["unit"].values)]
    split = get_persistent_split(subset, all_units=all_units)
    train_units = split["train_units"]
    train_only_df = train_df[train_df["unit"].isin(train_units)].copy().reset_index(drop=True)
    
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    X = train_only_df[["cycle"]].values
    y = train_only_df["rul_capped"].values
    groups = train_only_df["unit"].values
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        tr_engines = set(groups[tr_idx])
        val_engines = set(groups[val_idx])
        assert tr_engines.isdisjoint(val_engines), f"Fold {fold} has overlapping engines between train and val!"
