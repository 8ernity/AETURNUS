import pytest
import re
import numpy as np
import pandas as pd
from pathlib import Path
from src.ingest.fault_injector import FaultInjector
from src.clean.cleaner import DataCleaner, SENSOR_REGEX
from src.features.regime import RegimeNormalizer
from src.features.builder import FeatureBuilder
from src.features.pipeline import DataPipeline
from src.common.utils import load_yaml

def test_sensor_regex_excludes_settings():
    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3", "s1", "s2", "s21", "status"]
    sensors = [c for c in cols if SENSOR_REGEX.match(c)]
    assert sensors == ["s1", "s2", "s21"]
    assert "setting_1" not in sensors
    assert "setting_2" not in sensors
    assert "setting_3" not in sensors

def test_fault_injector_only_targets_sensors():
    df = pd.DataFrame({
        "unit": [1, 1, 1],
        "cycle": [1, 2, 3],
        "setting_1": [0.0, 0.0, 0.0],
        "setting_2": [0.0, 0.0, 0.0],
        "setting_3": [100.0, 100.0, 100.0],
        "s1": [10.0, 10.0, 10.0],
        "s2": [20.0, 20.0, 20.0],
    })
    injector = FaultInjector(seed=42)
    faulty = injector.inject_faults(df)
    # Settings should never be modified or NaN-injected
    assert not faulty["setting_1"].isna().any()
    assert not faulty["setting_2"].isna().any()
    assert not faulty["setting_3"].isna().any()

def test_causal_imputation_no_backfill():
    config = load_yaml("configs/pipeline.yaml")
    cleaner = DataCleaner(config)
    cleaner.train_medians = {"s1": 50.0, "s2": 100.0}
    cleaner.kept_sensors = ["s1", "s2"]
    
    # Unit with leading NaN and internal NaN
    df = pd.DataFrame({
        "unit": [1, 1, 1, 1],
        "cycle": [1, 2, 3, 4],
        "s1": [np.nan, 55.0, np.nan, 57.0],
        "s2": [100.0, 102.0, 104.0, 106.0]
    })
    
    cleaned, _ = cleaner.clean_impute(df, is_train=False)
    # Leading NaN must be filled with train median fallback (50.0), NOT backfilled from cycle 2 (55.0)
    assert cleaned.loc[0, "s1"] == 50.0
    # Cycle 3 NaN must be forward-filled from cycle 2 (55.0)
    assert cleaned.loc[2, "s1"] == 55.0
    assert cleaned.loc[3, "s1"] == 57.0

def test_dev_base_computed_on_norm_columns():
    config = load_yaml("configs/pipeline.yaml")
    builder = FeatureBuilder(config)
    builder.baseline_cycles = 2
    
    unit_df = pd.DataFrame({
        "unit": [1, 1, 1],
        "cycle": [1, 2, 3],
        "s2": [10.0, 20.0, 30.0],
        "s2_norm": [-1.0, 0.0, 1.0] # normalized values
    })
    
    result = builder.extract_features_for_unit(unit_df, ["s2"])
    # Mean of first 2 cycles of s2_norm is (-1.0 + 0.0)/2 = -0.5
    # dev_base for cycle 3 should be 1.0 - (-0.5) = 1.5
    assert "s2_dev_base" in result.columns
    assert np.isclose(result.loc[2, "s2_dev_base"], 1.5)

def test_pipeline_fd001_execution():
    dp = DataPipeline(subset="FD001", inject_faults=False)
    train_df, test_df, rul_truth, report = dp.run()
    
    assert "unit" in train_df.columns
    assert "cycle" in train_df.columns
    assert "rul_capped" in train_df.columns
    assert "health_index" in train_df.columns
    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(rul_truth) == 100
    assert "s2_dev_base" in train_df.columns
    assert report["kept_sensors_count"] == 14
