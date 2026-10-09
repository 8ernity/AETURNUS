import pytest
import numpy as np
import pandas as pd
from src.clean.cleaner import DataCleaner, SENSOR_REGEX

@pytest.fixture
def sample_cleaner_config():
    return {
        "cleaning": {
            "ffill_limit": 3,
            "hampel_window": 5,
            "hampel_sigma": 3.0,
            "outlier_z_clip": 4.0,
            "flatline_window": 10,
            "flatline_var_threshold": 1e-5,
            "constant_sensor_var_threshold": 1e-4
        }
    }

def test_cleaner_regex_preserves_settings(sample_cleaner_config):
    cleaner = DataCleaner(sample_cleaner_config)
    cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3", "s1", "s2", "s21"]
    df = pd.DataFrame(np.random.randn(20, len(cols)), columns=cols)
    sensors = cleaner.get_sensor_cols(df)
    assert sensors == ["s1", "s2", "s21"]
    for s in ["setting_1", "setting_2", "setting_3", "unit", "cycle"]:
        assert s not in sensors

def test_cleaner_causal_imputation_no_backfill(sample_cleaner_config):
    cleaner = DataCleaner(sample_cleaner_config)
    # Unit 1 starts with NaN at cycle 1
    # Future cycle 2 has 100.0
    data = {
        "unit": [1, 1, 1, 1],
        "cycle": [1, 2, 3, 4],
        "setting_1": [0.0, 0.0, 0.0, 0.0],
        "s2": [np.nan, 100.0, np.nan, 102.0],
        "s3": [50.0, 50.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)
    cleaner.fit_train(pd.DataFrame({"s2": [90.0, 95.0, 100.0], "s3": [50.0, 51.0, 49.0]}))
    
    cleaned, report = cleaner.clean_impute(df, is_train=False)
    # Cycle 1 had NaN: must NOT backfill from cycle 2 (100.0)
    # Instead, must fallback to train median (95.0)
    assert cleaned.loc[cleaned["cycle"] == 1, "s2"].values[0] == 95.0
    # Cycle 3 had NaN: causal forward fill from cycle 2 (100.0)
    assert cleaned.loc[cleaned["cycle"] == 3, "s2"].values[0] == 100.0

def test_cleaner_hampel_and_outlier_clipping(sample_cleaner_config):
    cleaner = DataCleaner(sample_cleaner_config)
    # Series with a massive outlier spike
    vals = [1.0, 1.2, 1.1, 999.0, 1.0, 1.3, 1.1]
    df = pd.DataFrame({
        "unit": [1] * len(vals),
        "cycle": list(range(1, len(vals) + 1)),
        "s2_norm": vals
    })
    clipped, report = cleaner.clip_outliers_normalized(df, ["s2"])
    # The spike at index 3 should be clamped to outlier_z_clip (4.0)
    assert clipped.loc[3, "s2_norm"] == 4.0
    assert report["outliers_cleaned"] == 1


def test_cleaner_constant_sensors_dropped(sample_cleaner_config):
    cleaner = DataCleaner(sample_cleaner_config)
    df = pd.DataFrame({
        "unit": [1, 1, 1, 1],
        "cycle": [1, 2, 3, 4],
        "s1": [100.0, 100.0, 100.0, 100.0], # variance 0.0
        "s2": [10.0, 15.0, 12.0, 18.0]       # high variance
    })
    kept = cleaner.fit_train(df)
    assert "s1" not in kept
    assert "s2" in kept
