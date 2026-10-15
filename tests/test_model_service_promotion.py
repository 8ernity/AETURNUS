import os
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir
from src.models.model_service import ModelService

def test_train_does_not_write_active_bundle(tmp_path):
    subset = "FD001"
    ms = ModelService(subset=subset)
    test_models_dir = tmp_path / "models"
    test_models_dir.mkdir(parents=True, exist_ok=True)
    ms.artifacts_dir = test_models_dir
    active_bundle_path = test_models_dir / f"model_bundle_{subset}.joblib"
    
    # Synthetic dataframe with sufficient rows and degradation signal to pass train() guards
    n_units = 10
    n_cycles = 50
    units = []
    cycles = []
    ruls = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            units.append(u)
            cycles.append(c)
            ruls.append(float(max(0, n_cycles - c)))

    cols = [f"s{i}" for i in range(1, 22)] + [f"s{i}_norm" for i in range(1, 22)]
    rng = np.random.RandomState(42)
    dummy_data = {c: rng.randn(len(units)) for c in cols}
    # Correlate s2 and s3 with RUL so LightGBM can split nodes effectively
    dummy_data["s2"] = [r * 0.1 + rng.randn() * 0.05 for r in ruls]
    dummy_data["s2_norm"] = dummy_data["s2"]
    dummy_data["s3"] = [r * -0.05 + rng.randn() * 0.05 for r in ruls]
    dummy_data["s3_norm"] = dummy_data["s3"]
    dummy_data["unit"] = units
    dummy_data["cycle"] = cycles
    dummy_data["rul_capped"] = ruls
    dummy_data["fails_within_10"] = [1 if r < 10 else 0 for r in ruls]
    dummy_data["fails_within_30"] = [1 if r < 30 else 0 for r in ruls]
    dummy_data["fails_within_60"] = [1 if r < 60 else 0 for r in ruls]
    df = pd.DataFrame(dummy_data)

    # Call train()
    ms.train(df)

    # Assert active bundle was NOT created by train()
    assert not active_bundle_path.exists(), "train() must NOT create active bundle directly"

    # Now test save with a version
    version_tag = f"test_v{int(time.time())}"
    versioned_path = ms.save(version=version_tag)
    assert versioned_path.exists()
    assert version_tag in str(versioned_path)

    # Active bundle should still NOT exist before promotion
    assert not active_bundle_path.exists()

    # Promote explicitly
    promoted_path = ms.promote(version=version_tag)
    assert promoted_path == active_bundle_path
    assert active_bundle_path.exists()

