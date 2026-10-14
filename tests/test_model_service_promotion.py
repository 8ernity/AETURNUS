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
    
    # Synthetic small dataframe to test train()
    cols = [f"s{i}" for i in range(1, 22)] + [f"s{i}_norm" for i in range(1, 22)]
    dummy_data = {c: np.random.randn(60) for c in cols}
    dummy_data["unit"] = [u for u in range(1, 7) for _ in range(10)]
    dummy_data["cycle"] = list(range(1, 11)) * 6
    dummy_data["rul_capped"] = [float(100 - c) for c in dummy_data["cycle"]]
    dummy_data["fails_within_30"] = [1 if r < 30 else 0 for r in dummy_data["rul_capped"]]
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

