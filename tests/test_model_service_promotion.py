import os
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir
from src.models.model_service import ModelService

def test_train_does_not_write_active_bundle(tmp_path):
    root_dir = get_root_dir()
    subset = "FD001"
    ms = ModelService(subset=subset)
    active_bundle_path = root_dir / "artifacts" / "models" / f"model_bundle_{subset}.joblib"
    
    # Record mtime of active bundle if it exists and backup content
    original_mtime = active_bundle_path.stat().st_mtime if active_bundle_path.exists() else None
    bundle_backup = active_bundle_path.read_bytes() if active_bundle_path.exists() else None

    # Synthetic small dataframe to test train()
    cols = [f"s{i}" for i in range(1, 22)] + [f"s{i}_norm" for i in range(1, 22)]
    dummy_data = {c: np.random.randn(60) for c in cols}
    dummy_data["unit"] = [u for u in range(1, 7) for _ in range(10)]
    dummy_data["cycle"] = list(range(1, 11)) * 6
    dummy_data["rul_capped"] = [float(100 - c) for c in dummy_data["cycle"]]
    dummy_data["fails_within_30"] = [1 if r < 30 else 0 for r in dummy_data["rul_capped"]]
    df = pd.DataFrame(dummy_data)

    versioned_path = None
    try:
        # Call train()
        ms.train(df)

        # Assert active bundle was NOT created or modified by train()
        if original_mtime is not None:
            assert active_bundle_path.stat().st_mtime == original_mtime, "train() must NOT modify active bundle"

        # Now test save with a version
        version_tag = f"test_v{int(time.time())}"
        versioned_path = ms.save(version=version_tag)
        assert versioned_path.exists()
        assert version_tag in str(versioned_path)

        # Active bundle mtime should still be unchanged
        if original_mtime is not None:
            assert active_bundle_path.stat().st_mtime == original_mtime

        # Promote explicitly
        promoted_path = ms.promote(version=version_tag)
        assert promoted_path == active_bundle_path
        assert active_bundle_path.exists()
        if original_mtime is not None:
            assert active_bundle_path.stat().st_mtime > original_mtime
    finally:
        # Clean up test artifact and restore active bundle
        if versioned_path is not None and versioned_path.exists():
            versioned_path.unlink()
        if bundle_backup is not None and active_bundle_path.exists():
            active_bundle_path.write_bytes(bundle_backup)

