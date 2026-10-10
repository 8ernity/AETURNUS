import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.features.streaming import StreamingEnginePipeline
from src.models.model_service import ModelService
from src.ingest.loader import CMAPSSLoader
from src.common.utils import get_root_dir

root_dir = get_root_dir()

def test_stream_vs_batch_predictions_within_1_cycle():
    subset = "FD001"
    unit = 1

    # 1. Load batch processed test data
    test_proc_path = root_dir / "data" / "processed" / subset / "test.parquet"
    assert test_proc_path.exists(), "Processed test.parquet must exist"
    test_proc_df = pd.read_parquet(test_proc_path)
    engine_batch_df = test_proc_df[test_proc_df["unit"] == unit].sort_values("cycle").reset_index(drop=True)

    # Load model
    ms = ModelService(subset=subset).load()

    # 2. Load raw test data for unit 1
    loader = CMAPSSLoader(subset=subset)
    _, raw_test, _ = loader.load_raw_data()
    raw_engine_df = raw_test[raw_test["unit"] == unit].sort_values("cycle").reset_index(drop=True)

    # 3. Initialize streaming pipeline
    streaming_pipeline = StreamingEnginePipeline(subset=subset, model_service=ms)
    streaming_pipeline.reset_engine(unit)

    # Stream through all cycles for this engine
    max_cycles_to_test = min(len(raw_engine_df), len(engine_batch_df))
    assert max_cycles_to_test >= 30, "Engine should have at least 30 cycles"

    rul_diffs = []

    for i in range(max_cycles_to_test):
        raw_row = raw_engine_df.iloc[i].to_dict()
        batch_row = engine_batch_df.iloc[i].to_dict()

        # Stream one cycle
        stream_out = streaming_pipeline.stream_cycle(raw_row)
        stream_p50 = stream_out["prediction"]["rul_p50"]

        # Batch prediction
        batch_pred = ms.predict_point(batch_row)
        batch_p50 = batch_pred["rul_p50"]

        diff = abs(stream_p50 - batch_p50)
        rul_diffs.append(diff)

        # Assert stream vs batch prediction is strictly within 1 cycle
        assert diff <= 1.0, (
            f"Cycle {i+1}: Stream prediction ({stream_p50}) and Batch prediction ({batch_p50}) "
            f"diverged by {diff:.4f} cycles (> 1.0 cycle threshold)"
        )

    # Verify per-engine running state
    engine_state = streaming_pipeline.get_or_create_engine_state(unit)
    assert len(engine_state.history) == max_cycles_to_test
    assert len(engine_state.baselines) > 0
    assert len(engine_state.ewma) > 0
    assert len(engine_state.rolling_windows) > 0

    mean_diff = float(np.mean(rul_diffs))
    max_diff = float(np.max(rul_diffs))
    print(f"\n[PASS] Tested {max_cycles_to_test} cycles: Mean RUL diff = {mean_diff:.4f}, Max RUL diff = {max_diff:.4f}")
    assert max_diff <= 1.0
