import pytest
import numpy as np
import pandas as pd
from src.models.neural_baseline import (
    WindowedCNN1DRegressor,
    WindowedGRURegressor,
    NeuralRULBenchmark
)
from src.common.utils import get_root_dir

root_dir = get_root_dir()

def test_windowed_neural_models_shapes():
    batch_size = 16
    window_size = 30
    num_features = 14

    # Dummy inputs: (batch, seq_len=30, features=14)
    x = np.random.randn(batch_size, window_size, num_features).astype(np.float32)
    import torch
    x_tensor = torch.from_numpy(x)

    # 1. CNN
    cnn = WindowedCNN1DRegressor(num_features=num_features, window_size=window_size)
    cnn.eval()
    with torch.no_grad():
        out_cnn = cnn(x_tensor)
    assert out_cnn.shape == (batch_size,)

    # 2. GRU
    gru = WindowedGRURegressor(num_features=num_features, hidden_dim=32, num_layers=2)
    gru.eval()
    with torch.no_grad():
        out_gru = gru(x_tensor)
    assert out_gru.shape == (batch_size,)

def test_neural_benchmark_training_and_evaluation():
    # Use real processed FD001 data
    train_path = root_dir / "data" / "processed" / "FD001" / "train.parquet"
    test_path = root_dir / "data" / "processed" / "FD001" / "test.parquet"

    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feature_cols = [c for c in train_df.columns if c.endswith("_norm") or c.endswith("_ewma")][:10]
    # Filter to first 5 units for quick test
    sub_train = train_df[train_df["unit"] <= 5].copy()
    sub_test = test_df[test_df["unit"] <= 5].copy()
    y_test_truth = np.array([112.0, 98.0, 69.0, 82.0, 91.0], dtype=np.float32)

    bench = NeuralRULBenchmark(feature_cols=feature_cols, window_size=30, seed=42)

    # 1. Sequence builder test
    bench.scaler.fit(sub_train[feature_cols].values)
    X_train, y_train = bench.build_windowed_sequences(sub_train, is_test=False)
    X_test, y_test = bench.build_windowed_sequences(sub_test, is_test=True, rul_truth=y_test_truth)

    assert X_train.shape[1:] == (30, len(feature_cols))
    assert X_test.shape == (5, 30, len(feature_cols))
    assert len(y_test) == 5

    # 2. Fast 2-epoch evaluation of CNN and GRU
    cnn_res = bench.evaluate_architecture("cnn", sub_train, sub_test, y_test_truth, epochs=2)
    assert cnn_res["architecture"] == "CNN"
    assert cnn_res["window_size"] == 30
    assert cnn_res["rmse"] > 0
    assert cnn_res["train_time_sec"] > 0

    gru_res = bench.evaluate_architecture("gru", sub_train, sub_test, y_test_truth, epochs=2)
    assert gru_res["architecture"] == "GRU"
    assert gru_res["window_size"] == 30
    assert gru_res["rmse"] > 0
    assert gru_res["train_time_sec"] > 0
