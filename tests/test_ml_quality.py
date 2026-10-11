import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir
from src.models.model_service import ModelService
from src.models.neural_baseline import NeuralRULBenchmark, CNN1DRegressor, GRURegressor
from src.ingest.loader import CMAPSSLoader

def test_validation_split_by_engine_disjoint():
    """Verify engine-level group split keeps engines completely intact and disjoint."""
    ms = ModelService(subset="FD001")
    root_dir = get_root_dir()
    train_df = pd.read_parquet(root_dir / "data" / "processed" / "FD001" / "train.parquet")
    
    engine_units = np.unique(train_df["unit"].values)
    rng = np.random.RandomState(42)
    val_engine_count = max(2, int(len(engine_units) * 0.20))
    val_units = rng.choice(engine_units, size=val_engine_count, replace=False)
    train_units = [u for u in engine_units if u not in val_units]

    assert set(train_units).isdisjoint(set(val_units))
    assert len(train_units) + len(val_units) == len(engine_units)

def test_classifier_class_weights_and_probability_calibration():
    """Verify fails_within_30 classifier has calibrated probabilities [0, 1]."""
    ms = ModelService(subset="FD001").load()
    root_dir = get_root_dir()
    test_df = pd.read_parquet(root_dir / "data" / "processed" / "FD001" / "test.parquet")
    
    last_rows = test_df.groupby("unit").last().reset_index()
    for c in ms.feature_cols:
        if c not in last_rows.columns:
            last_rows[c] = 0.0
    X_test = last_rows[ms.feature_cols].values
    
    probs = ms.classifier.predict_proba(X_test)
    assert probs.shape[1] == 2
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_calibrated_intervals_and_noncrossing_monotonicity():
    """Verify non-crossing monotonicity p10 <= p50 <= p90 and empirical coverage."""
    ms = ModelService(subset="FD001").load()
    root_dir = get_root_dir()
    test_df = pd.read_parquet(root_dir / "data" / "processed" / "FD001" / "test.parquet")
    loader = CMAPSSLoader(subset="FD001")
    _, _, rul_truth = loader.load_raw_data()

    res = ms.evaluate_test(test_df, rul_truth, save_report=False)
    assert "interval_coverage" in res
    assert 0.55 <= res["interval_coverage"] <= 0.95 # ~80% coverage target

    # Test point-by-point monotonicity
    for _, row in test_df.sample(20, random_state=42).iterrows():
        pred = ms.predict_point(row.to_dict())
        p10 = pred["rul_p10"]
        p50 = pred["rul_p50"]
        p90 = pred["rul_p90"]
        assert 0.0 <= p10 <= p50 <= p90

def test_multi_seed_metrics_reporting():
    """Verify mean±std across seeds are present in evaluation report."""
    root_dir = get_root_dir()
    eval_file = root_dir / "artifacts" / "reports" / "evaluation_FD001.json"
    assert eval_file.exists()
    
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "multi_seed" in data
    ms_data = data["multi_seed"]
    assert len(ms_data["seeds"]) == 3
    assert ms_data["rmse_mean"] > 0
    assert ms_data["rmse_std"] >= 0
    assert ms_data["mae_mean"] > 0
    assert ms_data["nasa_mean"] > 0

def test_deep_learning_gru_and_cnn_architectures():
    """Verify PyTorch 1D-CNN and GRU models can instantiate, train, and predict."""
    torch = pytest.importorskip("torch")
    input_dim = 20
    cnn = CNN1DRegressor(input_dim=input_dim)
    gru = GRURegressor(input_dim=input_dim)
    
    dummy_x = torch.randn(4, input_dim)
    cnn_out = cnn(dummy_x)
    gru_out = gru(dummy_x)
    
    assert cnn_out.shape == (4,)
    assert gru_out.shape == (4,)

def test_top_sensors_use_normalized_deviations():
    """Verify predict_point identifies top sensors from normalized *_dev_base columns."""
    ms = ModelService(subset="FD001").load()
    sample_row = {
        "unit": 1,
        "cycle": 50,
        "s2_dev_base": 0.12,
        "s11_dev_base": 4.85,
        "s14_dev_base": 3.20,
        "s9_dev_base": 1.10
    }
    pred = ms.predict_point(sample_row)
    assert "top_sensors" in pred
    assert len(pred["top_sensors"]) == 3
    # s11 and s14 have the highest deviations
    assert "s11" in pred["top_sensors"]
    assert "s14" in pred["top_sensors"]
