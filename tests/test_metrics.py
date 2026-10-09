import numpy as np
import pytest
from src.common.utils import calculate_nasa_score, calculate_metrics, calculate_psi

def test_nasa_asymmetric_scoring():
    # Perfect predictions
    assert calculate_nasa_score([100.0, 50.0], [100.0, 50.0]) == 0.0

    # Early prediction (diff = -10) vs Late prediction (diff = +10)
    # NASA score penalizes late predictions much more severely
    early_score = calculate_nasa_score([50.0], [40.0])  # d = -10: e^(10/13) - 1 ≈ 1.158
    late_score = calculate_nasa_score([40.0], [50.0])   # d = +10: e^(10/10) - 1 ≈ 1.718
    
    assert early_score > 0
    assert late_score > 0
    assert late_score > early_score
    assert abs(early_score - (np.exp(10.0 / 13.0) - 1.0)) < 1e-4
    assert abs(late_score - (np.exp(10.0 / 10.0) - 1.0)) < 1e-4

def test_calculate_metrics_correctness():
    y_true = np.array([100.0, 50.0, 30.0, 20.0])
    y_pred = np.array([104.0, 47.0, 30.0, 16.0]) # residuals: +4, -3, 0, -4
    
    res = calculate_metrics(y_true, y_pred)
    assert "rmse" in res
    assert "mae" in res
    assert "nasa_score" in res
    assert "late_rate" in res
    assert "early_rate" in res

    expected_mae = (4.0 + 3.0 + 0.0 + 4.0) / 4.0  # 2.75
    expected_rmse = np.sqrt((16.0 + 9.0 + 0.0 + 16.0) / 4.0) # sqrt(10.25) ≈ 3.2016
    assert abs(res["mae"] - expected_mae) < 1e-3
    assert abs(res["rmse"] - expected_rmse) < 1e-3
    assert res["late_rate"] == 0.25 # 1 of 4 is positive
    assert res["early_rate"] == 0.50 # 2 of 4 are negative

def test_calculate_psi_drift_detection():
    # Identical reference and actual distributions
    ref = np.random.normal(loc=100.0, scale=10.0, size=1000)
    act_same = np.random.normal(loc=100.0, scale=10.0, size=1000)
    psi_low = calculate_psi(ref, act_same)
    assert psi_low < 0.10  # Stable / no drift

    # Severely shifted distribution
    act_shifted = np.random.normal(loc=150.0, scale=10.0, size=1000)
    psi_high = calculate_psi(ref, act_shifted)
    assert psi_high > 0.25  # Significant drift detected
