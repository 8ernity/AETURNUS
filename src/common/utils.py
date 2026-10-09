import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Tuple
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def get_root_dir() -> Path:
    return ROOT_DIR

def load_yaml(rel_path: str) -> Dict[str, Any]:
    file_path = ROOT_DIR / rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def calculate_nasa_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Computes NASA asymmetric evaluation score:
    d = y_pred - y_true
    score = sum(exp(-d/13) - 1 for d < 0) + sum(exp(d/10) - 1 for d >= 0)
    Late predictions (d > 0) are penalized exponentially higher than early predictions (d < 0).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    d = y_pred_arr - y_true_arr
    
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_pred - y_true
    
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    nasa_score = calculate_nasa_score(y_true, y_pred)
    late_rate = float(np.mean(residuals > 0))
    early_rate = float(np.mean(residuals < 0))
    
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "nasa_score": round(nasa_score, 2),
        "late_rate": round(late_rate, 4),
        "early_rate": round(early_rate, 4)
    }

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current distribution.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
        
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_edges = np.percentile(expected, percentiles)
    bucket_edges[0] -= 1e-5
    bucket_edges[-1] += 1e-5
    
    # Avoid duplicate bucket edges
    bucket_edges = np.unique(bucket_edges)
    if len(bucket_edges) < 2:
        return 0.0
        
    expected_counts, _ = np.histogram(expected, bins=bucket_edges)
    actual_counts, _ = np.histogram(actual, bins=bucket_edges)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Laplace smoothing to prevent division by zero or log(0)
    eps = 1e-4
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)
    
    # Re-normalize
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))

def calculate_ks_drift(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float, bool]:
    """
    Performs Kolmogorov-Smirnov test to detect distribution drift.
    Returns: (ks_stat, p_value, is_drift)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0, 1.0, False
    res = stats.ks_2samp(ref_clean, curr_clean)
    is_drift = bool(res.pvalue < 0.05)
    return float(res.statistic), float(res.pvalue), is_drift
