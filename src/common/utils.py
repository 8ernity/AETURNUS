import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Tuple
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def get_root_dir() -> Path:
    return ROOT_DIR

def unwrap_config(node: Any) -> Any:
    """Recursively unpacks {value: X, kind: ..., source: ...} config nodes to X."""
    if isinstance(node, dict):
        if "value" in node and ("kind" in node or "source" in node):
            return unwrap_config(node["value"])
        return {k: unwrap_config(v) for k, v in node.items()}
    elif isinstance(node, list):
        return [unwrap_config(x) for x in node]
    return node

def load_yaml(rel_path: str, raw: bool = False) -> Dict[str, Any]:
    file_path = ROOT_DIR / rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if raw or data is None:
        return data
    return unwrap_config(data)

from src.eval.protocol import calculate_nasa_score, calculate_metrics

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
