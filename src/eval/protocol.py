# src/eval/protocol.py
"""
Single canonical evaluation protocol for AETURNUS.
Defines truncation sampling (seeded), RUL target cap, NASA PHM'08 score,
and standard metric computation.
"""
import json
import hashlib
from typing import Dict, Any, Union, Optional, List
import numpy as np
import pandas as pd

DEFAULT_RUL_CAP = 125.0
DEFAULT_SEED = 42
DEFAULT_CV_FOLDS = 5

def get_protocol_id(
    cap: float = DEFAULT_RUL_CAP,
    mode: str = "last",
    seed: int = DEFAULT_SEED,
    n_draws: int = 1,
    metric: str = "nasa_phm08",
    rul_cap: Optional[float] = None,
    metric_name: Optional[str] = None
) -> str:
    """
    Returns unique, deterministic protocol ID: sha256 of {cap, mode, seed, n_draws, metric}.
    Supports backward compatibility for rul_cap / metric_name keyword arguments.
    """
    c = float(rul_cap if rul_cap is not None else cap)
    m = str(metric_name if metric_name is not None else metric)
    payload = {
        "cap": c,
        "mode": str(mode),
        "seed": int(seed),
        "n_draws": int(n_draws),
        "metric": m
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def cap_rul(rul: Union[np.ndarray, pd.Series, float, list], max_rul: float = DEFAULT_RUL_CAP) -> np.ndarray:
    """Caps piecewise linear RUL target to suppress steady-state noise."""
    arr = np.asarray(rul, dtype=float)
    return np.clip(arr, 0.0, float(max_rul))

def calculate_nasa_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Official NASA PHM'08 Data Challenge Scoring Function.
    Asymmetric exponential penalty penalizing late predictions (overestimating RUL)
    more severely than early predictions.
    d = y_pred - y_true
    s_i = exp(-d / 13) - 1 for d < 0
    s_i = exp(d / 10) - 1 for d >= 0
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yp - yt

    early_mask = diff < 0
    late_mask = diff >= 0

    scores = np.zeros_like(diff, dtype=float)
    scores[early_mask] = np.exp(-diff[early_mask] / 13.0) - 1.0
    scores[late_mask] = np.exp(diff[late_mask] / 10.0) - 1.0

    return float(np.sum(scores))

def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    cap: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes standard regression & domain metrics:
    RMSE, MAE, NASA PHM'08 score, early rate, late rate.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if cap is not None:
        yt = cap_rul(yt, max_rul=cap)
        yp = cap_rul(yp, max_rul=cap)

    diff = yp - yt
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    nasa = calculate_nasa_score(yt, yp)

    late_count = int(np.sum(diff > 0))
    early_count = int(np.sum(diff < 0))
    n = max(1, len(diff))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "nasa_score": float(nasa),
        "late_rate": float(late_count / n),
        "early_rate": float(early_count / n)
    }

def sample_truncation(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    mode: str = "last",
    min_cycles: int = 15,
    max_fraction: float = 1.0
) -> pd.DataFrame:
    """
    Truncation sampling protocol for engine trajectories.
    - 'last': Returns the final available cycle per engine unit (NASA test protocol).
    - 'random': Deterministically samples an engine cycle cutoff per unit using seed.
    """
    if "unit" not in df.columns or "cycle" not in df.columns:
        raise ValueError("DataFrame must contain 'unit' and 'cycle' columns for truncation sampling.")

    if mode == "last":
        last_indices = df.groupby("unit")["cycle"].idxmax()
        return df.loc[last_indices].sort_values("unit").copy()

    elif mode == "random":
        rng = np.random.RandomState(seed)
        selected_indices = []
        for unit_id, group in df.groupby("unit"):
            cycles = group["cycle"].values
            max_c = int(cycles.max())
            min_c = min(int(cycles.min()) + min_cycles, max_c)
            if min_c >= max_c:
                cutoff = max_c
            else:
                cutoff = int(rng.randint(min_c, max_c + 1))
            
            # Index of row corresponding to cutoff cycle
            cutoff_row = group[group["cycle"] == cutoff]
            if not cutoff_row.empty:
                selected_indices.append(cutoff_row.index[0])
            else:
                selected_indices.append(group["cycle"].idxmax())
                
        return df.loc[selected_indices].sort_values("unit").copy()

    else:
        raise ValueError(f"Unknown truncation mode '{mode}'. Choose 'last' or 'random'.")

def eval_frames(
    df: pd.DataFrame,
    mode: str = "last",
    seed: int = DEFAULT_SEED,
    n_draws: int = 1,
    min_cycles: int = 15
) -> List[pd.DataFrame]:
    """
    Protocol evaluation frame generator used by BOTH evaluate_cv and evaluate_test.
    Returns a list of sampled DataFrame frames based on truncation mode and draws.
    """
    frames: List[pd.DataFrame] = []
    if mode == "last":
        frames.append(sample_truncation(df, seed=seed, mode="last"))
    elif mode == "random":
        for i in range(max(1, int(n_draws))):
            frames.append(
                sample_truncation(
                    df,
                    seed=seed + i,
                    mode="random",
                    min_cycles=min_cycles,
                )
            )
    else:
        frames.append(sample_truncation(df, seed=seed, mode=mode, min_cycles=min_cycles))
    return frames
