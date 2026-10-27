import os
import re
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

from src.common.utils import get_root_dir, load_yaml
from src.ingest.loader import CMAPSSLoader
from src.eval.protocol import (
    calculate_metrics,
    calculate_nasa_score,
    sample_truncation,
    eval_frames,
    cap_rul,
    get_protocol_id,
    DEFAULT_SEED,
    DEFAULT_CV_FOLDS,
)
from src.eval.regime import summarize_by_regime

SENSOR_REGEX = re.compile(r"^s\d+$")

# Canonical mapping from the served p50 regressor class to the algorithm key stored
# in manifests and registry rows. Provenance is derived from the bundle, never typed.
P50_MODEL_CLASS_TO_ALGORITHM = {
    "LGBMRegressor": "lightgbm",
    "HistGradientBoostingRegressor": "hist_gb",
    "Ridge": "ridge",
}

ALGORITHM_TO_P50_MODEL_CLASS = {v: k for k, v in P50_MODEL_CLASS_TO_ALGORITHM.items()}


def algorithm_for_p50_model(p50_model: Any) -> str:
    """Returns the canonical algorithm key for a fitted p50 regressor instance."""
    class_name = type(p50_model).__name__
    if class_name not in P50_MODEL_CLASS_TO_ALGORITHM:
        raise ValueError(
            f"Unsupported p50 regressor class '{class_name}'; cannot derive algorithm provenance. "
            f"Known classes: {sorted(P50_MODEL_CLASS_TO_ALGORITHM)}"
        )
    return P50_MODEL_CLASS_TO_ALGORITHM[class_name]

def load_cv_protocol() -> Dict[str, Any]:
    """Loads the canonical CV censoring protocol from configs/pipeline.yaml.

    CV must censor targets like the official test protocol: random truncation with
    at least five draws. Refusing other configurations prevents a silent switch back
    to last-cycle validation, whose RUL distribution is degenerate on train engines.
    """
    cfg = load_yaml("configs/pipeline.yaml")
    cv_cfg = cfg["evaluation"]["cv"]
    mode = str(cv_cfg["mode"])
    n_draws = int(cv_cfg["n_draws"])
    min_cycles = int(cv_cfg["min_cycles"])
    if mode != "random":
        raise ValueError(
            f"CV mode must be 'random' to censor validation targets like the test set; got '{mode}'."
        )
    if n_draws < 5:
        raise ValueError(
            f"CV n_draws must be >= 5 to stabilize the censored RUL distribution; got {n_draws}."
        )
    if min_cycles <= 0:
        raise ValueError(f"CV min_cycles must be positive; got {min_cycles}.")
    return {
        "mode": mode,
        "n_draws": n_draws,
        "min_cycles": min_cycles,
        "ks_threshold": float(cv_cfg["ks_threshold"]),
        "mean_rel_gap": float(cv_cfg["mean_rel_gap"]),
    }

def wilson_ci(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = 1.95996
    denominator = 1.0 + (z ** 2) / n
    centre = (p + (z ** 2) / (2 * n)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) + (z ** 2) / (4 * n)) / n) / denominator
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))

def get_rul_cap() -> float:
    """RUL target/reporting cap shared by evaluation and rendering."""
    cfg = load_yaml("configs/pipeline.yaml")
    return float(cfg["features"]["rul_cap"])


def get_rul_bucket_definitions(cap: Optional[float] = None) -> List[Dict[str, Any]]:
    """Builds RUL reporting buckets from configs/targets.yaml.

    The final bucket is open-ended (all truth above the last configured edge), so
    cap-region reporting remains explicit rather than silently dropping engines.
    """
    cap_value = float(cap if cap is not None else get_rul_cap())
    cfg = load_yaml("configs/targets.yaml", raw=True)
    edges = [float(e) for e in cfg["rul_buckets"]["edges"]["value"]]
    edges = sorted(edge for edge in edges if 0.0 < edge < cap_value)
    bounds = [0.0, *edges]
    buckets = []
    for index, lower in enumerate(bounds):
        is_last = index == len(bounds) - 1
        upper = None if is_last else float(bounds[index + 1])
        if is_last:
            label = f"> {lower:g} cycles"
        elif index == 0:
            label = f"0 - {upper:g} cycles"
        else:
            label = f"{lower:g} - {upper:g} cycles"
        buckets.append({"bucket": label, "lower": lower, "upper": upper, "is_first": index == 0})
    return buckets


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) == 0:
        return None
    return float(np.sqrt(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) == 0:
        return None
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))))


def compute_cap_aware_metrics(
    y_true_raw: np.ndarray,
    pred_p50: np.ndarray,
    cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes raw/capped-truth RMSE and the same breakdown by RUL bucket."""
    raw = np.asarray(y_true_raw, dtype=float)
    pred = np.asarray(pred_p50, dtype=float)
    cap_value = float(cap if cap is not None else get_rul_cap())
    capped = np.minimum(raw, cap_value)
    le_cap_mask = raw <= cap_value
    bucket_defs = get_rul_bucket_definitions(cap_value)

    def bucket_mask(bucket: Dict[str, Any]) -> np.ndarray:
        if bucket["upper"] is None:
            return raw > float(bucket["lower"])
        if bucket["is_first"]:
            return raw <= float(bucket["upper"])
        return (raw > float(bucket["lower"])) & (raw <= float(bucket["upper"]))

    bucket_metrics = []
    for bucket in bucket_defs:
        mask = bucket_mask(bucket)
        bucket_raw = raw[mask]
        bucket_pred = pred[mask]
        bucket_capped = capped[mask]
        bucket_le_cap = mask & le_cap_mask
        bucket_metrics.append({
            "bucket": bucket["bucket"],
            "lower": bucket["lower"],
            "upper": bucket["upper"],
            "sample_count": int(np.sum(mask)),
            "rmse_vs_raw_truth": _rmse(bucket_raw, bucket_pred),
            "rmse_vs_capped_truth": _rmse(bucket_capped, bucket_pred),
            "share_true_rul_above_cap": float(np.mean(bucket_raw > cap_value)) if bucket_raw.size else None,
            "rmse_on_true_rul_le_cap": _rmse(raw[bucket_le_cap], pred[bucket_le_cap]),
            "n_true_rul_le_cap": int(np.sum(bucket_le_cap)),
            "n_true_rul_above_cap": int(np.sum(mask & ~le_cap_mask)),
            "mae_vs_raw_truth": _mae(bucket_raw, bucket_pred),
            "mae_vs_capped_truth": _mae(bucket_capped, bucket_pred),
        })

    return {
        "rul_cap": cap_value,
        "rmse_vs_raw_truth": _rmse(raw, pred),
        "rmse_vs_capped_truth": _rmse(capped, pred),
        "share_true_rul_above_cap": float(np.mean(raw > cap_value)) if raw.size else 0.0,
        "rmse_on_true_rul_le_cap": _rmse(raw[le_cap_mask], pred[le_cap_mask]),
        "n_true_rul_le_cap": int(np.sum(le_cap_mask)),
        "n_true_rul_above_cap": int(np.sum(~le_cap_mask)),
        "rul_bucket_metrics": bucket_metrics,
    }


def censored_coverage_mask(
    y_true_raw: np.ndarray,
    pred_p10: np.ndarray,
    pred_p90: np.ndarray,
    censored_at_cap: np.ndarray,
) -> np.ndarray:
    """Coverage under the open-ended cap rule.

    A censored prediction has no finite upper bound, so it covers every true RUL
    at or above its lower bound. Non-censored predictions remain closed intervals.
    """
    y = np.asarray(y_true_raw, dtype=float)
    p10 = np.asarray(pred_p10, dtype=float)
    p90 = np.array(
        [np.nan if value is None else float(value) for value in np.asarray(pred_p90, dtype=object)],
        dtype=float,
    )
    censored = np.asarray(censored_at_cap, dtype=bool)
    covered = np.zeros(len(y), dtype=bool)

    if censored.any():
        covered[censored] = y[censored] >= p10[censored]
    closed = ~censored
    if closed.any():
        covered[closed] = (y[closed] >= p10[closed]) & (y[closed] <= p90[closed])
    return covered


def compute_coverage_split(
    y_true_raw: np.ndarray,
    pred_p10: np.ndarray,
    pred_p90: np.ndarray,
    censored_at_cap: np.ndarray,
    cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes all-engine, at-or-below-cap, and above-cap coverage with Wilson CIs."""
    y = np.asarray(y_true_raw, dtype=float)
    p10 = np.asarray(pred_p10, dtype=float)
    p90 = np.array(
        [np.nan if value is None else float(value) for value in np.asarray(pred_p90, dtype=object)],
        dtype=float,
    )
    censored = np.asarray(censored_at_cap, dtype=bool)
    cap_value = float(cap if cap is not None else get_rul_cap())
    covered = censored_coverage_mask(y, p10, p90, censored)

    def stats(mask: np.ndarray) -> Dict[str, Any]:
        n = int(np.sum(mask))
        if n == 0:
            return {
                "coverage": None,
                "n": 0,
                "wilson_ci_95": [None, None],
                "wilson_ci": [None, None],
            }
        value = float(np.mean(covered[mask]))
        lower, upper = wilson_ci(value, n, confidence=0.95)
        return {
            "coverage": value,
            "n": n,
            "wilson_ci_95": [round(lower, 4), round(upper, 4)],
            "wilson_ci": [round(lower, 4), round(upper, 4)],
        }

    masks = {
        "all": np.ones(len(y), dtype=bool),
        "le_cap": y <= cap_value,
        "above_cap": y > cap_value,
    }
    split = {name: stats(mask) for name, mask in masks.items()}

    closed = ~censored
    closed_widths = p90[closed] - p10[closed]
    mean_closed_width = float(np.mean(closed_widths)) if closed_widths.size else None
    above_cap_share = float(np.mean(y > cap_value)) if len(y) else 0.0

    result: Dict[str, Any] = {
        "rul_cap": cap_value,
        "coverage_rule": "censored_open_upper_bound",
        "coverage_all": split["all"]["coverage"],
        "coverage_all_n": split["all"]["n"],
        "coverage_all_wilson_ci_95": split["all"]["wilson_ci_95"],
        "coverage_le_cap": split["le_cap"]["coverage"],
        "coverage_le_cap_n": split["le_cap"]["n"],
        "coverage_le_cap_wilson_ci_95": split["le_cap"]["wilson_ci_95"],
        "coverage_above_cap": split["above_cap"]["coverage"],
        "coverage_above_cap_n": split["above_cap"]["n"],
        "coverage_above_cap_wilson_ci_95": split["above_cap"]["wilson_ci_95"],
        "coverage_split": split,
        "share_true_rul_above_cap": above_cap_share,
        "n_censored_intervals": int(np.sum(censored)),
        "n_closed_intervals": int(np.sum(closed)),
        "censored_interval_rate": float(np.mean(censored)) if len(censored) else 0.0,
        "mean_closed_interval_width": mean_closed_width,
    }
    return result


def resolve_target_policy(subset: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves target_met from the named target metric in configs/targets.yaml."""
    cfg = load_yaml("configs/targets.yaml", raw=True)
    target_metric_cfg = cfg["target_metric"]
    metric_name = str(target_metric_cfg["value"])
    allowed = [str(m) for m in target_metric_cfg.get("allowed_metrics", [])]
    if allowed and metric_name not in allowed:
        raise ValueError(
            f"target_metric '{metric_name}' is not listed in target_metric.allowed_metrics: {allowed}"
        )
    if metric_name not in metrics:
        raise ValueError(
            f"target_metric '{metric_name}' is not produced by evaluation; "
            f"available metrics: {sorted(metrics)}"
        )
    metric_value = metrics[metric_name]
    if metric_value is None:
        raise ValueError(f"target_metric '{metric_name}' is unavailable for subset {subset}")
    metric_value = float(metric_value)
    target_rmse = float(cfg["subsets"][subset]["target_rmse"]["value"])
    return {
        "target_rmse": target_rmse,
        "target_metric": metric_name,
        "target_metric_value": metric_value,
        "target_metric_provenance": {
            "value": metric_name,
            "kind": target_metric_cfg.get("kind"),
            "source": target_metric_cfg.get("source"),
        },
        "target_provenance": cfg["subsets"][subset]["target_rmse"],
        "target_met": bool(metric_value <= target_rmse),
    }


def resolve_coverage_policy(subset: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves coverage attainment from the named policy metric in targets.yaml."""
    targets_cfg = load_yaml("configs/targets.yaml", raw=True)
    coverage_cfg = targets_cfg["coverage_metric"]
    metric_name = str(coverage_cfg["value"])
    allowed = [str(m) for m in coverage_cfg.get("allowed_metrics", [])]
    if allowed and metric_name not in allowed:
        raise ValueError(
            f"coverage_metric '{metric_name}' is not listed in "
            f"coverage_metric.allowed_metrics: {allowed}"
        )
    if metric_name not in metrics:
        raise ValueError(
            f"coverage_metric '{metric_name}' is not produced by evaluation; "
            f"available metrics: {sorted(metrics)}"
        )
    metric_value = metrics[metric_name]
    if metric_value is None:
        raise ValueError(f"coverage_metric '{metric_name}' is unavailable for subset {subset}")
    metric_value = float(metric_value)

    pipeline_cfg = load_yaml("configs/pipeline.yaml", raw=True)
    target_cfg = pipeline_cfg["conformal"]["target_coverage"]
    target_coverage = float(target_cfg["value"])
    ci = metrics.get(f"{metric_name}_wilson_ci_95")
    ci_lower = float(ci[0]) if ci and ci[0] is not None else None
    ci_upper = float(ci[1]) if ci and ci[1] is not None else None

    return {
        "coverage_metric": metric_name,
        "coverage_metric_value": metric_value,
        "coverage_metric_wilson_ci_95": [ci_lower, ci_upper],
        "coverage_metric_ci_lower": ci_lower,
        "coverage_metric_ci_upper": ci_upper,
        "coverage_metric_provenance": {
            "value": metric_name,
            "kind": coverage_cfg.get("kind"),
            "source": coverage_cfg.get("source"),
            "config": "configs/targets.yaml",
        },
        "target_coverage": target_coverage,
        "target_coverage_provenance": {
            **target_cfg,
            "config": "configs/pipeline.yaml",
        },
        "target_coverage_met": bool(metric_value >= target_coverage),
        "coverage_ci_lower_meets_target": bool(
            ci_lower is not None and ci_lower >= target_coverage
        ),
        "coverage_ci_excludes_target": bool(
            ci_lower is not None
            and (ci_lower > target_coverage or (ci_upper is not None and ci_upper < target_coverage))
        ),
    }


def bootstrap_rmse_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Computes a 1000-resample bootstrap 95% CI for RMSE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    rmses = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rmse_b = np.sqrt(np.mean((y_pred[idx] - y_true[idx]) ** 2))
        rmses.append(rmse_b)
    return float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))

class ModelService:
    def __init__(self, subset: str = "FD001"):
        self.subset = subset
        self.root_dir = get_root_dir()
        self.models_cfg = load_yaml("configs/models.yaml")
        self.sensor_cfg = load_yaml("configs/sensor_map.yaml")
        self.artifacts_dir = self.root_dir / "artifacts" / "models"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Algorithm selection
        self.algorithm = self.models_cfg.get("algorithms", {}).get(
            self.subset, self.models_cfg.get("default_algorithm", "lightgbm")
        )
        
        self.feature_cols: List[str] = []
        self.anomaly_detector: Optional[IsolationForest] = None
        self.rul_p10_model: Optional[Any] = None
        self.rul_p50_model: Optional[Any] = None
        self.rul_p90_model: Optional[Any] = None
        self.classifier: Optional[Any] = None
        self.classifiers: Dict[int, Any] = {}
        self.horizon_thresholds: Dict[int, float] = {10: 0.50, 30: 0.70, 60: 0.50}
        self.classifier_threshold_30: float = 0.70
        # Calibration is derived from the bundle (or set by train/calibrate_intervals); never defaulted.
        self.interval_calibration_offset: Optional[float] = None
        self.conformal_q: Optional[float] = None
        # Per-predicted-RUL-bucket conformal offsets learned on calibration engines.
        # Kept as None for legacy bundles so the global offset remains a valid fallback.
        self.conformal_buckets: Optional[Dict[str, Any]] = None
        self.training_time_sec: float = 0.0
        self.version: str = "v1.0"
        self._shap_explainer: Optional[Any] = None

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        """Filters input dataframe down to active model features."""
        exclude_cols = ["unit", "cycle", "regime", "rul_raw", "rul_capped", "hi_raw"]
        candidate_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("fails_within_")]
        valid_cols = []
        for c in candidate_cols:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].std() > 1e-4:
                valid_cols.append(c)
        return valid_cols

    def get_shap_explainer(self) -> Any:
        """Lazily creates and caches the shap.TreeExplainer on the p50 model."""
        if self._shap_explainer is None and self.rul_p50_model is not None:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.rul_p50_model)
        return self._shap_explainer

    @staticmethod
    def _conformal_settings() -> Dict[str, Any]:
        """Reads conformal coverage policy and candidate levels from config."""
        cfg = load_yaml("configs/pipeline.yaml")
        conformal = cfg["conformal"]
        target_coverage = float(conformal["target_coverage"])
        nominal_levels = sorted(float(v) for v in conformal["nominal_levels"])
        validation_draws = int(conformal["validation_draws"])
        selection_tolerance = float(conformal["selection_tolerance"])
        cross_folds = int(conformal["cross_conformal_folds"])
        n_buckets = int(conformal["n_buckets"])
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(f"Conformal target_coverage must be in (0, 1); got {target_coverage}")
        if not nominal_levels or any(not 0.0 < level < 1.0 for level in nominal_levels):
            raise ValueError(f"Conformal nominal_levels must be non-empty and in (0, 1); got {nominal_levels}")
        if validation_draws < 10:
            raise ValueError(
                f"Conformal validation_draws must be at least 10 per engine; got {validation_draws}"
            )
        if not 0.0 <= selection_tolerance < target_coverage:
            raise ValueError(
                f"Conformal selection_tolerance must be in [0, {target_coverage}); "
                f"got {selection_tolerance}"
            )
        if cross_folds < 2:
            raise ValueError(f"cross_conformal_folds must be at least 2; got {cross_folds}")
        return {
            "target_coverage": target_coverage,
            "nominal_levels": nominal_levels,
            "validation_draws": validation_draws,
            "selection_tolerance": selection_tolerance,
            "cross_conformal_folds": cross_folds,
            "n_buckets": n_buckets,
        }

    def _bucket_indices(self, predicted_p50: np.ndarray) -> np.ndarray:
        """Maps predicted p50 values to the calibrated predicted-RUL buckets."""
        pred = np.asarray(predicted_p50, dtype=float)
        metadata = getattr(self, "conformal_buckets", None)
        if not metadata:
            return np.zeros(len(pred), dtype=int)
        edges = np.asarray(metadata.get("edges", []), dtype=float)
        if edges.size == 0:
            return np.zeros(len(pred), dtype=int)
        return np.searchsorted(edges, pred, side="right").astype(int)

    def _interval_offsets(self, predicted_p50: np.ndarray) -> np.ndarray:
        """Returns per-prediction conformal offsets, with global q as fallback."""
        pred = np.asarray(predicted_p50, dtype=float)
        metadata = getattr(self, "conformal_buckets", None)
        if not metadata or not metadata.get("bucket_q"):
            return np.full(len(pred), self._require_calibration_offset(), dtype=float)
        bucket_q = np.asarray(metadata["bucket_q"], dtype=float)
        indices = self._bucket_indices(pred)
        fallback_value = metadata.get("global_q")
        if fallback_value is None:
            fallback_value = self._require_calibration_offset()
        fallback = float(fallback_value)
        clipped = np.clip(indices, 0, len(bucket_q) - 1)
        return np.where((indices >= 0) & (indices < len(bucket_q)), bucket_q[clipped], fallback)

    def _offset_for_p50(self, predicted_p50: float) -> float:
        return float(self._interval_offsets(np.asarray([predicted_p50], dtype=float))[0])

    def _conformal_components(
        self,
        frame: pd.DataFrame,
        models: Optional[Tuple[Any, Any, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns p50, uncalibrated interval bounds, conformity scores, and truth."""
        p10_model, p50_model, p90_model = models or (
            self.rul_p10_model,
            self.rul_p50_model,
            self.rul_p90_model,
        )
        X = frame[self.feature_cols].to_numpy(dtype=np.float32)
        y = frame["rul_capped"].to_numpy(dtype=float)
        raw_p10 = p10_model.predict(X)
        pred_p50 = np.maximum(0.0, p50_model.predict(X))
        raw_p90 = p90_model.predict(X)
        base_p10 = np.maximum(0.0, np.minimum(raw_p10, pred_p50))
        base_p90 = np.maximum(pred_p50, np.maximum(raw_p90, pred_p50))
        scores = np.maximum(base_p10 - y, y - base_p90)
        return pred_p50, base_p10, base_p90, scores, y

    @staticmethod
    def _conformal_quantile(values: np.ndarray, nominal_level: float) -> float:
        n = len(values)
        if n == 0:
            return 0.0
        q_level = min(1.0, np.ceil((n + 1) * nominal_level) / n)
        return float(np.quantile(values, q_level))

    def _cross_conformal_scores(
        self,
        all_non_test_df: pd.DataFrame,
        folds: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Generates out-of-fold quantile residuals across all non-test engines.

        Each engine belongs to exactly one validation fold, so its conformity
        score comes from a model that never saw that engine. This is a true
        cross-conformal calibration rather than reusing in-sample residuals.
        """
        units = sorted(int(u) for u in np.unique(all_non_test_df["unit"].values))
        n_folds = min(int(folds), len(units))
        if n_folds < 2:
            raise ValueError(
                f"[{self.subset}] cross-conformal requires at least two non-test engines; "
                f"got {len(units)}"
            )

        protocol = load_cv_protocol()
        splitter = GroupKFold(n_splits=n_folds)
        oof_pred: List[np.ndarray] = []
        oof_scores: List[np.ndarray] = []
        completed_folds = 0

        for train_idx, validation_idx in splitter.split(
            all_non_test_df, groups=all_non_test_df["unit"]
        ):
            train_fold = all_non_test_df.iloc[train_idx]
            validation_fold = all_non_test_df.iloc[validation_idx]
            if train_fold.empty or validation_fold.empty:
                continue

            X_fold_train = train_fold[self.feature_cols].to_numpy(dtype=np.float32)
            y_fold_train = train_fold["rul_capped"].to_numpy(dtype=float)
            fold_models = (
                clone(self.rul_p10_model),
                clone(self.rul_p50_model),
                clone(self.rul_p90_model),
            )
            for model in fold_models:
                model.fit(X_fold_train, y_fold_train)

            validation_frame = pd.concat(
                eval_frames(
                    validation_fold,
                    mode=protocol["mode"],
                    seed=DEFAULT_SEED,
                    n_draws=protocol["n_draws"],
                    min_cycles=protocol["min_cycles"],
                ),
                ignore_index=True,
            )
            pred_p50, _, _, scores, _ = self._conformal_components(
                validation_frame, models=fold_models
            )
            oof_pred.append(pred_p50)
            oof_scores.append(scores)
            completed_folds += 1

        if not oof_scores:
            raise ValueError(f"[{self.subset}] cross-conformal produced no validation folds")
        return np.concatenate(oof_pred), np.concatenate(oof_scores), completed_folds

    def _fit_conformal_buckets(
        self,
        calibration_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        validation_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Selects the nominal level on validation engines and calibrates cross-conformally.

        The candidate nominal levels come from configs/pipeline.yaml. Selection
        requires the Wilson lower bound of censored validation coverage to reach
        ``target_coverage - selection_tolerance``; the final quantile is then
        estimated from out-of-fold residuals across all non-test engines.
        """
        if calibration_df.empty:
            raise ValueError(
                f"[{self.subset}] Calibration requires non-empty calibration engines; "
                "the persistent split must provide calib_units disjoint from val_units."
            )
        if validation_df is None or validation_df.empty:
            raise ValueError(
                f"[{self.subset}] Conformal nominal-level selection requires a held-out "
                "validation-engine fold."
            )

        settings = self._conformal_settings()
        target_coverage = settings["target_coverage"]
        nominal_levels = settings["nominal_levels"]
        validation_draws = settings["validation_draws"]
        selection_tolerance = settings["selection_tolerance"]
        selection_floor = target_coverage - selection_tolerance
        n_buckets = max(1, settings["n_buckets"])
        cap = get_rul_cap()
        protocol = load_cv_protocol()

        calibration_frame = pd.concat(
            eval_frames(
                calibration_df,
                mode=protocol["mode"],
                seed=DEFAULT_SEED,
                n_draws=protocol["n_draws"],
                min_cycles=protocol["min_cycles"],
            ),
            ignore_index=True,
        )
        validation_frame = pd.concat(
            eval_frames(
                validation_df,
                mode=protocol["mode"],
                seed=DEFAULT_SEED,
                n_draws=validation_draws,
                min_cycles=protocol["min_cycles"],
            ),
            ignore_index=True,
        )

        cal_pred_p50, _, _, cal_scores, _ = self._conformal_components(calibration_frame)
        val_pred_p50, val_base_p10, val_base_p90, _, val_truth = self._conformal_components(
            validation_frame
        )

        validation_levels = []
        selected_level: Optional[float] = None
        selected_coverage: Optional[float] = None
        selected_ci: Optional[Tuple[float, float]] = None
        selected_n = 0
        for level in nominal_levels:
            q = self._conformal_quantile(cal_scores, level)
            lower = np.maximum(0.0, val_base_p10 - q)
            upper = val_base_p90 + q
            censored = val_pred_p50 >= (cap - q)
            covered = censored_coverage_mask(val_truth, lower, upper, censored)
            coverage = float(np.mean(covered))
            n_validation = int(len(val_truth))
            ci_lower, ci_upper = wilson_ci(coverage, n_validation, confidence=0.95)
            passes_floor = bool(ci_lower >= selection_floor)
            validation_levels.append({
                "nominal_level": round(float(level), 4),
                "coverage": round(coverage, 4),
                "n": n_validation,
                "wilson_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
                "ci_lower": round(ci_lower, 4),
                "passes_floor": passes_floor,
                "censored_count": int(np.sum(censored)),
            })
            if selected_level is None and passes_floor:
                selected_level = float(level)
                selected_coverage = coverage
                selected_ci = (ci_lower, ci_upper)
                selected_n = n_validation

        if selected_level is None:
            detail = "; ".join(
                f"{item['nominal_level']:.2f}: cov={item['coverage']:.4f}, "
                f"CI_low={item['ci_lower']:.4f}"
                for item in validation_levels
            )
            raise ValueError(
                f"[{self.subset}] No nominal level reaches the validation coverage floor "
                f"{selection_floor:.4f} (target {target_coverage:.4f} - tolerance "
                f"{selection_tolerance:.4f}). {detail}"
            )

        cross_method = "cross_conformal_all_non_test_engines"
        cross_folds = 0
        calibration_source = calibration_df
        if train_df is not None and not train_df.empty:
            try:
                oof_pred_p50, scores, cross_folds = self._cross_conformal_scores(
                    train_df, settings["cross_conformal_folds"]
                )
                calibration_source = train_df
            except Exception as exc:
                print(
                    f"[{self.subset}] Cross-conformal fallback to calibration engines: {exc}"
                )
                oof_pred_p50 = cal_pred_p50
                scores = cal_scores
                cross_method = "split_conformal_cv_protocol_fallback"
        else:
            oof_pred_p50 = cal_pred_p50
            scores = cal_scores
            cross_method = "split_conformal_cv_protocol_fallback"

        n_scores = len(scores)
        effective_buckets = n_buckets if n_scores >= 2 * n_buckets else 1
        if effective_buckets > 1:
            edges = np.quantile(oof_pred_p50, np.linspace(0.0, 1.0, effective_buckets + 1)[1:-1])
            edges = np.unique(np.round(edges, 4))
            effective_buckets = len(edges) + 1
        else:
            edges = np.asarray([], dtype=float)

        bucket_indices = (
            np.searchsorted(edges, oof_pred_p50, side="right")
            if edges.size
            else np.zeros(n_scores, dtype=int)
        )

        # Cross-fitting can be mildly optimistic because its fold models see more
        # non-test engines than the served model. Keep the held-out calibration
        # quantile as a conservative floor so the selected nominal level still
        # delivers covered intervals rather than silently shrinking them.
        cross_global_q = self._conformal_quantile(scores, float(selected_level))
        calibration_global_q = self._conformal_quantile(cal_scores, float(selected_level))
        global_q = max(cross_global_q, calibration_global_q)

        calibration_bucket_indices = (
            np.searchsorted(edges, cal_pred_p50, side="right")
            if edges.size
            else np.zeros(len(cal_scores), dtype=int)
        )
        bucket_q = []
        bucket_counts = []
        bucket_score_counts = []
        for b in range(effective_buckets):
            mask = bucket_indices == b
            bucket_scores = scores[mask]
            calibration_mask = calibration_bucket_indices == b
            calibration_bucket_scores = cal_scores[calibration_mask]
            bucket_counts.append(int(np.sum(mask)))
            bucket_score_counts.append(int(len(bucket_scores)))
            cross_bucket_q = (
                self._conformal_quantile(bucket_scores, float(selected_level))
                if len(bucket_scores)
                else cross_global_q
            )
            calibration_bucket_q = (
                self._conformal_quantile(calibration_bucket_scores, float(selected_level))
                if len(calibration_bucket_scores)
                else calibration_global_q
            )
            bucket_q.append(max(cross_bucket_q, calibration_bucket_q))

        regime_count = (
            int(calibration_source["regime"].nunique())
            if "regime" in calibration_source.columns
            else 0
        )
        selected_ci = selected_ci or (0.0, 1.0)
        metadata = {
            "method": cross_method,
            "calibration_engine_count": int(calibration_source["unit"].nunique()),
            "n_calibration_engines": int(calibration_source["unit"].nunique()),
            "calibration_sample_count": int(n_scores),
            "target_coverage": target_coverage,
            "selection_tolerance": selection_tolerance,
            "selection_floor": round(float(selection_floor), 4),
            "nominal_level": round(float(selected_level), 4),
            "validation_coverage": round(float(selected_coverage), 4),
            "validation_coverage_wilson_ci_95": [
                round(float(selected_ci[0]), 4),
                round(float(selected_ci[1]), 4),
            ],
            "validation_coverage_ci_lower": round(float(selected_ci[0]), 4),
            "validation_coverage_ci_upper": round(float(selected_ci[1]), 4),
            "validation_sample_count": int(selected_n or len(val_truth)),
            "validation_engine_count": int(validation_df["unit"].nunique()),
            "validation_draws": validation_draws,
            "validation_truth_mode": "rul_capped",
            "validation_coverage_rule": "censored_open_upper_bound",
            "validation_levels": validation_levels,
            "selection_policy": (
                "smallest_nominal_level_with_validation_wilson_lower_bound_at_least_"
                "target_minus_tolerance"
            ),
            "cross_conformal_folds": int(cross_folds),
            "calibration_regime_count": regime_count,
            "n_calibration": int(n_scores),
            "protocol_id": get_protocol_id(
                mode=protocol["mode"], seed=DEFAULT_SEED, n_draws=protocol["n_draws"]
            ),
            "validation_protocol_id": get_protocol_id(
                mode=protocol["mode"], seed=DEFAULT_SEED, n_draws=validation_draws
            ),
            "n_buckets": effective_buckets,
            "edges": [float(e) for e in edges],
            "bucket_q": [float(q) for q in bucket_q],
            "bucket_counts": bucket_counts,
            "bucket_score_counts": bucket_score_counts,
            "global_q": global_q,
            "cross_conformal_global_q": cross_global_q,
            "calibration_global_q": calibration_global_q,
            "coverage_floor_policy": "max_cross_conformal_and_held_out_calibration_quantile",
        }
        self.conformal_buckets = metadata
        self.conformal_q = global_q
        self.interval_calibration_offset = global_q
        return metadata

    def recalibrate_conformal(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Recomputes conformal metadata for an already-loaded bundle."""
        from src.common.splits import get_persistent_split

        all_units = [int(u) for u in np.unique(train_df["unit"].values)]
        split = get_persistent_split(self.subset, all_units=all_units)
        calib_units = set(split.get("calib_units", []))
        val_units = set(split["val_units"])
        if not calib_units or not val_units:
            raise ValueError(
                f"[{self.subset}] Cannot recalibrate without distinct calibration and "
                "validation engine folds."
            )
        calibration_df = train_df[train_df["unit"].isin(calib_units)].copy()
        validation_df = train_df[train_df["unit"].isin(val_units)].copy()
        return self._fit_conformal_buckets(
            calibration_df,
            train_df=train_df,
            validation_df=validation_df,
        )

    def train(self, train_df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains the 3 complementary heads using group-based engine validation:
        1. Isolation Forest on healthy cycles
        2. Quantile RUL Regressors (p10, p50, p90) via LightGBM (or HistGB)
        3. Multi-horizon Calibrated Classifiers (10, 30, 60) with validation-tuned thresholds
        4. Validation-selected, cross-conformal interval calibration across all non-test engines
        """
        t0 = time.time()
        print(f"[{self.subset}] Selecting features for model training...")
        self.feature_cols = self._select_features(train_df)
        print(f"[{self.subset}] Selected {len(self.feature_cols)} active features.")

        # 0. Engine-level Group Split using persistent split (keep engines completely intact)
        from src.common.splits import get_persistent_split
        engine_units = [int(u) for u in np.unique(train_df["unit"].values)]
        split = get_persistent_split(self.subset, all_units=engine_units)

        persistent_train = set(split["train_units"])
        persistent_calib = set(split.get("calib_units", []))
        persistent_val = set(split["val_units"])

        # Strictly exclude lifecycle holdout engines from training, calibration, and validation.
        train_mask = train_df["unit"].isin(persistent_train)
        calib_mask = train_df["unit"].isin(persistent_calib)
        val_mask = train_df["unit"].isin(persistent_val)
        if not np.any(val_mask) or not np.any(calib_mask):
            # Synthetic/tiny-frame fallback: keep calibration and validation engines
            # disjoint while preserving a deterministic split for train guards.
            avail_units = sorted(int(u) for u in train_df["unit"].unique())
            group_size = max(2, int(len(avail_units) * 0.15)) if len(avail_units) > 3 else max(1, len(avail_units) - 1)
            group_size = min(group_size, max(1, len(avail_units) - 1))
            val_group = avail_units[-group_size:] if group_size else []
            n_calib = max(1, len(val_group) // 2) if len(val_group) > 1 else len(val_group)
            calib_units = set(val_group[:n_calib])
            val_units = set(val_group[n_calib:]) or calib_units
            train_units = set(avail_units) - calib_units - val_units
            train_mask = train_df["unit"].isin(train_units)
            calib_mask = train_df["unit"].isin(calib_units)
            val_mask = train_df["unit"].isin(val_units)

        X_train = train_df.loc[train_mask, self.feature_cols].to_numpy(dtype=np.float32)
        y_rul_train = train_df.loc[train_mask, "rul_capped"].values

        X_val = train_df.loc[val_mask, self.feature_cols].to_numpy(dtype=np.float32)
        y_rul_val = train_df.loc[val_mask, "rul_capped"].values

        # 1. Anomaly Detector on early healthy cycles
        print(f"[{self.subset}] Training Isolation Forest on early healthy cycles...")
        early_mask = train_df["cycle"] <= train_df.groupby("unit")["cycle"].transform("max") * 0.40
        X_healthy = train_df.loc[early_mask, self.feature_cols].to_numpy(dtype=np.float32)

        ad_cfg = self.models_cfg["anomaly_detector"]
        self.anomaly_detector = IsolationForest(
            contamination=ad_cfg["contamination"],
            n_estimators=ad_cfg["n_estimators"],
            random_state=random_state
        )
        self.anomaly_detector.fit(X_healthy)
        del X_healthy
        gc.collect()

        # 2. Check for tuned hyperparameters from Optuna
        best_params_path = self.root_dir / "configs" / f"best_params_{self.subset}.json"
        tuned_params = {}
        if best_params_path.exists():
            try:
                with open(best_params_path, "r", encoding="utf-8") as f:
                    tuned_params = json.load(f)
            except Exception:
                pass

        # 3. Quantile RUL Regressors
        reg_cfg = self.models_cfg.get("rul_regressor", {})
        n_est = tuned_params.get("n_estimators", reg_cfg.get("n_estimators", 150))
        lr = tuned_params.get("learning_rate", reg_cfg.get("learning_rate", 0.05))
        n_leaves = tuned_params.get("num_leaves", reg_cfg.get("num_leaves", 31))
        min_child = tuned_params.get("min_child_samples", reg_cfg.get("min_child_samples", 20))
        reg_alpha = tuned_params.get("reg_alpha", 0.0)
        reg_lambda = tuned_params.get("reg_lambda", 0.0)

        if self.algorithm == "lightgbm":
            print(f"[{self.subset}] Training LightGBM RUL Regressors: p50=huber, p10/p90=quantile...")
            # p10/p90: quantile objectives with boost_from_average=False for stable interval bounds
            self.rul_p10_model = lgb.LGBMRegressor(
                objective="quantile", alpha=0.10,
                n_estimators=n_est, learning_rate=lr,
                num_leaves=n_leaves, min_child_samples=min_child,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                boost_from_average=False,
                random_state=random_state, verbose=-1
            ).fit(X_train, y_rul_train)

            # p50: regression objective for unbiased point estimates
            # Quantile alpha=0.50 can produce biased medians on skewed subsets (e.g. FD003)
            self.rul_p50_model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=n_est, learning_rate=lr,
                num_leaves=n_leaves, min_child_samples=min_child,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                random_state=random_state, verbose=-1
            ).fit(X_train, y_rul_train)

            self.rul_p90_model = lgb.LGBMRegressor(
                objective="quantile", alpha=0.90,
                n_estimators=n_est, learning_rate=lr,
                num_leaves=n_leaves, min_child_samples=min_child,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                boost_from_average=False,
                random_state=random_state, verbose=-1
            ).fit(X_train, y_rul_train)
        else:
            print(f"[{self.subset}] Training HistGradientBoosting RUL Regressors: p50=squared_error, p10/p90=quantile...")
            self.rul_p10_model = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.10,
                max_iter=n_est, learning_rate=lr,
                random_state=random_state
            ).fit(X_train, y_rul_train)

            # p50: squared_error for unbiased point estimates (HistGB has no huber loss)
            self.rul_p50_model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=n_est, learning_rate=lr,
                random_state=random_state
            ).fit(X_train, y_rul_train)

            self.rul_p90_model = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.90,
                max_iter=n_est, learning_rate=lr,
                random_state=random_state
            ).fit(X_train, y_rul_train)

        # Invalidate cached explainer upon retraining
        self._shap_explainer = None

        # ── Train-time sanity assertions ──────────────────────────────────
        # Guard 1: model must have built sufficient trees
        if self.algorithm == "lightgbm":
            actual_trees = self.rul_p50_model.n_estimators_
            assert actual_trees > 20, (
                f"[{self.subset}] p50 model has only {actual_trees} trees "
                f"(expected >20). Training data may be too small."
            )

        # Guard 2: feature importances must be non-trivially positive
        if hasattr(self.rul_p50_model, "feature_importances_"):
            imp_sum = float(np.sum(np.maximum(0, self.rul_p50_model.feature_importances_)))
        else:
            # Subsample X_val to avoid excessive memory allocations
            n_sub = min(200, len(X_val))
            sub_idx = np.random.RandomState(42).choice(len(X_val), n_sub, replace=False)
            perm = permutation_importance(
                self.rul_p50_model, X_val[sub_idx], y_rul_val[sub_idx], n_repeats=2, random_state=42
            )
            imp_sum = float(np.sum(np.maximum(0, perm.importances_mean)))
        assert imp_sum > 0, (
            f"[{self.subset}] All feature importances are zero — "
            f"the model learned nothing useful from the training data."
        )

        # Guard 3: val last-window RMSE must beat a constant-mean baseline
        val_pred_p50 = np.maximum(0.0, self.rul_p50_model.predict(X_val))
        val_rmse = float(np.sqrt(np.mean((val_pred_p50 - y_rul_val) ** 2)))
        mean_baseline_rmse = float(np.sqrt(np.mean((np.mean(y_rul_train) - y_rul_val) ** 2)))
        assert val_rmse < mean_baseline_rmse, (
            f"[{self.subset}] Model val RMSE ({val_rmse:.2f}) is worse than "
            f"constant-mean baseline ({mean_baseline_rmse:.2f}). "
            f"The model failed to learn meaningful patterns."
        )
        self._train_val_rmse = val_rmse
        self._train_mean_baseline_rmse = mean_baseline_rmse

        # 4. Validation-selected nominal level, then cross-conformal calibration
        # across every non-test engine. Validation engines remain held out from
        # training and are only used to choose the nominal coverage level.
        calibration_df = train_df.loc[calib_mask].copy()
        validation_df = train_df.loc[val_mask].copy()
        conformal_metadata = self._fit_conformal_buckets(
            calibration_df,
            train_df=train_df,
            validation_df=validation_df,
        )
        del calibration_df, validation_df
        gc.collect()

        # 5. Multi-horizon Classifiers [10, 30, 60] with Validation-Tuned Thresholds
        horizons = self.models_cfg.get("classifier", {}).get("horizons", [10, 30, 60])
        target_recall = self.models_cfg.get("classifier", {}).get("target_recall", 0.90)
        cls_cfg = self.models_cfg.get("classifier", {})

        self.classifiers = {}
        self.horizon_thresholds = {}

        for h in horizons:
            gc.collect()
            target_col = f"fails_within_{h}"
            if target_col not in train_df.columns:
                rul_col = "rul_raw" if "rul_raw" in train_df.columns else "rul_capped"
                train_df[target_col] = (train_df[rul_col] <= h).astype(int)

            y_h_train = train_df.loc[train_mask, target_col].values
            y_h_val = train_df.loc[val_mask, target_col].values

            if self.algorithm == "lightgbm":
                base_cls = lgb.LGBMClassifier(
                    class_weight="balanced",
                    n_estimators=cls_cfg.get("n_estimators", 120),
                    learning_rate=cls_cfg.get("learning_rate", 0.05),
                    random_state=random_state,
                    verbose=-1
                )
            else:
                base_cls = HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=cls_cfg.get("max_iter", 120),
                    learning_rate=cls_cfg.get("learning_rate", 0.05),
                    random_state=random_state
                )

            cal_cls = CalibratedClassifierCV(estimator=base_cls, method="sigmoid", cv=3)
            cal_cls.fit(X_train, y_h_train)
            self.classifiers[h] = cal_cls

            # Choose threshold on validation engines
            probs_val = cal_cls.predict_proba(X_val)
            p_val = probs_val[:, 1] if probs_val.shape[1] > 1 else probs_val[:, 0]

            best_thresh = 0.50
            if len(np.unique(y_h_val)) > 1:
                threshold_candidates = np.linspace(0.05, 0.95, 91)
                valid_candidates = []
                for th in threshold_candidates:
                    preds_th = (p_val >= th).astype(int)
                    rec = recall_score(y_h_val, preds_th, zero_division=0)
                    prec = precision_score(y_h_val, preds_th, zero_division=0)
                    if rec >= target_recall:
                        valid_candidates.append((th, prec, rec))

                if valid_candidates:
                    # Smallest threshold with recall >= target_recall that maximizes precision
                    valid_candidates.sort(key=lambda x: (-x[1], x[0]))
                    best_thresh = float(valid_candidates[0][0])
                else:
                    # Pick threshold with highest recall
                    best_thresh = 0.30

            self.horizon_thresholds[h] = round(best_thresh, 3)

        # Primary classifier for backward compatibility (horizon 30)
        self.classifier = self.classifiers.get(30, list(self.classifiers.values())[0])
        self.classifier_threshold_30 = self.horizon_thresholds.get(30, 0.70)

        self.training_time_sec = round(time.time() - t0, 2)
        return {
            "status": "trained",
            "algorithm": self.algorithm,
            "features_count": len(self.feature_cols),
            "train_time_sec": self.training_time_sec,
            "train_engines": int(train_df.loc[train_mask, "unit"].nunique()),
            "calib_engines": int(train_df.loc[calib_mask, "unit"].nunique()),
            "val_engines": int(train_df.loc[val_mask, "unit"].nunique()),
            "calibration_offset": None if self.interval_calibration_offset is None else round(self.interval_calibration_offset, 3),
            "conformal_q": None if self.conformal_q is None else round(self.conformal_q, 3),
            "conformal_buckets": conformal_metadata,
            "horizon_thresholds": self.horizon_thresholds
        }

    def evaluate_cv(self, k: int = DEFAULT_CV_FOLDS, train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluates model over GroupKFold(k=5, groups=engine) on TRAIN engines only.
        Writes artifacts/reports/cv_{subset}.json with per-fold and mean+-std RMSE, NASA.
        """
        if train_df is None:
            train_path = self.root_dir / "data" / "processed" / self.subset / "train.parquet"
            train_df = pd.read_parquet(train_path)

        all_units = [int(u) for u in np.unique(train_df["unit"].values)]
        from src.common.splits import get_persistent_split
        split = get_persistent_split(self.subset, all_units=all_units)
        train_units = split["train_units"]

        mask = train_df["unit"].isin(train_units)
        if not self.feature_cols:
            self.feature_cols = self._select_features(train_df)

        cols_needed = ["unit", "cycle", "rul_capped"]
        if "regime" in train_df.columns:
            cols_needed.append("regime")
        cols_needed += [c for c in self.feature_cols if c not in cols_needed]
        train_only_df = train_df.loc[mask, cols_needed].reset_index(drop=True)

        X = train_only_df[self.feature_cols].to_numpy(dtype=np.float32)
        y = train_only_df["rul_capped"].to_numpy(dtype=np.float32)
        groups = train_only_df["unit"].to_numpy(dtype=np.int32)

        # Dynamically load hyperparameters from configs / best_params without literals
        models_cfg = load_yaml(self.root_dir / "configs" / "models.yaml")
        if self.algorithm == "lightgbm":
            best_params_path = self.root_dir / "configs" / f"best_params_{self.subset}.json"
            if best_params_path.exists():
                with open(best_params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
            else:
                lgb_cfg = models_cfg["baselines"]["lightgbm"]
                params = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in lgb_cfg.items()}
            params["objective"] = "regression"
            params["verbose"] = -1
            reg_builder = lambda: lgb.LGBMRegressor(**params)
        elif self.algorithm in ("hist_gb", "hist_gradient_boosting"):
            hgb_cfg = models_cfg["baselines"]["hist_gb"]
            params = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in hgb_cfg.items()}
            params["loss"] = "squared_error"
            reg_builder = lambda: HistGradientBoostingRegressor(**params)
        elif self.algorithm == "ridge":
            ridge_cfg = models_cfg["baselines"]["ridge"]
            params = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in ridge_cfg.items()}
            reg_builder = lambda: Ridge(**params)
        else:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}' for evaluate_cv; "
                "expected one of lightgbm, hist_gb, ridge."
            )

        cv_proto = load_cv_protocol()
        cv_mode = cv_proto["mode"]
        cv_draws = cv_proto["n_draws"]
        cv_min_cycles = cv_proto["min_cycles"]
        cv_protocol_id = get_protocol_id(mode=cv_mode, seed=DEFAULT_SEED, n_draws=cv_draws)

        def _censored_frame(fold_df: pd.DataFrame) -> pd.DataFrame:
            frames = eval_frames(
                fold_df,
                mode=cv_mode,
                seed=DEFAULT_SEED,
                n_draws=cv_draws,
                min_cycles=cv_min_cycles,
            )
            return pd.concat(frames, ignore_index=True)

        gkf = GroupKFold(n_splits=k)
        fold_results = []
        rmses, nasas, train_rmses = [], [], []
        pooled_val_y = []
        pooled_val_pred = []
        pooled_val_regime = []

        for f_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            reg = reg_builder()
            reg.fit(X[tr_idx], y[tr_idx])

            # Validation targets are censored with the same random truncation protocol
            # used to build the official test set, so CV metrics are test-like.
            val_frame = _censored_frame(train_only_df.iloc[val_idx])
            val_X = val_frame[self.feature_cols].values
            val_y = val_frame["rul_capped"].values

            tr_frame = _censored_frame(train_only_df.iloc[tr_idx])
            tr_X = tr_frame[self.feature_cols].values
            tr_y = tr_frame["rul_capped"].values

            pred_val = np.maximum(0.0, reg.predict(val_X))
            pred_tr = np.maximum(0.0, reg.predict(tr_X))
            m = calculate_metrics(val_y, pred_val)
            m_tr = calculate_metrics(tr_y, pred_tr)
            rmses.append(m["rmse"])
            nasas.append(m["nasa_score"])
            train_rmses.append(m_tr["rmse"])
            pooled_val_y.append(val_y)
            pooled_val_pred.append(pred_val)
            if "regime" in val_frame.columns:
                pooled_val_regime.append(val_frame["regime"].to_numpy(dtype=int))
            else:
                pooled_val_regime.append(np.full(len(val_y), -1, dtype=int))
            fold_results.append({
                "fold": f_idx,
                "train_rmse": round(float(m_tr["rmse"]), 2),
                "rmse": round(float(m["rmse"]), 2),
                "mae": round(float(m["mae"]), 2),
                "nasa_score": round(float(m["nasa_score"]), 1),
                "val_engines": int(len(np.unique(groups[val_idx])))
            })

        by_regime = summarize_by_regime(
            np.concatenate(pooled_val_y),
            np.concatenate(pooled_val_pred),
            np.concatenate(pooled_val_regime),
        )
        cv_report = {
            "subset": self.subset,
            "k_folds": k,
            "algorithm": self.algorithm,
            "protocol_id": cv_protocol_id,
            "evaluation_mode": cv_mode,
            "n_draws": cv_draws,
            "min_cycles": cv_min_cycles,
            "folds": fold_results,
            "mean_train_rmse": round(float(np.mean(train_rmses)), 2),
            "mean_rmse": round(float(np.mean(rmses)), 2),
            "std_rmse": round(float(np.std(rmses)), 2),
            "mean_nasa": round(float(np.mean(nasas)), 1),
            "std_nasa": round(float(np.std(nasas)), 1),
            "by_regime": by_regime,
            "regime_mix": {
                regime: {"sample_count": values["sample_count"], "share": values["share"]}
                for regime, values in by_regime.items()
            },
        }

        cv_file = self.root_dir / "artifacts" / "reports" / f"cv_{self.subset}.json"
        cv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cv_file, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)

        return cv_report

    def evaluate_test(self, test_df: pd.DataFrame, rul_truth: np.ndarray, save_report: bool = True) -> Dict[str, Any]:
        """
        Evaluates models on the official test set:
        - Real RMSE, MAE, NASA Score, 1000-resample Bootstrap 95% CI
        - Split-conformal coverage, mean width, Wilson 95% CI
        - Multi-horizon classification (PR-AUC, precision, recall at tuned threshold)
        """
        eval_rows = eval_frames(test_df, mode="last", seed=DEFAULT_SEED, n_draws=1)[0]

        for c in self.feature_cols:
            if c not in eval_rows.columns:
                raise ValueError(
                    f"[{self.subset}] evaluate_test: test dataframe is missing "
                    f"required feature column '{c}'. Ensure the feature pipeline "
                    f"produces all {len(self.feature_cols)} expected features."
                )

        X_test = eval_rows[self.feature_cols].values
        y_true = rul_truth

        pred_p50 = np.maximum(0.0, self.rul_p50_model.predict(X_test))
        raw_p10 = self.rul_p10_model.predict(X_test)
        raw_p90 = self.rul_p90_model.predict(X_test)

        # Enforce non-crossing quantile monotonicity: p10 <= p50 <= p90 with
        # per-predicted-RUL-bucket conformal offsets (global q for legacy bundles).
        cal_offsets = self._interval_offsets(pred_p50)
        base_p10 = np.maximum(0.0, np.minimum(raw_p10, pred_p50))
        base_p90 = np.maximum(pred_p50, np.maximum(raw_p90, pred_p50))
        pred_p10 = np.maximum(0.0, np.minimum(pred_p50, base_p10 - cal_offsets))
        pred_p90 = np.maximum(pred_p50, base_p90 + cal_offsets)

        # Regression metrics. Keep the legacy `rmse` key (raw truth) and expose
        # raw/capped-truth metrics side by side for policy and presentation.
        metrics = calculate_metrics(y_true, pred_p50)
        metrics.update(compute_cap_aware_metrics(y_true, pred_p50))

        # 1000-resample bootstrap 95% CI for last-window test RMSE
        boot_lower, boot_upper = bootstrap_rmse_ci(y_true, pred_p50, n_boot=1000, seed=DEFAULT_SEED)

        # Coverage under the censored open-upper-bound rule, split by true-RUL cap
        # region so cap effects are visible rather than diluted into one number.
        cap_value = float(metrics["rul_cap"])
        censored = pred_p50 >= (cap_value - cal_offsets)
        coverage_stats = compute_coverage_split(
            y_true,
            pred_p10,
            pred_p90,
            censored,
            cap=cap_value,
        )
        metrics.update(coverage_stats)
        covered = censored_coverage_mask(y_true, pred_p10, pred_p90, censored)
        interval_coverage = float(coverage_stats["coverage_all"])
        mean_interval_width = coverage_stats["mean_closed_interval_width"]
        wilson_lower, wilson_upper = coverage_stats["coverage_all_wilson_ci_95"]
        conformal_meta = getattr(self, "conformal_buckets", None) or {}
        target_value = conformal_meta.get("target_coverage")
        if target_value is None:
            target_value = self._conformal_settings()["target_coverage"]
        target_coverage = float(target_value)
        coverage_policy = resolve_coverage_policy(self.subset, metrics)
        policy_lower = coverage_policy["coverage_metric_ci_lower"]
        policy_upper = coverage_policy["coverage_metric_ci_upper"]
        ci_excludes_target = bool(
            policy_lower is not None
            and (policy_lower > target_coverage or (policy_upper is not None and policy_upper < target_coverage))
        )
        ci_contains_target = bool(
            policy_lower is not None
            and policy_upper is not None
            and policy_lower <= target_coverage <= policy_upper
        )
        all_ci_excludes_target = bool(
            wilson_lower is not None
            and (wilson_lower > target_coverage or wilson_upper < target_coverage)
        )
        all_ci_contains_target = bool(
            wilson_lower is not None
            and wilson_upper is not None
            and wilson_lower <= target_coverage <= wilson_upper
        )

        # Per-bucket held-out coverage diagnostics.
        bucket_indices = self._bucket_indices(pred_p50)
        edges = [float(e) for e in conformal_meta.get("edges", [])]
        bucket_q = [float(q) for q in conformal_meta.get("bucket_q", [])]
        coverage_by_bucket = []
        for bucket_idx in sorted(np.unique(bucket_indices).tolist()):
            bucket_mask = bucket_indices == bucket_idx
            n_bucket = int(np.sum(bucket_mask))
            bucket_cov = float(np.mean(covered[bucket_mask])) if n_bucket else 0.0
            b_lower, b_upper = wilson_ci(bucket_cov, n_bucket, confidence=0.95) if n_bucket else (0.0, 1.0)
            lower_edge = float(edges[bucket_idx - 1]) if bucket_idx > 0 and bucket_idx - 1 < len(edges) else None
            upper_edge = float(edges[bucket_idx]) if bucket_idx < len(edges) else None
            closed_bucket = bucket_mask & ~censored
            bucket_widths = (pred_p90 - pred_p10)[closed_bucket]
            coverage_by_bucket.append({
                "bucket_index": int(bucket_idx),
                "predicted_p50_lower": lower_edge,
                "predicted_p50_upper": upper_edge,
                "sample_count": n_bucket,
                "coverage": round(bucket_cov, 4),
                "wilson_ci_95": [round(b_lower, 4), round(b_upper, 4)],
                "n_censored_intervals": int(np.sum(bucket_mask & censored)),
                "mean_interval_width": round(float(np.mean(bucket_widths)), 2) if bucket_widths.size else None,
                "conformal_q": round(bucket_q[bucket_idx], 4) if bucket_idx < len(bucket_q) else round(float(self._require_calibration_offset()), 4),
            })
        test_by_regime = summarize_by_regime(
            y_true,
            pred_p50,
            eval_rows["regime"].to_numpy(dtype=int) if "regime" in eval_rows.columns else np.full(len(y_true), -1),
        )
        global_cal_offset = self._require_calibration_offset()

        # Multi-horizon classification metrics
        multi_horizon_metrics = {}
        horizons = [10, 30, 60]
        for h in horizons:
            y_true_h = (y_true <= h).astype(int)
            cal_cls = self.classifiers.get(h, self.classifier)
            probs_mat = cal_cls.predict_proba(X_test)
            probs = probs_mat[:, 1] if probs_mat.shape[1] > 1 else probs_mat[:, 0]
            
            if len(np.unique(y_true_h)) > 1:
                pr_auc_h = float(average_precision_score(y_true_h, probs))
            else:
                pr_auc_h = float(np.mean(y_true_h == (probs >= 0.5)))

            th = self.horizon_thresholds.get(h, 0.50)
            preds_h = (probs >= th).astype(int)
            prec_h = float(precision_score(y_true_h, preds_h, zero_division=0))
            rec_h = float(recall_score(y_true_h, preds_h, zero_division=0))

            multi_horizon_metrics[str(h)] = {
                "horizon": h,
                "pr_auc": round(pr_auc_h, 3),
                "threshold": round(th, 3),
                "precision": round(prec_h, 3),
                "recall": round(rec_h, 3),
                "positive_count": int(np.sum(y_true_h))
            }

        if "30" not in multi_horizon_metrics:
            raise ValueError(
                f"[{self.subset}] multi-horizon evaluation did not produce a 30-cycle metric"
            )
        pr_auc_30 = multi_horizon_metrics["30"]["pr_auc"]

        # Real Bucket Errors, derived from the same config-driven RUL buckets as
        # the raw/capped metrics block.
        bucket_errors = []
        for bucket in metrics["rul_bucket_metrics"]:
            raw_rmse = bucket["rmse_vs_raw_truth"]
            raw_mae = bucket["mae_vs_raw_truth"]
            bucket_errors.append({
                "bucket": bucket["bucket"],
                "lower": bucket["lower"],
                "upper": bucket["upper"],
                "rmse": round(raw_rmse, 2) if raw_rmse is not None else 0.0,
                "mae": round(raw_mae, 2) if raw_mae is not None else 0.0,
                "sample_count": bucket["sample_count"],
                "rmse_vs_raw_truth": bucket["rmse_vs_raw_truth"],
                "rmse_vs_capped_truth": bucket["rmse_vs_capped_truth"],
                "share_true_rul_above_cap": bucket["share_true_rul_above_cap"],
                "rmse_on_true_rul_le_cap": bucket["rmse_on_true_rul_le_cap"],
                "n_true_rul_le_cap": bucket["n_true_rul_le_cap"],
                "n_true_rul_above_cap": bucket["n_true_rul_above_cap"],
            })

        # Real Feature Importance (Permutation on test sample)
        sample_size = min(100, len(X_test))
        perm = permutation_importance(
            self.rul_p50_model, X_test[:sample_size], y_true[:sample_size],
            n_repeats=3, random_state=42
        )
        importances = perm.importances_mean
        importances = np.maximum(0, importances)
        total_imp = np.sum(importances)
        norm_imp = importances / total_imp if total_imp > 0 else np.ones(len(importances)) / len(importances)

        top_indices = np.argsort(norm_imp)[::-1][:6]
        feature_importance_list = []
        sensor_info = self.sensor_cfg.get("sensors", {})
        colors = [
            "from-indigo-500 to-purple-500", "from-purple-500 to-pink-500",
            "from-cyan-500 to-indigo-500", "from-emerald-500 to-teal-500",
            "from-amber-500 to-orange-500", "from-blue-500 to-cyan-500"
        ]

        for i, idx in enumerate(top_indices):
            f_name = self.feature_cols[idx]
            pct = round(float(norm_imp[idx]) * 100.0, 1)
            m = SENSOR_REGEX.search(f_name)
            desc_str = f_name
            if m:
                s_id = m.group(0)
                s_meta = sensor_info.get(s_id, {})
                s_name = s_meta.get("name", s_id)
                s_sub = s_meta.get("subsystem", "Turbofan")
                desc_str = f"{f_name} ({s_sub} {s_name})"

            feature_importance_list.append({
                "feat": desc_str,
                "weight": f"{pct}%",
                "raw_weight": pct,
                "color": colors[i % len(colors)]
            })

        target_policy = resolve_target_policy(self.subset, metrics)
        target_rmse = target_policy["target_rmse"]
        target_metric = target_policy["target_metric"]
        target_metric_value = target_policy["target_metric_value"]
        target_met = target_policy["target_met"]
        loader = CMAPSSLoader(subset=self.subset)
        cond, faults = loader.operating_conditions, loader.fault_modes
        status = "Target Met" if target_met else "Target Missed"

        # Parity Scatter points for held-out test engines
        parity_scatter = []
        for i in range(len(y_true)):
            act = round(float(y_true[i]), 1)
            pr = round(float(pred_p50[i]), 1)
            diff = round(pr - act, 1)
            pen = round(calculate_nasa_score([act], [pr]), 2)
            cat = "early_safe" if diff < 0 else "late_penalty"
            
            unit_val = int(eval_rows.iloc[i]["unit"]) if "unit" in eval_rows.columns else (i + 1)
            is_censored = bool(censored[i])
            parity_scatter.append({
                "unit": unit_val,
                "actual": act,
                "predicted": pr,
                "p10": round(float(pred_p10[i]), 1),
                "p90": None if is_censored else round(float(pred_p90[i]), 1),
                "censored_at_cap": is_censored,
                "interval_label": ">cap" if is_censored else "p10-p90",
                "diff": diff,
                "penalty": pen,
                "category": cat
            })

        # Generalization gap / Overfitting diagnostics (Dr-DKP criterion)
        cv_file = self.root_dir / "artifacts" / "reports" / f"cv_{self.subset}.json"
        mean_cv_rmse = 0.0
        mean_train_rmse = 0.0
        if cv_file.exists():
            try:
                with open(cv_file, "r", encoding="utf-8") as f:
                    cv_d = json.load(f)
                mean_cv_rmse = float(cv_d.get("mean_rmse", 0.0))
                mean_train_rmse = float(cv_d.get("mean_train_rmse", 0.0))
            except Exception:
                pass

        train_test_gap = round(float(metrics["rmse"] - mean_train_rmse), 2)
        train_val_gap = round(float(mean_cv_rmse - mean_train_rmse), 2)
        val_test_gap = round(float(metrics["rmse"] - mean_cv_rmse), 2)
        gap_ratio_pct = round(float(train_test_gap / metrics["rmse"] * 100.0), 1) if metrics["rmse"] > 0 else 0.0

        generalization_gap = {
            "train_rmse": round(float(mean_train_rmse), 2),
            "val_rmse": round(float(mean_cv_rmse), 2),
            "test_rmse": round(float(metrics["rmse"]), 2),
            "train_test_gap": train_test_gap,
            "train_val_gap": train_val_gap,
            "val_test_gap": val_test_gap,
            "gap_ratio_pct": gap_ratio_pct,
            "overfitting_status": "Low Overfitting" if train_test_gap <= 3.5 else ("Moderate Gap" if train_test_gap <= 6.0 else "High Overfitting Risk"),
            "memorization_risk": "Low (<25%)" if gap_ratio_pct < 25.0 else "Elevated"
        }

        results = {
            "subset": self.subset,
            "algorithm": self.algorithm,
            "protocol_id": get_protocol_id(),
            "condition": cond,
            "faults": faults,
            "metrics": metrics,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "nasa_score": metrics["nasa_score"],
            "early_rate": metrics["early_rate"],
            "late_rate": metrics["late_rate"],
            "nasa_phm08": {
                "score": metrics["nasa_score"],
                "early_rate": metrics["early_rate"],
                "late_rate": metrics["late_rate"],
                "early_count": int(sum(1 for p in parity_scatter if p["category"] == "early_safe")),
                "late_count": int(sum(1 for p in parity_scatter if p["category"] == "late_penalty")),
                "formula": "S = sum(exp(-d/13)-1 if d<0 else exp(d/10)-1)"
            },
            "generalization_gap": generalization_gap,
            "pr_auc": round(pr_auc_30, 3),
            "target_rmse": target_rmse,
            "target_metric": target_metric,
            "target_metric_value": round(float(target_metric_value), 4),
            "target_metric_provenance": target_policy["target_metric_provenance"],
            "target_provenance": target_policy["target_provenance"],
            "target_met": target_met,
            "status": status,
            "interval_coverage": round(interval_coverage, 4),
            "mean_interval_width": (
                round(float(mean_interval_width), 2)
                if mean_interval_width is not None
                else None
            ),
            "coverage_target": target_coverage,
            "coverage_wilson_ci_95": [round(wilson_lower, 4), round(wilson_upper, 4)],
            "coverage_wilson_ci_contains_target": all_ci_contains_target,
            "coverage_all_ci_excludes_target": all_ci_excludes_target,
            "coverage_all_ci_contains_target": all_ci_contains_target,
            "coverage_ci_excludes_80": all_ci_excludes_target,
            "coverage_ci_excludes_target": ci_excludes_target,
            "coverage_ci_contains_target": ci_contains_target,
            "coverage_metric": coverage_policy["coverage_metric"],
            "coverage_metric_value": round(float(coverage_policy["coverage_metric_value"]), 4),
            "coverage_metric_wilson_ci_95": coverage_policy["coverage_metric_wilson_ci_95"],
            "coverage_metric_ci_lower": coverage_policy["coverage_metric_ci_lower"],
            "coverage_metric_ci_upper": coverage_policy["coverage_metric_ci_upper"],
            "coverage_metric_provenance": coverage_policy["coverage_metric_provenance"],
            "target_coverage": coverage_policy["target_coverage"],
            "target_coverage_provenance": coverage_policy["target_coverage_provenance"],
            "target_coverage_met": coverage_policy["target_coverage_met"],
            "coverage_ci_lower_meets_target": coverage_policy["coverage_ci_lower_meets_target"],
            "coverage_all": round(float(coverage_stats["coverage_all"]), 4),
            "coverage_all_n": coverage_stats["coverage_all_n"],
            "coverage_all_wilson_ci_95": coverage_stats["coverage_all_wilson_ci_95"],
            "coverage_le_cap": (
                round(float(coverage_stats["coverage_le_cap"]), 4)
                if coverage_stats["coverage_le_cap"] is not None
                else None
            ),
            "coverage_le_cap_n": coverage_stats["coverage_le_cap_n"],
            "coverage_le_cap_wilson_ci_95": coverage_stats["coverage_le_cap_wilson_ci_95"],
            "coverage_above_cap": (
                round(float(coverage_stats["coverage_above_cap"]), 4)
                if coverage_stats["coverage_above_cap"] is not None
                else None
            ),
            "coverage_above_cap_n": coverage_stats["coverage_above_cap_n"],
            "coverage_above_cap_wilson_ci_95": coverage_stats["coverage_above_cap_wilson_ci_95"],
            "coverage_split": coverage_stats["coverage_split"],
            "coverage_rule": coverage_stats["coverage_rule"],
            "n_censored_intervals": coverage_stats["n_censored_intervals"],
            "n_closed_intervals": coverage_stats["n_closed_intervals"],
            "censored_interval_rate": round(float(coverage_stats["censored_interval_rate"]), 4),
            "coverage_by_bucket": coverage_by_bucket,
            "conformal_buckets": conformal_meta,
            "test_by_regime": test_by_regime,
            "test_rmse_bootstrap_ci_95": [round(boot_lower, 2), round(boot_upper, 2)],
            "multi_horizon_metrics": multi_horizon_metrics,
            "test_engines_count": len(y_true),
            "train_time_sec": self.training_time_sec,
            "calibration_offset": round(global_cal_offset, 3),
            "bucket_errors": bucket_errors,
            "feature_importance": feature_importance_list,
            "parity_scatter": parity_scatter
        }

        # Save evaluation report if requested
        if save_report:
            report_file = self.root_dir / "artifacts" / "reports" / f"evaluation_{self.subset}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            if report_file.exists():
                try:
                    with open(report_file, "r", encoding="utf-8") as f_prev:
                        prev = json.load(f_prev)
                        if prev.get("algorithm") == self.algorithm and "multi_seed" in prev:
                            results["multi_seed"] = prev["multi_seed"]
                except Exception:
                    pass
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        print(
            f"[{self.subset}] Real Evaluation: {target_metric}={target_metric_value:.4f} "
            f"(target<={target_rmse}), raw_rmse={metrics['rmse_vs_raw_truth']}, "
            f"capped_rmse={metrics['rmse_vs_capped_truth']}, "
            f"NASA={metrics['nasa_score']}, Status={status}"
        )
        return results

    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates the active model on an arbitrary DataFrame (e.g. holdout engines).
        Computes RMSE, MAE, NASA Score, and directional error rates.
        """
        eval_df = df.copy()
        for c in self.feature_cols:
            if c not in eval_df.columns:
                raise ValueError(
                    f"[{self.subset}] evaluate_dataframe: input dataframe is missing "
                    f"required feature column '{c}'. Ensure the feature pipeline "
                    f"produces all {len(self.feature_cols)} expected features."
                )
        X = eval_df[self.feature_cols].values
        rul_col = "rul_capped" if "rul_capped" in eval_df.columns else ("rul_raw" if "rul_raw" in eval_df.columns else "RUL")
        y = eval_df[rul_col].values
        pred = np.maximum(0.0, self.rul_p50_model.predict(X))
        return calculate_metrics(y, pred)

    def predict_point(
        self,
        row_dict: Dict[str, Any],
        recent_history: Optional[List[Dict[str, Any]]] = None,
        compute_shap: bool = False
    ) -> Dict[str, Any]:
        """
        Generates prediction contract according to PRD §9.9:
        {engine_id, subset, cycle, rul_p10, rul_p50, rul_p90, fail_prob_10, fail_prob_30, fail_prob_60,
         classifier_threshold_30, anomaly_score, health_index, top_contributors, top_sensors, model_version}
        """
        # Validate that all required features are present — never silently default to 0.0
        missing_features = [c for c in self.feature_cols if c not in row_dict]
        if missing_features:
            raise ValueError(
                f"[{self.subset}] predict_point received row missing {len(missing_features)} "
                f"required feature(s): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}. "
                f"Total features expected: {len(self.feature_cols)}."
            )
        feats = np.array([[float(row_dict[c]) for c in self.feature_cols]], dtype=np.float32)
        
        # Anomaly detector
        raw_score = self.anomaly_detector.decision_function(feats)[0] if self.anomaly_detector else 0.0
        anomaly_score = float(np.clip(0.5 - (raw_score * 2.0), 0.0, 1.0))

        # Quantile RUL Regressors
        raw_p10 = float(self.rul_p10_model.predict(feats)[0])
        rul_p50 = float(np.maximum(0.0, self.rul_p50_model.predict(feats)[0]))
        raw_p90 = float(self.rul_p90_model.predict(feats)[0])

        cal_offset: Optional[float] = None
        censored_at_cap = False
        if self.interval_calibration_offset is None and not self.conformal_buckets:
            # Uncalibrated model (e.g. hand-built unit model): report a degenerate interval
            # rather than inventing an offset.
            rul_p10, rul_p90 = rul_p50, rul_p50
        else:
            cal_offset = self._offset_for_p50(rul_p50)
            rul_p10 = float(np.maximum(0.0, min(rul_p50, min(raw_p10, rul_p50) - cal_offset)))
            rul_p90 = float(max(rul_p50, max(raw_p90, rul_p50) + cal_offset))
            censored_at_cap = bool(rul_p50 >= (get_rul_cap() - cal_offset))
            if censored_at_cap:
                # The calibrated upper bound reaches the RUL cap: the honest interval
                # is open-ended and must never be rendered as a closed p10-p90 range.
                rul_p90 = None

        interval_label = ">cap" if censored_at_cap else "p10-p90"

        # Multi-horizon probabilities [10, 30, 60]
        horizons = [10, 30, 60]
        raw_probs = {}
        for h in horizons:
            cls_h = self.classifiers.get(h, self.classifier)
            if cls_h:
                proba = cls_h.predict_proba(feats)[0]
                raw_probs[h] = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                raw_probs[h] = 0.0

        # Enforce monotone probabilities across horizons: p10 <= p30 <= p60
        fail_prob_10 = round(raw_probs[10], 3)
        fail_prob_30 = round(max(fail_prob_10, raw_probs[30]), 3)
        fail_prob_60 = round(max(fail_prob_30, raw_probs[60]), 3)

        health_index = float(row_dict.get("health_index", 1.0))

        # Conservative RUL: median of last 3 p10 predictions
        p10_history = [rul_p10]
        if recent_history:
            for h in recent_history[-2:]:
                if "rul_p10" in h:
                    p10_history.append(h["rul_p10"])
        conservative_rul = float(np.median(p10_history))

        # SHAP calculation or normalized deviation fallback
        top_contributors = []
        if compute_shap:
            explainer = self.get_shap_explainer()
            if explainer is not None:
                shap_vals = explainer.shap_values(feats)
                shap_row = shap_vals[0] if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2 else shap_vals

                sensor_shap: Dict[str, float] = {}
                sensor_abs_shap: Dict[str, float] = {}
                for f_name, val in zip(self.feature_cols, shap_row):
                    m = re.match(r"^(s\d+)", f_name)
                    if not m:
                        continue
                    s_id = m.group(1)
                    sensor_shap[s_id] = sensor_shap.get(s_id, 0.0) + float(val)
                    sensor_abs_shap[s_id] = sensor_abs_shap.get(s_id, 0.0) + abs(float(val))

                total_abs = sum(sensor_abs_shap.values()) if sensor_abs_shap else 1.0
                sorted_sensors = sorted(sensor_abs_shap.keys(), key=lambda s: sensor_abs_shap[s], reverse=True)
                for s in sorted_sensors[:3]:
                    val = sensor_shap[s]
                    subsystem = self.sensor_cfg.get("sensors", {}).get(s, {}).get("subsystem", "Unknown")
                    s_name = self.sensor_cfg.get("sensors", {}).get(s, {}).get("name", s)
                    top_contributors.append({
                        "sensor": s,
                        "name": s_name,
                        "shap_value": round(float(val), 3),
                        "contribution": round(min(1.0, max(0.01, abs(float(val)) / (total_abs + 1e-6))), 3),
                        "direction": "shortens_life" if val < 0.0 else "prolongs_life",
                        "subsystem": subsystem
                    })

        if not top_contributors:
            # Fallback to normalized deviations for speed on fleet ticks
            sensor_names = [f"s{i}" for i in range(1, 22)]
            top_deviations = []
            for s in sensor_names:
                dev_key = f"{s}_dev_base"
                if dev_key in row_dict:
                    val = float(row_dict[dev_key])
                    top_deviations.append((s, val, abs(val)))
            top_deviations.sort(key=lambda x: x[2], reverse=True)
            for s, val, _ in top_deviations[:3]:
                subsystem = self.sensor_cfg.get("sensors", {}).get(s, {}).get("subsystem", "Unknown")
                s_name = self.sensor_cfg.get("sensors", {}).get(s, {}).get("name", s)
                top_contributors.append({
                    "sensor": s,
                    "name": s_name,
                    "shap_value": round(float(val), 3),
                    "contribution": round(min(1.0, max(0.05, abs(float(val)) / 10.0)), 3),
                    "direction": "shortens_life" if val < 0.0 else "prolongs_life",
                    "subsystem": subsystem
                })

        if not top_contributors and self.feature_cols:
            for f in self.feature_cols:
                m = re.match(r"^(s\d+)", f)
                s_id = m.group(1) if m else f
                if not any(tc.get("sensor") == s_id for tc in top_contributors):
                    subsystem = self.sensor_cfg.get("sensors", {}).get(s_id, {}).get("subsystem", "Unknown")
                    s_name = self.sensor_cfg.get("sensors", {}).get(s_id, {}).get("name", s_id)
                    top_contributors.append({
                        "sensor": s_id,
                        "name": s_name,
                        "shap_value": 0.0,
                        "contribution": 0.33,
                        "direction": "shortens_life",
                        "subsystem": subsystem
                    })
                if len(top_contributors) >= 3:
                    break

        top_sensors = [c["sensor"] for c in top_contributors]

        return {
            "engine_id": f"{self.subset}-{int(row_dict['unit']):03d}",
            "unit": int(row_dict["unit"]),
            "subset": self.subset,
            "cycle": int(row_dict["cycle"]),
            "rul_p10": round(rul_p10, 1),
            "rul_p50": round(rul_p50, 1),
            "rul_p90": round(rul_p90, 1) if rul_p90 is not None else None,
            "censored_at_cap": censored_at_cap,
            "interval_label": interval_label,
            "rul_interval_label": interval_label,
            "interval_width": (
                round(float(rul_p90 - rul_p10), 1)
                if rul_p90 is not None
                else None
            ),
            "conservative_rul": round(conservative_rul, 1),
            "fail_prob_10": fail_prob_10,
            "fail_prob_30": fail_prob_30,
            "fail_prob_60": fail_prob_60,
            "classifier_threshold_30": getattr(self, "classifier_threshold_30", 0.70),
            "anomaly_score": round(anomaly_score, 3),
            "health_index": round(health_index, 3),
            "top_contributors": top_contributors,
            "top_sensors": top_sensors,
            "model_version": f"rul-{self.subset.lower()}@{getattr(self, 'version', 'v1.0')}"
        }

    def predict_trajectories(self, unit_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Ultra-fast vectorized prediction of full cycle trajectories for an engine.
        Replaces slow row-by-row predict_point calls to achieve < 50ms latency.
        """
        if unit_df.empty:
            return []
        df_sorted = unit_df.sort_values("cycle").reset_index(drop=True)
        missing_features = [c for c in self.feature_cols if c not in df_sorted.columns]
        if missing_features:
            raise ValueError(f"[{self.subset}] predict_trajectories missing features: {missing_features[:5]}")

        feats = df_sorted[self.feature_cols].to_numpy(dtype=np.float32)
        n = len(feats)

        # Anomaly scores
        if self.anomaly_detector:
            raw_scores = self.anomaly_detector.decision_function(feats)
            anomaly_scores = np.clip(0.5 - (raw_scores * 2.0), 0.0, 1.0)
        else:
            anomaly_scores = np.zeros(n, dtype=np.float32)

        # Quantile RUL Regressors
        raw_p10 = self.rul_p10_model.predict(feats) if self.rul_p10_model else np.zeros(n)
        rul_p50 = np.maximum(0.0, self.rul_p50_model.predict(feats)) if self.rul_p50_model else np.zeros(n)
        raw_p90 = self.rul_p90_model.predict(feats) if self.rul_p90_model else np.zeros(n)

        censored_at_cap = np.zeros(n, dtype=bool)
        if self.interval_calibration_offset is None and not self.conformal_buckets:
            # Uncalibrated model: degenerate intervals instead of a typed default offset.
            rul_p10, rul_p90 = rul_p50, rul_p50
        else:
            cal_offsets = self._interval_offsets(rul_p50)
            rul_p10 = np.maximum(0.0, np.minimum(rul_p50, np.minimum(raw_p10, rul_p50) - cal_offsets))
            rul_p90 = np.maximum(rul_p50, np.maximum(raw_p90, rul_p50) + cal_offsets)
            censored_at_cap = rul_p50 >= (get_rul_cap() - cal_offsets)

        # Multi-horizon probabilities [10, 30, 60]
        fail_prob_10 = np.zeros(n, dtype=np.float32)
        fail_prob_30 = np.zeros(n, dtype=np.float32)
        fail_prob_60 = np.zeros(n, dtype=np.float32)
        for h, target_arr in [(10, fail_prob_10), (30, fail_prob_30), (60, fail_prob_60)]:
            cls_h = self.classifiers.get(h, self.classifier)
            if cls_h:
                probs = cls_h.predict_proba(feats)
                target_arr[:] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        fail_prob_30 = np.maximum(fail_prob_10, fail_prob_30)
        fail_prob_60 = np.maximum(fail_prob_30, fail_prob_60)

        cycles = df_sorted["cycle"].to_numpy(dtype=int)
        true_ruls = df_sorted["rul_raw"].to_numpy(dtype=float) if "rul_raw" in df_sorted.columns else np.zeros(n)
        health_indices = df_sorted["health_index"].to_numpy(dtype=float) if "health_index" in df_sorted.columns else np.ones(n)

        def get_col(col_name: str) -> np.ndarray:
            return df_sorted[col_name].to_numpy(dtype=float) if col_name in df_sorted.columns else np.zeros(n)

        s2_raw = get_col("s2")
        s2_ewma = get_col("s2_ewma")
        s3_raw = get_col("s3")
        s3_ewma = get_col("s3_ewma")
        s4_raw = get_col("s4")
        s4_ewma = get_col("s4_ewma")
        s7_raw = get_col("s7")
        s7_ewma = get_col("s7_ewma")

        trajectory = []
        for i in range(n):
            is_censored = bool(censored_at_cap[i])
            trajectory.append({
                "cycle": int(cycles[i]),
                "rul_p10": round(float(rul_p10[i]), 1),
                "rul_p50": round(float(rul_p50[i]), 1),
                "rul_p90": None if is_censored else round(float(rul_p90[i]), 1),
                "censored_at_cap": is_censored,
                "interval_label": ">cap" if is_censored else "p10-p90",
                "true_rul": round(float(true_ruls[i]), 1),
                "health_index": round(float(health_indices[i]), 3),
                "anomaly_score": round(float(anomaly_scores[i]), 3),
                "fail_prob_10": round(float(fail_prob_10[i]), 3),
                "fail_prob_30": round(float(fail_prob_30[i]), 3),
                "fail_prob_60": round(float(fail_prob_60[i]), 3),
                "s2_raw": round(float(s2_raw[i]), 2),
                "s2_ewma": round(float(s2_ewma[i]), 2),
                "s3_raw": round(float(s3_raw[i]), 2),
                "s3_ewma": round(float(s3_ewma[i]), 2),
                "s4_raw": round(float(s4_raw[i]), 2),
                "s4_ewma": round(float(s4_ewma[i]), 2),
                "s7_raw": round(float(s7_raw[i]), 2),
                "s7_ewma": round(float(s7_ewma[i]), 2),
            })
        return trajectory

    def evaluate_multi_seed(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, rul_truth: np.ndarray, seeds: List[int] = [42, 43, 44]
    ) -> Dict[str, Any]:
        """Trains and evaluates model over 3 distinct random seeds."""
        rmses, maes, nasas, praucs, coverages = [], [], [], [], []
        last_indices = test_df.groupby("unit")["cycle"].idxmax()
        eval_rows = test_df.loc[last_indices].sort_values("unit")
        y_true = rul_truth
        y_true_cls = (y_true <= 30).astype(int)

        for s in seeds:
            eval_ms = ModelService(self.subset)
            eval_ms.algorithm = self.algorithm
            if s != 42 or self.rul_p50_model is None:
                eval_ms.train(train_df, random_state=s)
            else:
                eval_ms = self

            X_test = eval_rows[eval_ms.feature_cols].values
            pred_p50 = np.maximum(0.0, eval_ms.rul_p50_model.predict(X_test))
            m = calculate_metrics(y_true, pred_p50)
            rmses.append(m["rmse"])
            maes.append(m["mae"])
            nasas.append(m["nasa_score"])

            raw_p10 = eval_ms.rul_p10_model.predict(X_test)
            raw_p90 = eval_ms.rul_p90_model.predict(X_test)
            cal_offsets = eval_ms._interval_offsets(pred_p50)
            cal_p10 = np.maximum(0.0, np.minimum(pred_p50, np.minimum(raw_p10, pred_p50) - cal_offsets))
            cal_p90 = np.maximum(pred_p50, np.maximum(raw_p90, pred_p50) + cal_offsets)
            censored = pred_p50 >= (get_rul_cap() - cal_offsets)
            cov = float(np.mean(censored_coverage_mask(y_true, cal_p10, cal_p90, censored)))
            coverages.append(cov)

            probs_mat = eval_ms.classifier.predict_proba(X_test)
            probs = probs_mat[:, 1] if probs_mat.shape[1] > 1 else probs_mat[:, 0]
            if len(np.unique(y_true_cls)) > 1:
                praucs.append(float(average_precision_score(y_true_cls, probs)))
            else:
                praucs.append(float(np.mean(y_true_cls == (probs >= 0.5))))

            import gc
            gc.collect()

        multi_seed_report = {
            "algorithm": self.algorithm,
            "seeds": seeds,
            "rmse_mean": round(float(np.mean(rmses)), 2),
            "rmse_std": round(float(np.std(rmses)), 2),
            "mae_mean": round(float(np.mean(maes)), 2),
            "mae_std": round(float(np.std(maes)), 2),
            "nasa_mean": round(float(np.mean(nasas)), 2),
            "nasa_std": round(float(np.std(nasas)), 2),
            "pr_auc_mean": round(float(np.mean(praucs)), 3),
            "pr_auc_std": round(float(np.std(praucs)), 3),
            "coverage_mean": round(float(np.mean(coverages)), 4),
            "coverage_std": round(float(np.std(coverages)), 4)
        }

        report_file = self.root_dir / "artifacts" / "reports" / f"evaluation_{self.subset}.json"
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["multi_seed"] = multi_seed_report
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return multi_seed_report

    def compare_deep_learning_models(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, rul_truth: np.ndarray
    ) -> Dict[str, Any]:
        """
        Benchmarks Ridge, HistGB, LightGBM default, LightGBM tuned, and PyTorch (CNN/GRU if present).
        Champion rule: lowest mean CV RMSE, ties by val NASA.
        Saves to artifacts/reports/model_comparison_{subset}.json.
        """
        print(f"[{self.subset}] Benchmarking Ridge, HistGB, LightGBM, and Deep Learning models...")
        eval_rows = eval_frames(test_df, mode="last", seed=DEFAULT_SEED, n_draws=1)[0]
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df["rul_capped"].values
        X_test = eval_rows[self.feature_cols].values
        y_test = rul_truth

        # Persistent train engines for CV
        from src.common.splits import get_persistent_split
        all_units = [int(u) for u in np.unique(train_df["unit"].values)]
        split = get_persistent_split(self.subset, all_units=all_units)
        train_units = split["train_units"]
        train_only_df = train_df[train_df["unit"].isin(train_units)].copy().reset_index(drop=True)
        X_tr_cv = train_only_df[self.feature_cols].values
        y_tr_cv = train_only_df["rul_capped"].values
        groups_cv = train_only_df["unit"].values

        gkf = GroupKFold(n_splits=DEFAULT_CV_FOLDS)
        splits_list = list(gkf.split(X_tr_cv, y_tr_cv, groups=groups_cv))

        # Protocol matching evaluate_cv: GroupKFold-5 on persistent train engines with
        # randomly censored validation targets, so comparison CV is test-like.
        cv_proto = load_cv_protocol()
        cv_mode = cv_proto["mode"]
        cv_draws = cv_proto["n_draws"]
        cv_min_cycles = cv_proto["min_cycles"]
        cv_protocol_id = get_protocol_id(mode=cv_mode, seed=DEFAULT_SEED, n_draws=cv_draws)

        def get_cv_metrics(model_fn):
            fold_rmses, fold_nasas = [], []
            for tr_i, val_i in splits_list:
                m = model_fn()
                m.fit(X_tr_cv[tr_i], y_tr_cv[tr_i])
                val_frames = eval_frames(
                    train_only_df.iloc[val_i],
                    mode=cv_mode,
                    seed=DEFAULT_SEED,
                    n_draws=cv_draws,
                    min_cycles=cv_min_cycles,
                )
                val_frame = pd.concat(val_frames, ignore_index=True)
                preds = np.maximum(0.0, m.predict(val_frame[self.feature_cols].values))
                met = calculate_metrics(val_frame["rul_capped"].values, preds)
                fold_rmses.append(met["rmse"])
                fold_nasas.append(met["nasa_score"])
            return (
                float(np.mean(fold_rmses)),
                float(np.std(fold_rmses)),
                float(np.mean(fold_nasas)),
                float(np.std(fold_nasas))
            )

        # Load hyperparameters dynamically from configs/models.yaml
        models_cfg = load_yaml(self.root_dir / "configs" / "models.yaml")
        ridge_p = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in models_cfg["baselines"]["ridge"].items()}
        hgb_p = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in models_cfg["baselines"]["hist_gb"].items()}
        hgb_p["loss"] = "squared_error"
        lgb_def_p = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in models_cfg["baselines"]["lightgbm"].items()}
        lgb_def_p["objective"] = "regression"
        lgb_def_p["verbose"] = -1

        # 1. Ridge Baseline
        t0 = time.time()
        ridge_cv_rmse, ridge_cv_std, ridge_cv_nasa, ridge_cv_std_nasa = get_cv_metrics(lambda: Ridge(**ridge_p))
        ridge_model = Ridge(**ridge_p).fit(X_train, y_train)
        ridge_pred = np.maximum(0.0, ridge_model.predict(X_test))
        ridge_met = calculate_metrics(y_test, ridge_pred)
        ridge_time = round(time.time() - t0, 2)

        # 2. HistGradientBoosting Baseline
        t0 = time.time()
        hgb_cv_rmse, hgb_cv_std, hgb_cv_nasa, hgb_cv_std_nasa = get_cv_metrics(
            lambda: HistGradientBoostingRegressor(**hgb_p)
        )
        hgb_model = HistGradientBoostingRegressor(**hgb_p).fit(X_train, y_train)
        hgb_pred = np.maximum(0.0, hgb_model.predict(X_test))
        hgb_met = calculate_metrics(y_test, hgb_pred)
        hgb_time = round(time.time() - t0, 2)

        # 3. LightGBM Default
        t0 = time.time()
        lgb_def_cv_rmse, lgb_def_cv_std, lgb_def_cv_nasa, lgb_def_cv_std_nasa = get_cv_metrics(
            lambda: lgb.LGBMRegressor(**lgb_def_p)
        )
        lgb_def_model = lgb.LGBMRegressor(**lgb_def_p).fit(X_train, y_train)
        lgb_def_pred = np.maximum(0.0, lgb_def_model.predict(X_test))
        lgb_def_met = calculate_metrics(y_test, lgb_def_pred)
        lgb_def_time = round(time.time() - t0, 2)

        models_list = [
            {
                "model_name": "Ridge Regression",
                "architecture": "Linear",
                "servable": True,
                "cv_mean_rmse": round(ridge_cv_rmse, 2),
                "cv_std_rmse": round(ridge_cv_std, 2),
                "cv_mean_nasa": round(ridge_cv_nasa, 1),
                "cv_std_nasa": round(ridge_cv_std_nasa, 1),
                "rmse": ridge_met["rmse"],
                "mae": ridge_met["mae"],
                "nasa_score": ridge_met["nasa_score"],
                "train_time_sec": ridge_time
            },
            {
                "model_name": "HistGradientBoosting Regressor",
                "architecture": "GBDT",
                "servable": True,
                "cv_mean_rmse": round(hgb_cv_rmse, 2),
                "cv_std_rmse": round(hgb_cv_std, 2),
                "cv_mean_nasa": round(hgb_cv_nasa, 1),
                "cv_std_nasa": round(hgb_cv_std_nasa, 1),
                "rmse": hgb_met["rmse"],
                "mae": hgb_met["mae"],
                "nasa_score": hgb_met["nasa_score"],
                "train_time_sec": hgb_time
            },
            {
                "model_name": "LightGBM Regressor (Default)",
                "architecture": "LightGBM",
                "servable": True,
                "cv_mean_rmse": round(lgb_def_cv_rmse, 2),
                "cv_std_rmse": round(lgb_def_cv_std, 2),
                "cv_mean_nasa": round(lgb_def_cv_nasa, 1),
                "cv_std_nasa": round(lgb_def_cv_std_nasa, 1),
                "rmse": lgb_def_met["rmse"],
                "mae": lgb_def_met["mae"],
                "nasa_score": lgb_def_met["nasa_score"],
                "train_time_sec": lgb_def_time
            }
        ]

        # 4. LightGBM Tuned (generate only if actually tuned variant exists, drop Untuned duplicates)
        best_params_path = self.root_dir / "configs" / f"best_params_{self.subset}.json"
        is_tuned = best_params_path.exists()
        if is_tuned:
            with open(best_params_path, "r", encoding="utf-8") as f:
                tuned_p = json.load(f)
            tuned_p["objective"] = "regression"
            tuned_p["verbose"] = -1

            t0 = time.time()
            lgb_tuned_cv_rmse, lgb_tuned_cv_std, lgb_tuned_cv_nasa, lgb_tuned_cv_std_nasa = get_cv_metrics(
                lambda: lgb.LGBMRegressor(**tuned_p)
            )
            lgb_tuned_model = lgb.LGBMRegressor(**tuned_p).fit(X_train, y_train)
            lgb_tuned_pred = np.maximum(0.0, lgb_tuned_model.predict(X_test))
            lgb_tuned_met = calculate_metrics(y_test, lgb_tuned_pred)
            lgb_tuned_time = round(time.time() - t0, 2)

            models_list.append({
                "model_name": "LightGBM Regressor (Tuned)",
                "architecture": "LightGBM",
                "servable": True,
                "cv_mean_rmse": round(lgb_tuned_cv_rmse, 2),
                "cv_std_rmse": round(lgb_tuned_cv_std, 2),
                "cv_mean_nasa": round(lgb_tuned_cv_nasa, 1),
                "cv_std_nasa": round(lgb_tuned_cv_std_nasa, 1),
                "rmse": lgb_tuned_met["rmse"],
                "mae": lgb_tuned_met["mae"],
                "nasa_score": lgb_tuned_met["nasa_score"],
                "train_time_sec": lgb_tuned_time
            })

        # 5. Deep Models (lazy import if PyTorch present)
        try:
            from src.models.neural_baseline import NeuralRULBenchmark
            window_len = int(load_yaml(self.root_dir / "configs" / "pipeline.yaml")["features"]["sequence_window_len"])
            bench = NeuralRULBenchmark(feature_cols=self.feature_cols, window_size=window_len, seed=DEFAULT_SEED)
            cnn_res = bench.evaluate_architecture("cnn", train_df, test_df, y_test)
            gru_res = bench.evaluate_architecture("gru", train_df, test_df, y_test)
            # Deep benchmarks are reported but not servable by ModelService and do not
            # implement the censored CV protocol, so they are ineligible for champion.
            for deep_res in (cnn_res, gru_res):
                deep_res["servable"] = False
                deep_res["cv_protocol_id"] = None
                models_list.append(deep_res)
            # Release torch/benchmark objects before the caller retrains the champion;
            # otherwise the deep graphs and reference/copy tensors stay resident.
            del bench, cnn_res, gru_res
            gc.collect()
        except Exception as e:
            # Never silently drop comparison rows: record that deep models were not run
            # and why, so the manifest keeps the full candidate set.
            print(f"[{self.subset}] PyTorch deep benchmarks not run: {e}")
            for arch, name in (
                ("CNN", "1D-CNN PyTorch (30-Cycle Window)"),
                ("GRU", "GRU PyTorch (30-Cycle Sequence)"),
            ):
                models_list.append({
                    "model_name": name,
                    "architecture": arch,
                    "servable": False,
                    "cv_mean_rmse": None,
                    "cv_std_rmse": None,
                    "cv_mean_nasa": None,
                    "cv_std_nasa": None,
                    "rmse": None,
                    "mae": None,
                    "nasa_score": None,
                    "train_time_sec": None,
                    "status": "not_run",
                    "not_run_reason": f"deep models: not run ({e})",
                })

        # Best CV candidate for comparison reporting. The served champion is decided by
        # the canonical protocol (scripts/evaluate_all_models.select_champion), which also
        # records the CV margin and tie-break in the run manifest.
        missing_cv = [
            m["model_name"]
            for m in models_list
            if m.get("servable", True) and (m.get("cv_mean_rmse") is None or m.get("cv_std_rmse") is None)
        ]
        if missing_cv:
            raise ValueError(f"[{self.subset}] Servable candidates missing CV mean/std: {missing_cv}")
        servable_candidates = [m for m in models_list if m.get("servable", True)]
        if not servable_candidates:
            raise ValueError(f"[{self.subset}] No servable candidate has CV metrics.")
        champion_candidate = min(servable_candidates, key=lambda m: float(m["cv_mean_rmse"]))
        champion_name = champion_candidate["model_name"]

        champion_rule = (
            f"Selected {champion_name} via lowest mean CV RMSE ({champion_candidate['cv_mean_rmse']:.2f}) "
            f"on TRAIN engines (GroupKFold-5, capped RUL target matching champion training path, "
            f"NASA asymmetric penalty score). Test set metrics are reported for evaluation only, never used for selection."
        )

        comparison = {
            "subset": self.subset,
            "cv_protocol_id": cv_protocol_id,
            "cv_evaluation_mode": cv_mode,
            "cv_n_draws": cv_draws,
            "cv_min_cycles": cv_min_cycles,
            "champion_selection_rule": champion_rule,
            "champion": champion_name,
            "models": models_list
        }

        comp_file = self.root_dir / "artifacts" / "reports" / f"model_comparison_{self.subset}.json"
        comp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        return comparison

    evaluate_model_comparison = compare_deep_learning_models

    def _to_bundle(self, version: Optional[str] = None) -> Dict[str, Any]:
        bundle = {
            "subset": self.subset,
            "algorithm": getattr(self, "algorithm", "lightgbm"),
            "version": version or getattr(self, "version", "v1.0"),
            "sklearn_version": sklearn.__version__,
            "feature_cols": self.feature_cols,
            "anomaly_detector": self.anomaly_detector,
            "rul_p10_model": self.rul_p10_model,
            "rul_p50_model": self.rul_p50_model,
            "rul_p90_model": self.rul_p90_model,
            "classifier": self.classifier,
            "classifiers": getattr(self, "classifiers", {}),
            "horizon_thresholds": getattr(self, "horizon_thresholds", {10: 0.50, 30: 0.70, 60: 0.50}),
            "classifier_threshold_30": getattr(self, "classifier_threshold_30", 0.70),
            "conformal_q": self.conformal_q,
            "interval_calibration_offset": self.interval_calibration_offset,
            "conformal_buckets": self.conformal_buckets,
            "training_time_sec": self.training_time_sec
        }
        if hasattr(self, "bundle_metadata") and isinstance(self.bundle_metadata, dict):
            for k, v in self.bundle_metadata.items():
                if k not in bundle:
                    bundle[k] = v
        return bundle

    def save(self, version: Optional[str] = None, path: Optional[Path] = None) -> Path:
        """Saves model bundle to versioned or explicit file path. Must NOT write active bundle."""
        bundle = self._to_bundle(version=version)
        if path:
            target_path = Path(path)
        elif version:
            target_path = self.artifacts_dir / f"model_bundle_{self.subset}_{version}.joblib"
        else:
            target_path = self.artifacts_dir / f"model_bundle_{self.subset}_temp_{int(time.time()*1000)}.joblib"

        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, target_path)
        print(f"[{self.subset}] Saved versioned model bundle to {target_path}")
        return target_path

    def promote(self, version: Optional[str] = None, path: Optional[Path] = None) -> Path:
        """Promotes a versioned bundle or in-memory model to the active champion bundle."""
        if path:
            source_path = Path(path)
            bundle = joblib.load(source_path)
            if version:
                bundle["version"] = version
        elif version:
            versioned_path = self.artifacts_dir / f"model_bundle_{self.subset}_{version}.joblib"
            if not versioned_path.exists():
                raise FileNotFoundError(f"Cannot promote missing versioned bundle: {versioned_path}")
            bundle = joblib.load(versioned_path)
            bundle["version"] = version
        else:
            bundle = self._to_bundle(version=getattr(self, "version", "v1.0"))

        active_path = self.artifacts_dir / f"model_bundle_{self.subset}.joblib"
        joblib.dump(bundle, active_path)
        print(f"[{self.subset}] Successfully promoted model to active champion bundle at {active_path}")
        return active_path

    def load(self, version: Optional[str] = None):
        """Loads model bundle with seamless backward compatibility for older bundles."""
        if version:
            file_path = self.artifacts_dir / f"model_bundle_{self.subset}_{version}.joblib"
            if not file_path.exists():
                file_path = self.artifacts_dir / f"model_bundle_{self.subset}.joblib"
        else:
            file_path = self.artifacts_dir / f"model_bundle_{self.subset}.joblib"

        if not file_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {file_path}")
        bundle = joblib.load(file_path)

        bundle_sklearn = bundle.get("sklearn_version")
        if bundle_sklearn and bundle_sklearn != sklearn.__version__:
            import warnings
            warnings.warn(
                f"Model bundle '{file_path.name}' was serialized with scikit-learn {bundle_sklearn}, "
                f"but active environment has {sklearn.__version__}."
            )
        self.sklearn_version = bundle_sklearn or sklearn.__version__
        self.algorithm = bundle.get("algorithm", "lightgbm")

        self.feature_cols = bundle["feature_cols"]
        self.anomaly_detector = bundle["anomaly_detector"]
        self.rul_p10_model = bundle["rul_p10_model"]
        self.rul_p50_model = bundle["rul_p50_model"]
        self.rul_p90_model = bundle["rul_p90_model"]
        self.classifier = bundle["classifier"]
        self.classifiers = bundle.get("classifiers", {30: self.classifier})
        self.horizon_thresholds = bundle.get("horizon_thresholds", {10: 0.50, 30: 0.70, 60: 0.50})
        self.classifier_threshold_30 = bundle.get("classifier_threshold_30", self.horizon_thresholds.get(30, 0.70))
        
        loaded_q = bundle.get("conformal_q", bundle.get("interval_calibration_offset"))
        # Absence is preserved as None (never replaced by a typed default) and surfaces
        # as an explicit error from the methods that need the calibrated offset.
        self.conformal_q = None if loaded_q is None else float(loaded_q)
        self.interval_calibration_offset = self.conformal_q
        self.conformal_buckets = bundle.get("conformal_buckets")
        if self.conformal_buckets is not None:
            # Legacy bundles may not have a global_q alongside bucket values.
            self.conformal_buckets.setdefault("global_q", self.interval_calibration_offset)

        self.training_time_sec = bundle.get("training_time_sec", 0.0)
        self.version = bundle.get("version", version or "v1.0")
        self.bundle_metadata = {
            k: v for k, v in bundle.items()
            if k not in ["rul_p10_model", "rul_p50_model", "rul_p90_model", "classifier", "classifiers", "anomaly_detector"]
        }
        self._shap_explainer = None
        return self

    def _require_calibration_offset(self) -> float:
        """Returns the bundle-calibrated interval offset or raises if it was never derived."""
        offset = self.interval_calibration_offset
        if offset is None:
            raise ValueError(
                f"[{self.subset}] No calibration offset in the model bundle. "
                "Train or load a bundle produced by calibration on validation engines."
            )
        return float(offset)

    def calibrate_intervals(
        self,
        target_coverage: float = 0.80,
        test_df: Optional[pd.DataFrame] = None,
        rul_truth: Optional[np.ndarray] = None
    ) -> float:
        """Calibrates interval offset to achieve target coverage."""
        if test_df is not None and rul_truth is not None:
            last_indices = test_df.groupby("unit")["cycle"].idxmax()
            eval_rows = test_df.loc[last_indices].sort_values("unit")
            X_test = eval_rows[self.feature_cols].values
            p50 = np.maximum(0.0, self.rul_p50_model.predict(X_test))
            raw_p10 = self.rul_p10_model.predict(X_test)
            raw_p90 = self.rul_p90_model.predict(X_test)
            base_p10 = np.maximum(0.0, np.minimum(raw_p10, p50))
            base_p90 = np.maximum(p50, np.maximum(raw_p90, p50))
            residuals = np.maximum(base_p10 - rul_truth, rul_truth - base_p90)
            residuals = np.maximum(0.0, residuals)
            offset = float(np.quantile(residuals, target_coverage))
            self.interval_calibration_offset = round(offset, 2)
            self.conformal_q = self.interval_calibration_offset
            # This legacy helper calibrates a single global offset; drop stale bucket
            # metadata so predictions consistently use the new global q.
            self.conformal_buckets = None
        else:
            raise ValueError(
                "calibrate_intervals requires held-out test/validation frames to derive the offset; "
                "refusing to fall back to a typed default."
            )
        return self.interval_calibration_offset
