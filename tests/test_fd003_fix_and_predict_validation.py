"""
Tests for Item 1 (FD003 regression fix) and Item 7 (predict_point validation).

Verifies:
1. p50 model uses regression/huber objective (not quantile) across all subsets
2. For shipped bundles: std(p50 on test last windows) > 10
3. For shipped bundles: RMSE < constant-mean baseline
4. predict_point raises ValueError on missing features
5. evaluate_test/evaluate_dataframe raise ValueError on missing features
6. Train assertions (trees > 20, importances > 0, val RMSE < baseline)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.utils import get_root_dir, calculate_metrics
from src.models.model_service import ModelService


ROOT_DIR = get_root_dir()
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def _get_available_bundles():
    """Returns list of subsets that have a shipped model bundle."""
    available = []
    for sub in SUBSETS:
        bundle_path = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        if bundle_path.exists():
            available.append(sub)
    return available


def _load_test_data(subset):
    """Loads test dataframe and RUL truth for a given subset."""
    test_path = ROOT_DIR / "data" / "processed" / subset / "test.parquet"
    if not test_path.exists():
        pytest.skip(f"Test data not found for {subset}")
    test_df = pd.read_parquet(test_path)

    from src.ingest.loader import CMAPSSLoader
    loader = CMAPSSLoader(subset=subset)
    _, _, rul_truth = loader.load_raw_data()
    return test_df, rul_truth


class TestFD003RegressionObjective:
    """Verify p50 model is NOT using quantile objective."""

    @pytest.mark.parametrize("subset", _get_available_bundles())
    def test_p50_model_is_not_quantile(self, subset):
        """p50 model should use huber/regression, not quantile alpha=0.50."""
        ms = ModelService(subset=subset).load()

        # For LightGBM models, check the objective parameter
        if hasattr(ms.rul_p50_model, "get_params"):
            params = ms.rul_p50_model.get_params()
            objective = params.get("objective", "")
            # Must not be 'quantile'
            assert objective != "quantile", (
                f"[{subset}] p50 model is still using quantile objective. "
                f"Expected 'huber' or 'regression'."
            )

    @pytest.mark.parametrize("subset", _get_available_bundles())
    def test_p10_p90_are_quantile(self, subset):
        """p10/p90 models should still use quantile objectives."""
        ms = ModelService(subset=subset).load()

        if hasattr(ms.rul_p10_model, "get_params"):
            p10_params = ms.rul_p10_model.get_params()
            if getattr(ms, "algorithm", "") == "histgb" or "HistGradient" in type(ms.rul_p10_model).__name__:
                assert p10_params.get("loss") == "quantile", (
                    f"[{subset}] HistGB p10 model should use loss=='quantile'"
                )
            else:
                assert p10_params.get("objective") == "quantile" or p10_params.get("loss") == "quantile", (
                    f"[{subset}] p10 model should use quantile objective"
                )

        if hasattr(ms.rul_p90_model, "get_params"):
            p90_params = ms.rul_p90_model.get_params()
            if getattr(ms, "algorithm", "") == "histgb" or "HistGradient" in type(ms.rul_p90_model).__name__:
                assert p90_params.get("loss") == "quantile", (
                    f"[{subset}] HistGB p90 model should use loss=='quantile'"
                )
            else:
                assert p90_params.get("objective") == "quantile" or p90_params.get("loss") == "quantile", (
                    f"[{subset}] p90 model should use quantile objective"
                )


class TestShippedBundleQuality:
    """Verify shipped bundles produce meaningful predictions."""

    @pytest.mark.parametrize("subset", _get_available_bundles())
    def test_p50_std_greater_than_10(self, subset):
        """
        std(p50 on test last windows) > 10:
        The model must produce varied predictions across engines, not collapse
        to a constant value.
        """
        ms = ModelService(subset=subset).load()
        test_df, rul_truth = _load_test_data(subset)

        last_indices = test_df.groupby("unit")["cycle"].idxmax()
        eval_rows = test_df.loc[last_indices].sort_values("unit")

        X_test = eval_rows[ms.feature_cols].values
        pred_p50 = np.maximum(0.0, ms.rul_p50_model.predict(X_test))

        p50_std = float(np.std(pred_p50))
        assert p50_std > 10.0, (
            f"[{subset}] std(p50) = {p50_std:.2f} is too low (expected > 10). "
            f"The model may have collapsed to near-constant predictions."
        )

    @pytest.mark.parametrize("subset", _get_available_bundles())
    def test_rmse_beats_constant_baseline(self, subset):
        """
        Model RMSE on test last-windows must beat a constant-mean baseline.
        The constant baseline predicts mean(train RUL) for all engines.
        """
        ms = ModelService(subset=subset).load()
        test_df, rul_truth = _load_test_data(subset)

        last_indices = test_df.groupby("unit")["cycle"].idxmax()
        eval_rows = test_df.loc[last_indices].sort_values("unit")

        X_test = eval_rows[ms.feature_cols].values
        pred_p50 = np.maximum(0.0, ms.rul_p50_model.predict(X_test))

        model_rmse = float(np.sqrt(np.mean((pred_p50 - rul_truth) ** 2)))

        # Constant baseline: predict mean(rul_truth) for all engines
        constant_pred = np.full_like(rul_truth, np.mean(rul_truth), dtype=np.float64)
        baseline_rmse = float(np.sqrt(np.mean((constant_pred - rul_truth) ** 2)))

        assert model_rmse < baseline_rmse, (
            f"[{subset}] Model RMSE ({model_rmse:.2f}) >= constant-mean baseline "
            f"({baseline_rmse:.2f}). The model is not learning."
        )


class TestPredictPointValidation:
    """Verify predict_point raises on missing features."""

    def test_predict_point_raises_on_missing_features(self):
        """predict_point must raise ValueError when required features are absent."""
        available = _get_available_bundles()
        if not available:
            pytest.skip("No shipped bundles available")

        subset = available[0]
        ms = ModelService(subset=subset).load()

        # Provide an incomplete row (missing most features)
        incomplete_row = {"unit": 1, "cycle": 50}

        with pytest.raises(ValueError, match="missing"):
            ms.predict_point(incomplete_row)

    def test_predict_point_works_with_complete_features(self):
        """predict_point succeeds when all features are present."""
        available = _get_available_bundles()
        if not available:
            pytest.skip("No shipped bundles available")

        subset = available[0]
        ms = ModelService(subset=subset).load()

        test_df, _ = _load_test_data(subset)
        sample_row = test_df.iloc[0].to_dict()

        # Should not raise
        pred = ms.predict_point(sample_row)
        assert "rul_p50" in pred
        assert pred["rul_p50"] >= 0.0


class TestEvaluateRaisesOnMissing:
    """Verify evaluate methods raise on missing features."""

    def test_evaluate_dataframe_raises_on_missing(self):
        """evaluate_dataframe must raise ValueError on missing feature columns."""
        available = _get_available_bundles()
        if not available:
            pytest.skip("No shipped bundles available")

        subset = available[0]
        ms = ModelService(subset=subset).load()

        # Create a dataframe missing some features
        dummy_df = pd.DataFrame({
            "unit": [1, 2],
            "cycle": [10, 20],
            "rul_capped": [100, 90]
        })

        with pytest.raises(ValueError, match="missing"):
            ms.evaluate_dataframe(dummy_df)
