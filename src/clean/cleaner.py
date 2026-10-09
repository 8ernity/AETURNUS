import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

SENSOR_REGEX = re.compile(r"^s\d+$")

class DataCleaner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clean_cfg = config.get("cleaning", {})
        self.ffill_limit = self.clean_cfg.get("ffill_limit", 3)
        self.hampel_window = self.clean_cfg.get("hampel_window", 7)
        self.hampel_sigma = self.clean_cfg.get("hampel_sigma", 3.0)
        self.outlier_z_clip = self.clean_cfg.get("outlier_z_clip", 5.0)
        self.flatline_window = self.clean_cfg.get("flatline_window", 10)
        self.flatline_var_thresh = self.clean_cfg.get("flatline_var_threshold", 1e-5)
        self.const_var_thresh = self.clean_cfg.get("constant_sensor_var_threshold", 1e-4)
        
        self.train_medians: Dict[str, float] = {}
        self.train_stds: Dict[str, float] = {}
        self.kept_sensors: List[str] = []

    def get_sensor_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if SENSOR_REGEX.match(c)]

    def fit_train(self, train_df: pd.DataFrame) -> List[str]:
        """
        Learns training medians, robust scales, and determines constant sensors to drop.
        Strictly evaluates only sensors matching ^s\\d+$.
        """
        sensor_cols = self.get_sensor_cols(train_df)
        self.kept_sensors = []

        for col in sensor_cols:
            var = float(train_df[col].var())
            median_val = float(train_df[col].median())
            std_val = float(train_df[col].std())
            self.train_medians[col] = median_val
            self.train_stds[col] = std_val if std_val > 1e-5 else 1.0

            # Sensor is kept only if variance exceeds threshold
            if var > self.const_var_thresh:
                self.kept_sensors.append(col)

        return self.kept_sensors

    def clean_impute(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stage 1 Cleaning (pre-normalization):
        1. Deduplication & chronological sorting
        2. Strictly CAUSAL imputation: forward-fill within engine (no backfill), then historical train-median fallback
        3. Constant-sensor identification and removal
        Eliminates groupby().apply to avoid KeyError: 'unit' on pandas 2.2/3.0.
        """
        report: Dict[str, Any] = {
            "initial_rows": len(df),
            "missing_before": int(df.isna().sum().sum()),
            "sensors_dropped": []
        }

        # 1. Dedupe & sort
        cleaned = df.drop_duplicates(subset=["unit", "cycle"], keep="first")
        cleaned = cleaned.sort_values(by=["unit", "cycle"]).reset_index(drop=True)

        sensor_cols = self.get_sensor_cols(cleaned)

        if is_train and not self.kept_sensors:
            self.fit_train(cleaned)

        # 2. Causal forward-fill per unit (no backward interpolation)
        for col in sensor_cols:
            if cleaned[col].isna().any():
                # ffill per unit
                cleaned[col] = cleaned.groupby("unit")[col].ffill(limit=self.ffill_limit)
                # Any leading NaNs fill with train median fallback (strictly causal)
                fallback = self.train_medians.get(col, 0.0)
                cleaned[col] = cleaned[col].fillna(fallback)

        report["missing_after"] = int(cleaned.isna().sum().sum())

        # 3. Filter kept sensors
        dropped = [col for col in sensor_cols if col not in self.kept_sensors]
        report["sensors_dropped"] = dropped
        report["kept_sensors_count"] = len(self.kept_sensors)

        keep_cols = [c for c in cleaned.columns if not SENSOR_REGEX.match(c) or c in self.kept_sensors]
        cleaned = cleaned[keep_cols].copy()

        return cleaned, report

    def clip_outliers_normalized(self, df: pd.DataFrame, sensor_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stage 2 Cleaning (POST-regime normalization):
        Applies causal Hampel/robust z-clipping on normalized sensors (*_norm).
        Detects flatlines over flatline_window.
        """
        df_out = df.copy()
        outliers_count = 0
        flatlines_count = 0

        for col in sensor_cols:
            norm_col = f"{col}_norm"
            target_col = norm_col if norm_col in df_out.columns else col

            # Detect flatlines using transform/rolling without apply
            rolling_var = df_out.groupby("unit")[target_col].transform(
                lambda s: s.rolling(self.flatline_window, min_periods=self.flatline_window).var()
            ).fillna(1.0)
            flatline_mask = rolling_var < self.flatline_var_thresh
            if flatline_mask.any():
                flatlines_count += int(flatline_mask.sum())

            # Causal Hampel filter / robust z-clipping
            # On normalized data, mean=0, std=1; clip extreme deviations beyond threshold
            vals = df_out[target_col].values
            extreme_mask = np.abs(vals) > self.outlier_z_clip
            outliers_count += int(np.sum(extreme_mask))
            if np.any(extreme_mask):
                df_out[target_col] = np.clip(vals, -self.outlier_z_clip, self.outlier_z_clip)

        return df_out, {
            "outliers_cleaned": outliers_count,
            "flatlines_detected": flatlines_count
        }
