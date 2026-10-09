import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

SENSOR_REGEX = re.compile(r"^s\d+$")

class FeatureBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        feat_cfg = config.get("features", {})
        self.rolling_windows = feat_cfg.get("rolling_windows", [5, 10, 30])
        self.baseline_cycles = feat_cfg.get("healthy_baseline_cycles", 20)
        self.alpha = feat_cfg.get("smoothing_alpha", 0.2)
        self.rul_cap = feat_cfg.get("rul_cap", 125)
        
        self.pca: Optional[PCA] = None
        self.pca_scaler_min: float = 0.0
        self.pca_scaler_max: float = 1.0

    @staticmethod
    def _compute_fast_slope(series: pd.Series, window: int) -> pd.Series:
        """
        Fast vectorized rolling linear regression slope calculation via convolution.
        y = a*x + b -> slope = cov(x, y) / var(x)
        """
        x = np.arange(window)
        x_mean = (window - 1) / 2.0
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return pd.Series(0.0, index=series.index)
        
        weights = (x - x_mean) / denom
        kernel = weights[::-1]
        
        vals = series.values
        padded = np.pad(vals, (window - 1, 0), mode='edge')
        slopes = np.convolve(padded, kernel, mode='valid')
        return pd.Series(slopes, index=series.index)

    def extract_features_for_unit(self, unit_df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """
        Extracts causal EWMA, rolling statistics, slopes, baseline deviations, and age.
        Computes *_dev_base strictly on *_norm columns.
        """
        df = unit_df.sort_values("cycle").copy()
        new_cols: Dict[str, Any] = {}

        # Healthy baseline per sensor (mean of first N cycles on *_norm columns)
        early_data = df[df["cycle"] <= self.baseline_cycles]
        
        for s in sensor_cols:
            norm_col = f"{s}_norm"
            base_s = norm_col if norm_col in df.columns else s
            base_series = df[base_s]

            # Baseline computed strictly on normalized sensor column
            base_val = early_data[base_s].mean() if len(early_data) > 0 else df[base_s].mean()
            
            # Causal EWMA & Baseline deviation strictly on *_norm columns
            new_cols[f"{s}_ewma"] = base_series.ewm(alpha=self.alpha, adjust=False).mean()
            new_cols[f"{s}_dev_base"] = base_series - base_val

            # Rolling stats on normalized series
            for w in self.rolling_windows:
                rolled = base_series.rolling(w, min_periods=1)
                new_cols[f"{s}_roll_mean_{w}"] = rolled.mean()
                new_cols[f"{s}_roll_std_{w}"] = rolled.std().fillna(0.0)
                new_cols[f"{s}_roll_min_{w}"] = rolled.min()
                new_cols[f"{s}_roll_max_{w}"] = rolled.max()
                new_cols[f"{s}_roll_slope_{w}"] = self._compute_fast_slope(base_series, w)

        new_cols["cycle_norm"] = df["cycle"] / 200.0
        
        feat_df = pd.DataFrame(new_cols, index=df.index)
        result = pd.concat([df, feat_df], axis=1)
        return result

    def fit_health_index(self, train_df: pd.DataFrame, sensor_cols: List[str]):
        """
        Fits PCA on normalized sensors to construct a continuous 0-1 Health Index (HI).
        """
        norm_cols = [f"{s}_norm" if f"{s}_norm" in train_df.columns else s for s in sensor_cols]
        self.pca = PCA(n_components=1, random_state=42)
        pca_proj = self.pca.fit_transform(train_df[norm_cols]).flatten()
        
        # Align orientation: early cycles should have higher health than late cycles
        early_mask = train_df["cycle"] <= 20
        late_mask = train_df["cycle"] > 100
        if late_mask.any() and early_mask.any():
            if pca_proj[early_mask].mean() < pca_proj[late_mask].mean():
                self.pca.components_ = -self.pca.components_
                pca_proj = -pca_proj

        self.pca_scaler_min = float(pca_proj.min())
        self.pca_scaler_max = float(pca_proj.max())

    def compute_health_index(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.Series:
        """
        Projects onto PCA component, scales between 0.0 and 1.0, and applies monotonic smoothing.
        Uses explicit loops per unit (no groupby.apply).
        """
        if self.pca is None:
            self.fit_health_index(df, sensor_cols)

        norm_cols = [f"{s}_norm" if f"{s}_norm" in df.columns else s for s in sensor_cols]
        raw_proj = self.pca.transform(df[norm_cols]).flatten()
        
        denom = (self.pca_scaler_max - self.pca_scaler_min) if (self.pca_scaler_max - self.pca_scaler_min) > 1e-5 else 1.0
        hi_scaled = (raw_proj - self.pca_scaler_min) / denom
        hi_scaled = np.clip(hi_scaled, 0.0, 1.0)
        
        df_temp = pd.DataFrame({
            "unit": df["unit"].values,
            "hi_raw": hi_scaled
        }, index=df.index)
        
        # Explicit loop over units to avoid any groupby().apply() issues
        smoothed_hi = pd.Series(index=df.index, dtype=np.float64)
        for unit_id, group in df_temp.groupby("unit", sort=False):
            cummin = group["hi_raw"].cummin()
            smoothed = cummin.ewm(alpha=0.3, adjust=False).mean()
            smoothed_hi.loc[group.index] = smoothed

        return smoothed_hi

    def create_targets(self, train_df: pd.DataFrame) -> pd.DataFrame:
        df = train_df.copy()
        max_cycles = df.groupby("unit")["cycle"].transform("max")
        raw_rul = max_cycles - df["cycle"]
        df["rul_raw"] = raw_rul
        df["rul_capped"] = np.minimum(raw_rul, self.rul_cap)
        df["fails_within_30"] = (raw_rul <= 30).astype(int)
        return df

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "FeatureBuilder":
        return joblib.load(file_path)
