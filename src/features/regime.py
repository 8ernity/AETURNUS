import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Tuple, Optional

class RegimeNormalizer:
    def __init__(self, subset: str, n_clusters: Optional[int] = None):
        self.subset = subset
        # PRD DE-6: k=6 for FD002/FD004, k=1 otherwise
        if n_clusters is not None:
            self.k = n_clusters
        else:
            self.k = 6 if subset in ["FD002", "FD004"] else 1
            
        self.kmeans: Optional[KMeans] = None
        self.setting_cols = ["setting_1", "setting_2", "setting_3"]
        self.regime_stats: Dict[int, Dict[str, Tuple[float, float]]] = {} # regime_id -> {sensor: (mean, std)}

    def fit(self, train_df: pd.DataFrame, sensor_cols: List[str]):
        """
        Fits K-Means on operating settings and computes per-regime sensor means and standard deviations.
        """
        if self.k > 1:
            self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
            regimes = self.kmeans.fit_predict(train_df[self.setting_cols])
        else:
            self.kmeans = None
            regimes = np.zeros(len(train_df), dtype=int)

        train_copy = train_df.copy()
        train_copy["regime"] = regimes

        self.regime_stats = {}
        for r in range(self.k):
            self.regime_stats[r] = {}
            r_data = train_copy[train_copy["regime"] == r]
            for s in sensor_cols:
                if len(r_data) > 0:
                    mean_val = float(r_data[s].mean())
                    std_val = float(r_data[s].std())
                    std_val = std_val if std_val > 1e-4 else 1.0
                else:
                    mean_val = float(train_df[s].mean())
                    std_val = float(train_df[s].std())
                    std_val = std_val if std_val > 1e-4 else 1.0
                self.regime_stats[r][s] = (mean_val, std_val)

    def transform(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """
        Assigns regime cluster and normalizes sensors according to their regime baselines.
        """
        df_out = df.copy()
        if self.k > 1 and self.kmeans is not None:
            regimes = self.kmeans.predict(df[self.setting_cols])
        else:
            regimes = np.zeros(len(df), dtype=int)

        df_out["regime"] = regimes

        for s in sensor_cols:
            norm_col_name = f"{s}_norm"
            norm_vals = np.zeros(len(df_out), dtype=np.float32)
            for r in range(self.k):
                mask = regimes == r
                if np.any(mask):
                    mean_val, std_val = self.regime_stats[r].get(s, (0.0, 1.0))
                    norm_vals[mask] = (df_out.loc[mask, s].values - mean_val) / std_val
            df_out[norm_col_name] = norm_vals

        return df_out

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "RegimeNormalizer":
        return joblib.load(file_path)
