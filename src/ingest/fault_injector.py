import re
import numpy as np
import pandas as pd
from typing import Dict, Any

SENSOR_REGEX = re.compile(r"^s\d+$")

class FaultInjector:
    """
    Test harness for injecting synthetic sensor flaws:
    - 2-5% missing values
    - Gaussian noise
    - 0.5% spike outliers
    - flatlined sensor segments
    - duplicate rows & out-of-order cycles
    Strictly operates only on sensors matching ^s\\d+$ (never settings).
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def inject_faults(self, df: pd.DataFrame) -> pd.DataFrame:
        faulty_df = df.copy()
        # Strictly select columns matching ^s\d+$
        sensor_cols = [c for c in faulty_df.columns if SENSOR_REGEX.match(c)]

        # 1. Inject missing values (3%)
        for col in sensor_cols:
            mask = self.rng.rand(len(faulty_df)) < 0.03
            faulty_df.loc[mask, col] = np.nan

        # 2. Inject Gaussian noise (1% std relative noise)
        for col in sensor_cols:
            valid_vals = faulty_df[col].dropna()
            if len(valid_vals) > 0 and valid_vals.std() > 1e-4:
                noise = self.rng.normal(0, valid_vals.std() * 0.10, size=len(faulty_df))
                faulty_df[col] = faulty_df[col] + noise

        # 3. Inject spike outliers (0.5%)
        for col in sensor_cols[:5]:
            spike_mask = self.rng.rand(len(faulty_df)) < 0.005
            multiplier = self.rng.choice([3.0, -3.0], size=len(faulty_df))
            faulty_df.loc[spike_mask, col] = faulty_df.loc[spike_mask, col] * (1.0 + multiplier[spike_mask])

        # 4. Inject flatline segment on 1 engine and 1 sensor
        if "unit" in faulty_df.columns and len(sensor_cols) > 2:
            target_unit = faulty_df["unit"].unique()[0]
            unit_mask = faulty_df["unit"] == target_unit
            unit_indices = faulty_df[unit_mask].index[:15]
            if len(unit_indices) >= 10:
                faulty_df.loc[unit_indices, sensor_cols[1]] = 500.0

        # 5. Inject a few duplicate rows
        sample_dupes = faulty_df.sample(n=min(5, len(faulty_df)), random_state=self.seed)
        faulty_df = pd.concat([faulty_df, sample_dupes], ignore_index=True)

        return faulty_df
