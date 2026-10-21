import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.common.utils import get_root_dir, load_yaml

class CMAPSSLoader:
    def __init__(self, subset: str = "FD001", data_dir: Optional[Path] = None):
        self.subset = subset
        self.root_dir = get_root_dir()
        self.config = load_yaml("configs/pipeline.yaml")
        self.data_dir = data_dir or (self.root_dir / "data" / "raw")
        self.columns = self.config["dataset"]["columns"]

    @property
    def operating_conditions(self) -> str:
        if self.subset in ("FD001", "FD003"):
            return "1 (Sea Level)"
        return "6 (Full Envelope)"

    @property
    def fault_modes(self) -> str:
        if self.subset in ("FD001", "FD002"):
            return "1 (HPC Degradation)"
        return "2 (HPC + Fan Degradation)"

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Loads raw train, test, and RUL truth arrays for the specified subset.
        """
        train_file = self.data_dir / f"train_{self.subset}.txt"
        test_file = self.data_dir / f"test_{self.subset}.txt"
        rul_file = self.data_dir / f"RUL_{self.subset}.txt"

        if not train_file.exists() or not test_file.exists() or not rul_file.exists():
            raise FileNotFoundError(f"Missing raw data files for subset {self.subset} in {self.data_dir}")

        train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=self.columns)
        test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=self.columns)
        rul_arr = pd.read_csv(rul_file, sep=r"\s+", header=None).values.flatten()

        return train_df, test_df, rul_arr

    def validate(self, df: pd.DataFrame, name: str = "train") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates schema, checks duplicates, ensures strictly increasing cycles.
        Quarantines invalid rows instead of silently discarding.
        """
        initial_rows = len(df)
        report = {
            "dataset": name,
            "subset": self.subset,
            "initial_rows": initial_rows,
            "duplicate_rows": 0,
            "out_of_order_cycles": 0,
            "missing_values_count": int(df.isna().sum().sum()),
            "quarantined_rows": 0,
            "valid_rows": 0
        }

        # 1. Deduplicate (unit, cycle)
        duplicates = df.duplicated(subset=["unit", "cycle"], keep="first")
        num_dupes = int(duplicates.sum())
        report["duplicate_rows"] = num_dupes

        clean_df = df[~duplicates].copy()

        # 2. Sort by unit, cycle
        clean_df = clean_df.sort_values(by=["unit", "cycle"]).reset_index(drop=True)

        # 3. Check for strictly increasing cycles
        cycle_diff = clean_df.groupby("unit")["cycle"].diff().fillna(1)
        out_of_order = cycle_diff <= 0
        num_ooo = int(out_of_order.sum())
        report["out_of_order_cycles"] = num_ooo

        if num_ooo > 0:
            clean_df = clean_df[~out_of_order].reset_index(drop=True)

        report["valid_rows"] = len(clean_df)
        report["quarantined_rows"] = initial_rows - len(clean_df)

        return clean_df, report
