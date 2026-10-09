import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.common.utils import get_root_dir, load_yaml
from src.ingest.loader import CMAPSSLoader
from src.ingest.fault_injector import FaultInjector
from src.clean.cleaner import DataCleaner
from src.features.regime import RegimeNormalizer
from src.features.builder import FeatureBuilder

SENSOR_REGEX = re.compile(r"^s\d+$")

class DataPipeline:
    def __init__(self, subset: str = "FD001", inject_faults: bool = False):
        self.subset = subset
        self.inject_faults = inject_faults
        self.root_dir = get_root_dir()
        self.config = load_yaml("configs/pipeline.yaml")
        
        self.loader = CMAPSSLoader(subset=self.subset)
        self.fault_injector = FaultInjector(seed=42)
        self.cleaner = DataCleaner(self.config)
        self.normalizer = RegimeNormalizer(subset=self.subset)
        self.builder = FeatureBuilder(self.config)
        
        self.artifacts_dir = self.root_dir / "artifacts"
        self.processed_dir = self.root_dir / "data" / "processed" / self.subset
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.quality_report: Dict[str, Any] = {}

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Runs the end-to-end data pipeline:
        1. Ingest & Validate
        2. Optional Fault Injection (test harness, sensors only)
        3. Clean Stage 1: Dedupe, causal imputation, constant-sensor removal
        4. Regime Normalization: K-Means (k=6 for multi-regime, k=1 for single)
        5. Clean Stage 2: Outlier clipping on normalized sensors
        6. Feature Extraction: Rolling windows, slopes, EWMA, dev_base on norm columns (explicit loop, no groupby.apply)
        7. Health Index: Monotonic 0-1 PCA
        8. Targets: Piecewise RUL (cap=125), fails_within_30
        """
        print(f"[{self.subset}] Ingesting raw data...")
        raw_train, raw_test, rul_test = self.loader.load_raw_data()

        # 1. Validation
        train_val, val_report_train = self.loader.validate(raw_train, "train")
        test_val, val_report_test = self.loader.validate(raw_test, "test")

        if self.inject_faults:
            print(f"[{self.subset}] Injecting synthetic faults for robustness test...")
            train_val = self.fault_injector.inject_faults(train_val)

        # 2. Clean Stage 1: Causal imputation (no backfill), constant-sensor removal
        print(f"[{self.subset}] Cleaning with causal imputation and removing constant sensors...")
        train_clean, clean_report_train = self.cleaner.clean_impute(train_val, is_train=True)
        test_clean, clean_report_test = self.cleaner.clean_impute(test_val, is_train=False)

        kept_sensors = self.cleaner.kept_sensors
        print(f"[{self.subset}] Kept sensors ({len(kept_sensors)}): {kept_sensors}")

        # 3. Regime Normalization: performed BEFORE outlier clipping
        print(f"[{self.subset}] Fitting regime clustering (k={self.normalizer.k})...")
        self.normalizer.fit(train_clean, kept_sensors)
        train_norm = self.normalizer.transform(train_clean, kept_sensors)
        test_norm = self.normalizer.transform(test_clean, kept_sensors)

        # 4. Clean Stage 2: Outlier clipping & flatline detection on normalized sensors
        print(f"[{self.subset}] Clipping extreme outliers on regime-normalized sensors...")
        train_clipped, out_report_train = self.cleaner.clip_outliers_normalized(train_norm, kept_sensors)
        test_clipped, out_report_test = self.cleaner.clip_outliers_normalized(test_norm, kept_sensors)

        # Merge outlier audit into reports
        clean_report_train.update(out_report_train)
        clean_report_test.update(out_report_test)

        # 5. Feature Engineering per unit: EXPLICIT LOOP (zero groupby.apply to avoid pandas 2.2/3.0 KeyError)
        print(f"[{self.subset}] Building rolling features and trends via explicit unit loops...")
        train_feat_list = []
        for unit_id, group in train_clipped.groupby("unit", sort=False):
            train_feat_list.append(self.builder.extract_features_for_unit(group, kept_sensors))
        train_feats = pd.concat(train_feat_list, ignore_index=True)

        test_feat_list = []
        for unit_id, group in test_clipped.groupby("unit", sort=False):
            test_feat_list.append(self.builder.extract_features_for_unit(group, kept_sensors))
        test_feats = pd.concat(test_feat_list, ignore_index=True)

        # 6. Health Index: Monotonic 0-1 PCA
        print(f"[{self.subset}] Computing 0-1 monotonic Health Index...")
        self.builder.fit_health_index(train_clipped, kept_sensors)
        train_feats["health_index"] = self.builder.compute_health_index(train_feats, kept_sensors)
        test_feats["health_index"] = self.builder.compute_health_index(test_feats, kept_sensors)

        # 7. Target Creation on Train
        print(f"[{self.subset}] Creating RUL piecewise targets (cap={self.builder.rul_cap})...")
        train_final = self.builder.create_targets(train_feats)

        # 8. Test Ground-Truth RUL for every cycle
        test_final = test_feats.copy()
        max_test_cycles = test_final.groupby("unit")["cycle"].max().to_dict()
        rul_dict = {unit_id: rul_test[idx] for idx, unit_id in enumerate(sorted(test_final["unit"].unique()))}
        
        test_final["rul_raw"] = test_final.apply(
            lambda r: rul_dict[r["unit"]] + (max_test_cycles[r["unit"]] - r["cycle"]), axis=1
        )
        test_final["rul_capped"] = np.minimum(test_final["rul_raw"], self.builder.rul_cap)
        test_final["fails_within_30"] = (test_final["rul_raw"] <= 30).astype(int)

        # Save Processed Parquets
        train_parquet = self.processed_dir / "train.parquet"
        test_parquet = self.processed_dir / "test.parquet"
        train_final.to_parquet(train_parquet, index=False)
        test_final.to_parquet(test_parquet, index=False)

        # Save Artifacts
        self.normalizer.save(self.artifacts_dir / "scalers" / f"regime_normalizer_{self.subset}.joblib")
        self.builder.save(self.artifacts_dir / "scalers" / f"feature_builder_{self.subset}.joblib")

        self.quality_report = {
            "subset": self.subset,
            "train_validation": val_report_train,
            "test_validation": val_report_test,
            "train_cleaning": clean_report_train,
            "test_cleaning": clean_report_test,
            "kept_sensors": kept_sensors,
            "kept_sensors_count": len(kept_sensors),
            "regime_clusters": self.normalizer.k,
            "train_rows": len(train_final),
            "test_rows": len(test_final),
            "feature_count": train_final.shape[1]
        }

        report_file = self.artifacts_dir / "reports" / f"data_quality_{self.subset}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.quality_report, f, indent=2)

        print(f"[{self.subset}] Pipeline complete! Saved {len(train_final)} train rows, {len(test_final)} test rows, {train_final.shape[1]} features.")
        return train_final, test_final, rul_test, self.quality_report

if __name__ == "__main__":
    dp = DataPipeline(subset="FD001", inject_faults=False)
    dp.run()
