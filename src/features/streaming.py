import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.common.utils import get_root_dir, load_yaml
from src.clean.cleaner import DataCleaner
from src.features.regime import RegimeNormalizer
from src.features.builder import FeatureBuilder
from src.models.model_service import ModelService
from src.agent.triage_agent import TriageAgent

class StreamingEnginePipeline:
    """
    Online per-engine streaming processor:
    Performs streaming through:
    raw cycle row -> clean -> causal features -> predict -> triage
    """
    def __init__(self, subset: str = "FD001", model_service: Optional[ModelService] = None, triage_agent: Optional[TriageAgent] = None):
        self.subset = subset
        self.root_dir = get_root_dir()
        self.config = load_yaml("configs/pipeline.yaml")
        self.cleaner = DataCleaner(self.config)

        norm_file = self.root_dir / "artifacts" / "scalers" / f"regime_normalizer_{subset}.joblib"
        if norm_file.exists():
            self.normalizer = RegimeNormalizer.load(norm_file)
        else:
            self.normalizer = RegimeNormalizer(subset=subset)

        builder_file = self.root_dir / "artifacts" / "scalers" / f"feature_builder_{subset}.joblib"
        if builder_file.exists():
            self.builder = FeatureBuilder.load(builder_file)
        else:
            self.builder = FeatureBuilder(self.config)

        self.model_service = model_service or ModelService(subset=subset).load()
        self.triage_agent = triage_agent
        self.engine_buffers: Dict[int, List[Dict[str, Any]]] = {}

        # Pre-fit cleaner from training data if available
        train_path = self.root_dir / "data" / "processed" / subset / "train.parquet"
        if train_path.exists():
            train_df = pd.read_parquet(train_path)
            raw_s = [c for c in train_df.columns if self.cleaner.get_sensor_cols(pd.DataFrame(columns=[c])) and not c.endswith("_norm") and not c.endswith("_roll_mean") and not c.endswith("_ewma")]
            if raw_s:
                self.cleaner.fit_train(train_df[raw_s + ["unit", "cycle"]])
                if not norm_file.exists():
                    self.normalizer.fit(train_df, self.cleaner.kept_sensors)

    def stream_cycle(self, raw_row: Dict[str, Any]) -> Dict[str, Any]:
        unit = int(raw_row.get("unit", 1))
        cycle = int(raw_row.get("cycle", 1))

        if unit not in self.engine_buffers:
            self.engine_buffers[unit] = []
        self.engine_buffers[unit].append(raw_row)

        # Keep rolling window of last 40 cycles for this engine
        if len(self.engine_buffers[unit]) > 40:
            self.engine_buffers[unit] = self.engine_buffers[unit][-40:]

        buffer_df = pd.DataFrame(self.engine_buffers[unit])

        # If cleaner hasn't learned kept sensors from training, fit on current buffer
        if not self.cleaner.kept_sensors:
            self.cleaner.fit_train(buffer_df)
            if self.normalizer.kmeans is None and self.normalizer.k > 1:
                self.normalizer.fit(buffer_df, self.cleaner.kept_sensors)

        # 1. Clean Stage 1: Causal imputation (no backfill)
        cleaned_df, _ = self.cleaner.clean_impute(buffer_df, is_train=False)

        # 2. Regime Normalization
        norm_df = self.normalizer.transform(cleaned_df, self.cleaner.kept_sensors)

        # 3. Clean Stage 2: Outlier clipping on normalized sensors
        clipped_df, _ = self.cleaner.clip_outliers_normalized(norm_df, self.cleaner.kept_sensors)

        # 4. Feature Extraction
        features_df = self.builder.extract_features_for_unit(clipped_df, self.cleaner.kept_sensors)
        if self.builder.pca is not None:
            features_df["health_index"] = self.builder.compute_health_index(features_df, self.cleaner.kept_sensors)
        else:
            features_df["health_index"] = 1.0

        curr_feat_row = features_df.iloc[-1].to_dict()

        # 5. Predict
        prediction = self.model_service.predict_point(curr_feat_row)

        # 6. Triage
        order = None
        trace = []
        if self.triage_agent:
            order, trace = self.triage_agent.evaluate_engine_cycle(prediction)

        return {
            "unit": unit,
            "cycle": cycle,
            "features": curr_feat_row,
            "prediction": prediction,
            "work_order": order,
            "trace": trace
        }
