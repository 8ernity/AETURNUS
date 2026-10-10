import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from src.common.utils import get_root_dir, load_yaml
from src.clean.cleaner import DataCleaner
from src.features.regime import RegimeNormalizer
from src.features.builder import FeatureBuilder
from src.models.model_service import ModelService
from src.agent.triage_agent import TriageAgent

@dataclass
class EngineStreamState:
    unit: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    baselines: Dict[str, float] = field(default_factory=dict)
    ewma: Dict[str, float] = field(default_factory=dict)
    rolling_windows: Dict[int, Dict[str, float]] = field(default_factory=dict)
    last_cycle: int = 0

class StreamingEnginePipeline:
    """
    Online per-engine streaming processor that maintains per-engine running state
    (baselines, EWMA, rolling windows over full history):
    raw cycle row -> clean -> causal features -> predict -> triage
    """
    def __init__(
        self,
        subset: str = "FD001",
        model_service: Optional[ModelService] = None,
        triage_agent: Optional[TriageAgent] = None
    ):
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

        # Persistent per-engine running state
        self.engine_states: Dict[int, EngineStreamState] = {}
        # Backward compatibility alias
        self.engine_buffers: Dict[int, List[Dict[str, Any]]] = {}

        # Pre-fit cleaner from raw training data or processed data
        self._init_cleaner()

    def _init_cleaner(self):
        try:
            from src.ingest.loader import CMAPSSLoader
            raw_train, _, _ = CMAPSSLoader(self.subset).load_raw_data()
            self.cleaner.fit_train(raw_train)
        except Exception:
            train_path = self.root_dir / "data" / "processed" / self.subset / "train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                raw_s = [c for c in train_df.columns if self.cleaner.get_sensor_cols(pd.DataFrame(columns=[c])) and "_" not in c]
                if raw_s:
                    self.cleaner.fit_train(train_df[raw_s + ["unit", "cycle"]])

    def get_or_create_engine_state(self, unit: int) -> EngineStreamState:
        if unit not in self.engine_states:
            self.engine_states[unit] = EngineStreamState(unit=unit)
            self.engine_buffers[unit] = self.engine_states[unit].history
        return self.engine_states[unit]

    def reset_engine(self, unit: int):
        """Resets running state for a specific engine."""
        if unit in self.engine_states:
            del self.engine_states[unit]
        if unit in self.engine_buffers:
            del self.engine_buffers[unit]

    def reset_all(self):
        """Resets all engine running states."""
        self.engine_states.clear()
        self.engine_buffers.clear()

    def stream_cycle(self, raw_row: Dict[str, Any]) -> Dict[str, Any]:
        unit = int(raw_row.get("unit", 1))
        cycle = int(raw_row.get("cycle", 1))

        state = self.get_or_create_engine_state(unit)
        state.history.append(raw_row)
        state.last_cycle = cycle

        # Full chronological history for this engine
        buffer_df = pd.DataFrame(state.history)

        # If cleaner hasn't learned kept sensors yet, fit on buffer
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

        # 4. Feature Extraction over full history (retaining early healthy baseline & EWMA)
        features_df = self.builder.extract_features_for_unit(clipped_df, self.cleaner.kept_sensors)
        if self.builder.pca is not None:
            features_df["health_index"] = self.builder.compute_health_index(features_df, self.cleaner.kept_sensors)
        else:
            features_df["health_index"] = 1.0

        curr_feat_row = features_df.iloc[-1].to_dict()

        # Update per-engine running state attributes
        early_data = clipped_df[clipped_df["cycle"] <= self.builder.baseline_cycles]
        for s in self.cleaner.kept_sensors:
            norm_col = f"{s}_norm"
            base_col = norm_col if norm_col in clipped_df.columns else s
            if len(early_data) > 0:
                state.baselines[s] = float(early_data[base_col].mean())
            else:
                state.baselines[s] = float(clipped_df[base_col].mean())

            if f"{s}_ewma" in curr_feat_row:
                state.ewma[s] = float(curr_feat_row[f"{s}_ewma"])

        for w in self.builder.rolling_windows:
            state.rolling_windows[w] = {
                s: float(curr_feat_row.get(f"{s}_roll_mean_{w}", 0.0))
                for s in self.cleaner.kept_sensors
            }

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
            "trace": trace,
            "state": {
                "baselines": state.baselines,
                "ewma": state.ewma,
                "history_length": len(state.history)
            }
        }
