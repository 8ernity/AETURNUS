import math
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.common.utils import load_yaml, get_root_dir

class MaintenanceQueue:
    def __init__(self, config_path: Optional[str] = None):
        root_dir = get_root_dir()
        cfg_file = Path(config_path) if config_path else root_dir / "configs" / "queue.yaml"
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"Maintenance queue policy config not found: {cfg_file}. "
                "Weights and normalization limits must come from config, not code defaults."
            )
        self.config = load_yaml(str(cfg_file))

        w = self.config["weights"]
        self.w_rul = float(w["w_rul"])
        self.w_fail = float(w["w_fail"])
        self.w_anom = float(w["w_anom"])
        self.w_trend = float(w["w_trend"])

        norm = self.config["normalization"]
        self.max_urgency_rul = float(norm["max_urgency_rul"])
        self.max_rul_drop = float(norm["max_rul_drop"])
        self.default_capacity = int(self.config["daily_service_capacity"])
        self.disclaimer = self.config["disclaimer"]

    def compute_priority_score(self, item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Computes composite maintenance priority score:
        S = w_rul * (1 - p10/60)+ + w_fail * P(fail<=30) + w_anom * anomaly_score + w_trend * (drop/30)+
        """
        if "conservative_rul" in item:
            cons_rul = float(item["conservative_rul"])
        elif "rul_p10" in item:
            cons_rul = float(item["rul_p10"])
        else:
            raise ValueError("MaintenanceQueue items require conservative_rul or rul_p10")
        rul_term = max(0.0, 1.0 - (cons_rul / self.max_urgency_rul))

        if "fail_prob_30" in item:
            fail_prob = float(item["fail_prob_30"])
        elif "fail_prob" in item:
            fail_prob = float(item["fail_prob"])
        else:
            raise ValueError("MaintenanceQueue items require fail_prob_30")
        fail_term = max(0.0, min(1.0, fail_prob))

        anom_score = float(item["anomaly_score"])
        anom_term = max(0.0, min(1.0, anom_score))

        # Missing history is represented as null, not a synthetic zero trend.
        rul_drop_value = item.get("rul_drop")
        if rul_drop_value is None:
            trend_term = 0.0
        else:
            rul_drop = float(rul_drop_value)
            trend_term = max(0.0, min(1.0, rul_drop / self.max_rul_drop))

        score = (
            self.w_rul * rul_term +
            self.w_fail * fail_term +
            self.w_anom * anom_term +
            self.w_trend * trend_term
        )

        components = {
            "rul_component": round(float(self.w_rul * rul_term), 4),
            "fail_component": round(float(self.w_fail * fail_term), 4),
            "anomaly_component": round(float(self.w_anom * anom_term), 4),
            "trend_component": round(float(self.w_trend * trend_term), 4),
        }

        return round(float(score), 4), components

    def rank_items(
        self,
        items: List[Dict[str, Any]],
        capacity: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate maintenance items descending by score.
        Ties are broken by lower conservative_rul (p10).
        Flags top K items as actionable_today=True.
        """
        cap = capacity if capacity is not None else self.default_capacity
        scored_items = []

        for item in items:
            score, comps = self.compute_priority_score(item)
            if "conservative_rul" in item:
                c_rul = float(item["conservative_rul"])
            else:
                c_rul = float(item["rul_p10"])
            entry = dict(item)
            entry["priority_score"] = score
            entry["score_components"] = comps
            scored_items.append((score, c_rul, entry))

        # Sort descending by score, ascending by conservative RUL (ties broken by lowest RUL)
        scored_items.sort(key=lambda x: (-x[0], x[1]))

        ranked_queue = []
        for idx, (_, _, entry) in enumerate(scored_items, start=1):
            is_actionable = idx <= cap
            entry["rank"] = idx
            entry["actionable_today"] = is_actionable
            entry["backlog"] = not is_actionable
            ranked_queue.append(entry)

        return ranked_queue
