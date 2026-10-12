import math
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.common.utils import load_yaml, get_root_dir

class MaintenanceQueue:
    def __init__(self, config_path: Optional[str] = None):
        root_dir = get_root_dir()
        cfg_file = Path(config_path) if config_path else root_dir / "configs" / "queue.yaml"
        if cfg_file.exists():
            self.config = load_yaml(str(cfg_file))
        else:
            self.config = {
                "weights": {"w_rul": 0.40, "w_fail": 0.30, "w_anom": 0.15, "w_trend": 0.15},
                "normalization": {"max_urgency_rul": 60.0, "max_rul_drop": 30.0},
                "daily_service_capacity": 5,
                "disclaimer": "Configured starting weights for decision support, not empirical physical claims."
            }

        w = self.config.get("weights", {})
        self.w_rul = float(w.get("w_rul", 0.40))
        self.w_fail = float(w.get("w_fail", 0.30))
        self.w_anom = float(w.get("w_anom", 0.15))
        self.w_trend = float(w.get("w_trend", 0.15))

        norm = self.config.get("normalization", {})
        self.max_urgency_rul = float(norm.get("max_urgency_rul", 60.0))
        self.max_rul_drop = float(norm.get("max_rul_drop", 30.0))
        self.default_capacity = int(self.config.get("daily_service_capacity", 5))
        self.disclaimer = self.config.get("disclaimer", "Configured starting weights for decision support.")

    def compute_priority_score(self, item: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Computes composite maintenance priority score:
        S = w_rul * (1 - p10/60)+ + w_fail * P(fail<=30) + w_anom * anomaly_score + w_trend * (drop/30)+
        """
        cons_rul = float(item.get("conservative_rul", item.get("rul_p10", 60.0)))
        rul_term = max(0.0, 1.0 - (cons_rul / self.max_urgency_rul))

        fail_prob = float(item.get("fail_prob_30", item.get("fail_prob", 0.0)))
        fail_term = max(0.0, min(1.0, fail_prob))

        anom_score = float(item.get("anomaly_score", 0.0))
        anom_term = max(0.0, min(1.0, anom_score))

        rul_drop = float(item.get("rul_drop", 0.0))
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
            c_rul = float(item.get("conservative_rul", item.get("rul_p10", 999.0)))
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
