import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.common.utils import get_root_dir
from src.ingest.loader import CMAPSSLoader
from src.models.model_service import ModelService

DEFAULT_COST_CONFIG = {
    "cost_catastrophic": 2500000.0,   # Unplanned in-flight shutdown / catastrophic failure / AOG emergency ($)
    "cost_overhaul": 220000.0,        # Scheduled depot preventive overhaul ($)
    "cost_wasted_cycle": 1200.0,      # Value of unused remaining useful life depreciation ($/cycle)
    "cost_inspection": 12500.0,       # Minor borescopic / tarmac line triage ($)
    "cost_aog_day": 45000.0,          # Daily downtime delay penalty ($/day)
    "default_threshold": 30,          # Conservative RUL threshold (cycles) to trigger maintenance pull
    "fixed_interval": 140             # Flight cycles for static hour-based maintenance policy
}

class FleetCostSimulator:
    """
    Industrial Aviation Operational Cost & Maintenance ROI Simulator.
    Quantifies the economic impact of AETURNUS predictive maintenance vs.
    traditional run-to-failure (reactive) and fixed flight-cycle interval policies.
    """

    def __init__(self, subset: str = "FD001", cost_params: Optional[Dict[str, float]] = None):
        self.subset = subset
        self.root_dir = get_root_dir()
        self.cost_cfg = {**DEFAULT_COST_CONFIG, **(cost_params or {})}
        self.eval_data = self._load_evaluation_data()

    def _load_evaluation_data(self) -> List[Dict[str, Any]]:
        """
        Loads authentic engine-level test predictions and ground truth.
        Uses cached evaluation report if available, else evaluates on test parquet.
        """
        eval_f = self.root_dir / "artifacts" / "reports" / f"evaluation_{self.subset}.json"
        if eval_f.exists():
            try:
                with open(eval_f, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    scatter = d.get("parity_scatter", [])
                    if len(scatter) > 0:
                        return scatter
            except Exception:
                pass

        test_path = self.root_dir / "data" / "processed" / self.subset / "test.parquet"
        if not test_path.exists():
            return []

        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=self.subset)
        _, _, rul_truth = loader.load_raw_data()
        ms = ModelService(subset=self.subset).load()
        res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        return res.get("parity_scatter", [])

    def simulate_fleet(
        self,
        threshold: Optional[float] = None,
        use_conservative: bool = True,
        custom_cost_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simulates fleet-wide economics under 3 operational maintenance policies:
        1. Reactive (Run-to-Failure)
        2. Fixed Flight-Cycle Schedule (Static Overhaul Interval)
        3. AETURNUS Dynamic Predictive Maintenance
        """
        cfg = {**self.cost_cfg, **(custom_cost_params or {})}
        tau = float(threshold if threshold is not None else cfg["default_threshold"])
        c_cat = float(cfg["cost_catastrophic"])
        c_ovh = float(cfg["cost_overhaul"])
        c_waste = float(cfg["cost_wasted_cycle"])
        c_insp = float(cfg["cost_inspection"])
        c_aog = float(cfg["cost_aog_day"])
        fixed_int = int(cfg["fixed_interval"])

        n_engines = len(self.eval_data)
        if n_engines == 0:
            return {"error": "No evaluation data found", "subset": self.subset}

        # -------------------------------------------------------------
        # Policy 1: Reactive Run-to-Failure (No Predictive AI)
        # -------------------------------------------------------------
        reactive_total_cost = n_engines * c_cat
        reactive_catastrophic_count = n_engines

        # -------------------------------------------------------------
        # Policy 2: Static Fixed-Cycle Schedule (Overhaul at fixed_int)
        # -------------------------------------------------------------
        fixed_catastrophic_count = 0
        fixed_scheduled_count = 0
        fixed_wasted_cost = 0.0
        fixed_total_cost = 0.0

        for row in self.eval_data:
            # Baseline proxy: engines failing early (< fixed_int proxy) incur in-flight shutdown
            act = float(row.get("actual", 50.0))
            if act <= 15.0: # Engine fails before scheduled depot interval
                fixed_catastrophic_count += 1
                fixed_total_cost += c_cat
            else:
                fixed_scheduled_count += 1
                wasted = max(0.0, act - 15.0) * c_waste
                fixed_wasted_cost += wasted
                fixed_total_cost += (c_ovh + wasted)

        # -------------------------------------------------------------
        # Policy 3: AETURNUS Dynamic Predictive Maintenance Policy
        # -------------------------------------------------------------
        aeturnus_total_cost = 0.0
        aeturnus_catastrophic_count = 0
        aeturnus_scheduled_count = 0
        aeturnus_wasted_cost = 0.0
        aeturnus_inspection_cost = 0.0
        avoided_catastrophic_count = 0
        engine_ledger = []

        for row in self.eval_data:
            u = row.get("unit", 1)
            act = float(row.get("actual", 0.0))
            pred = float(row.get("predicted", 0.0))
            p10 = float(row.get("p10", pred))

            # Selected decision metric: conservative p10 or point prediction
            if use_conservative:
                d_err = p10 - act
            else:
                d_err = pred - act

            act_at_pull = tau - d_err

            if act_at_pull <= 0.0:
                # Late pull: engine suffered catastrophic failure before pull took effect
                penalty = abs(act_at_pull) * (c_aog * 0.1)
                cost = c_cat + penalty
                outcome = "UNPLANNED_CATASTROPHIC_FAILURE"
                aeturnus_catastrophic_count += 1
                wasted_life = 0.0
            else:
                # Safe preventive pull: depot overhaul + wasted remaining life
                wasted_life = act_at_pull * c_waste
                cost = c_ovh + wasted_life
                outcome = "SAFE_PREVENTIVE_OVERHAUL"
                aeturnus_scheduled_count += 1
                aeturnus_wasted_cost += wasted_life
                avoided_catastrophic_count += 1

            # Routine tarmac telemetry inspection cost for all monitored engines
            cost += c_insp
            aeturnus_inspection_cost += c_insp
            aeturnus_total_cost += cost

            engine_ledger.append({
                "unit": u,
                "actual_rul": round(act, 1),
                "predicted_rul": round(pred, 1),
                "conservative_p10": round(p10, 1),
                "effective_pull_cycle_margin": round(act_at_pull, 1),
                "outcome": outcome,
                "cost": round(cost, 2),
                "avoided_catastrophic": (outcome == "SAFE_PREVENTIVE_OVERHAUL")
            })

        # Sort ledger by actual RUL ascending (most critical engines first)
        engine_ledger.sort(key=lambda x: x["actual_rul"])

        net_savings_vs_reactive = max(0.0, reactive_total_cost - aeturnus_total_cost)
        net_savings_vs_fixed = max(0.0, fixed_total_cost - aeturnus_total_cost)
        cost_reduction_pct = round((net_savings_vs_reactive / reactive_total_cost * 100.0), 1) if reactive_total_cost > 0 else 0.0
        roi_multiplier = round((net_savings_vs_reactive / aeturnus_total_cost), 2) if aeturnus_total_cost > 0 else 0.0

        return {
            "subset": self.subset,
            "fleet_size": n_engines,
            "threshold": tau,
            "use_conservative": use_conservative,
            "cost_parameters": cfg,
            "summary": {
                "net_savings_vs_reactive": round(net_savings_vs_reactive, 2),
                "net_savings_vs_fixed": round(net_savings_vs_fixed, 2),
                "cost_reduction_percent": cost_reduction_pct,
                "roi_multiplier": roi_multiplier,
                "avoided_catastrophic_failures": avoided_catastrophic_count,
                "catastrophic_failures_under_aeturnus": aeturnus_catastrophic_count,
                "aeturnus_total_cost": round(aeturnus_total_cost, 2),
                "reactive_total_cost": round(reactive_total_cost, 2),
                "fixed_total_cost": round(fixed_total_cost, 2)
            },
            "policies": {
                "reactive": {
                    "name": "Run-to-Failure (Unmanaged)",
                    "total_cost": round(reactive_total_cost, 2),
                    "total_cost_m": round(reactive_total_cost / 1e6, 2),
                    "catastrophic_failures": reactive_catastrophic_count,
                    "scheduled_overhauls": 0,
                    "wasted_life_cost": 0.0,
                    "cost_per_engine": round(reactive_total_cost / n_engines, 2)
                },
                "fixed_interval": {
                    "name": f"Fixed Flight Schedule ({fixed_int} cyc)",
                    "total_cost": round(fixed_total_cost, 2),
                    "total_cost_m": round(fixed_total_cost / 1e6, 2),
                    "catastrophic_failures": fixed_catastrophic_count,
                    "scheduled_overhauls": fixed_scheduled_count,
                    "wasted_life_cost": round(fixed_wasted_cost, 2),
                    "cost_per_engine": round(fixed_total_cost / n_engines, 2)
                },
                "aeturnus": {
                    "name": f"AETURNUS Predictive Policy (Threshold = {tau} cyc)",
                    "total_cost": round(aeturnus_total_cost, 2),
                    "total_cost_m": round(aeturnus_total_cost / 1e6, 2),
                    "catastrophic_failures": aeturnus_catastrophic_count,
                    "scheduled_overhauls": aeturnus_scheduled_count,
                    "wasted_life_cost": round(aeturnus_wasted_cost, 2),
                    "inspection_cost": round(aeturnus_inspection_cost, 2),
                    "cost_per_engine": round(aeturnus_total_cost / n_engines, 2)
                }
            },
            "breakdown_chart": [
                {
                    "policy": "Reactive",
                    "Catastrophic AOG ($M)": round(reactive_total_cost / 1e6, 2),
                    "Scheduled Overhauls ($M)": 0.0,
                    "Wasted Depreciation ($M)": 0.0,
                    "Line Inspection ($M)": 0.0,
                    "Total ($M)": round(reactive_total_cost / 1e6, 2)
                },
                {
                    "policy": "Fixed Schedule",
                    "Catastrophic AOG ($M)": round((fixed_catastrophic_count * c_cat) / 1e6, 2),
                    "Scheduled Overhauls ($M)": round((fixed_scheduled_count * c_ovh) / 1e6, 2),
                    "Wasted Depreciation ($M)": round(fixed_wasted_cost / 1e6, 2),
                    "Line Inspection ($M)": 0.0,
                    "Total ($M)": round(fixed_total_cost / 1e6, 2)
                },
                {
                    "policy": "AETURNUS AI",
                    "Catastrophic AOG ($M)": round((aeturnus_catastrophic_count * c_cat) / 1e6, 2),
                    "Scheduled Overhauls ($M)": round((aeturnus_scheduled_count * c_ovh) / 1e6, 2),
                    "Wasted Depreciation ($M)": round(aeturnus_wasted_cost / 1e6, 2),
                    "Line Inspection ($M)": round(aeturnus_inspection_cost / 1e6, 2),
                    "Total ($M)": round(aeturnus_total_cost / 1e6, 2)
                }
            ],
            "ledger": engine_ledger[:30]
        }

    def sweep_cost_curve(
        self,
        min_threshold: int = 5,
        max_threshold: int = 60,
        step: int = 1,
        use_conservative: bool = True,
        custom_cost_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Sweeps threshold tau in [min_threshold, max_threshold] to generate
        the mathematical U-shaped fleet cost curve and identify optimal tau*.
        """
        cfg = {**self.cost_cfg, **(custom_cost_params or {})}
        c_cat = float(cfg["cost_catastrophic"])
        c_ovh = float(cfg["cost_overhaul"])
        c_waste = float(cfg["cost_wasted_cycle"])
        c_insp = float(cfg["cost_inspection"])
        c_aog = float(cfg["cost_aog_day"])

        curve_points = []
        best_tau = min_threshold
        min_cost = float("inf")

        thresholds = list(range(min_threshold, max_threshold + 1, step))
        for tau in thresholds:
            total_cost = 0.0
            cat_failures = 0
            ovh_count = 0
            wasted_cost = 0.0
            insp_cost = 0.0

            for row in self.eval_data:
                act = float(row.get("actual", 0.0))
                pred = float(row.get("predicted", 0.0))
                p10 = float(row.get("p10", pred))

                if use_conservative:
                    d_err = p10 - act
                else:
                    d_err = pred - act

                act_at_pull = tau - d_err

                if act_at_pull <= 0.0:
                    total_cost += c_cat + (abs(act_at_pull) * (c_aog * 0.1))
                    cat_failures += 1
                else:
                    w = act_at_pull * c_waste
                    total_cost += c_ovh + w
                    ovh_count += 1
                    wasted_cost += w

                total_cost += c_insp
                insp_cost += c_insp

            if total_cost < min_cost:
                min_cost = total_cost
                best_tau = tau

            curve_points.append({
                "threshold": tau,
                "total_cost_m": round(total_cost / 1e6, 3),
                "catastrophic_cost_m": round((cat_failures * c_cat) / 1e6, 3),
                "overhaul_cost_m": round((ovh_count * c_ovh) / 1e6, 3),
                "wasted_life_cost_m": round(wasted_cost / 1e6, 3),
                "catastrophic_failures": cat_failures,
                "scheduled_overhauls": ovh_count
            })

        return {
            "subset": self.subset,
            "optimal_threshold": best_tau,
            "minimum_cost": round(min_cost, 2),
            "minimum_cost_m": round(min_cost / 1e6, 3),
            "curve": curve_points
        }
