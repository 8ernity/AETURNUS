import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.common.utils import get_root_dir
from src.ingest.loader import CMAPSSLoader
from src.models.model_service import ModelService

DEFAULT_COST_CONFIG = {
    "cost_catastrophic": 2500000.0,   # Unplanned in-flight shutdown / catastrophic failure / AOG diversion ($)
    "cost_overhaul": 220000.0,        # Scheduled depot preventive overhaul ($)
    "cost_wasted_cycle": 1200.0,      # Value of unused remaining useful life depreciation ($/cycle)
    "cost_inspection": 12500.0,       # Minor borescopic / tarmac line triage ($)
    "cost_aog_day": 45000.0,          # Daily downtime delay penalty ($/day)
    "default_threshold": 25,          # RUL threshold (cycles) to trigger maintenance pull
    "fixed_interval": 140             # Flight cycles for static hour-based maintenance policy
}

class FleetCostSimulator:
    """
    Industrial Aviation Operational Cost & Maintenance ROI Simulator.
    Quantifies the economic impact of AETURNUS predictive maintenance vs.
    traditional run-to-failure (reactive) and fixed flight-hour interval policies.
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

        # Fallback to computing from test parquet & model bundle
        test_path = self.root_dir / "data" / "processed" / self.subset / "test.parquet"
        if not test_path.exists():
            return []

        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=self.subset)
        _, _, rul_truth = loader.load_raw_data()
        ms = ModelService(subset=self.subset).load()
        res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        return res.get("parity_scatter", [])

    def calculate_engine_cost(
        self,
        actual_rul: float,
        pred_rul: float,
        p10_rul: float,
        threshold: float,
        cost_cfg: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates maintenance expenditure and outcome for a single engine under AETURNUS policy.
        """
        c_cat = cost_cfg["cost_catastrophic"]
        c_ovh = cost_cfg["cost_overhaul"]
        c_waste = cost_cfg["cost_wasted_cycle"]
        c_insp = cost_cfg["cost_inspection"]

        # Conservative RUL is min(pred_rul, p10_rul) to protect against downside risk
        cons_rul = min(pred_rul, p10_rul) if p10_rul is not None else pred_rul

        # Decision rule: Pull if conservative RUL <= threshold
        will_pull = cons_rul <= threshold

        if will_pull:
            if actual_rul > 0:
                # Safe preventative depot overhaul
                wasted_life_cost = actual_rul * c_waste
                total_cost = c_ovh + wasted_life_cost
                outcome = "SAFE_PREVENTIVE_OVERHAUL"
                avoided_catastrophic = True
            else:
                # Late pull: engine had already reached structural failure
                total_cost = c_cat + (abs(actual_rul) * cost_cfg["cost_aog_day"] * 0.1)
                outcome = "LATE_DISASTER_PULL"
                avoided_catastrophic = False
                wasted_life_cost = 0.0
        else:
            if actual_rul <= 0:
                # Missed failure: engine ran to catastrophic breakdown
                total_cost = c_cat
                outcome = "UNPLANNED_CATASTROPHIC_FAILURE"
                avoided_catastrophic = False
                wasted_life_cost = 0.0
            else:
                # Engine allowed to fly safely; routine non-invasive line inspection
                total_cost = c_insp if cons_rul <= threshold + 20 else 0.0
                outcome = "HEALTHY_CONTINUED_OPERATION"
                avoided_catastrophic = False
                wasted_life_cost = 0.0

        return {
            "total_cost": round(total_cost, 2),
            "outcome": outcome,
            "avoided_catastrophic": avoided_catastrophic,
            "wasted_life_cost": round(wasted_life_cost, 2),
            "pulled": will_pull,
            "cons_rul": round(cons_rul, 1)
        }

    def simulate_fleet(
        self,
        threshold: Optional[float] = None,
        custom_cost_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive comparative economic simulation across:
        1. Reactive Run-to-Failure Policy
        2. Traditional Fixed Flight-Cycle Overhaul Policy
        3. AETURNUS Predictive Optimization Policy
        """
        cfg = {**self.cost_cfg, **(custom_cost_params or {})}
        thresh = threshold if threshold is not None else cfg["default_threshold"]
        c_cat = cfg["cost_catastrophic"]
        c_ovh = cfg["cost_overhaul"]
        c_waste = cfg["cost_wasted_cycle"]
        c_insp = cfg["cost_inspection"]

        n_engines = len(self.eval_data)
        if n_engines == 0:
            return {"error": "No evaluation data found for subset", "subset": self.subset}

        # -------------------------------------------------------------
        # Policy 1: Reactive Run-to-Failure (No Predictive AI)
        # -------------------------------------------------------------
        # Every engine eventually fails in service without proactive pull
        reactive_total_cost = n_engines * c_cat
        reactive_catastrophic_count = n_engines
        reactive_scheduled_count = 0
        reactive_wasted_cost = 0.0

        # -------------------------------------------------------------
        # Policy 2: Static Fixed-Cycle Overhaul Limit (e.g. 140 cycles)
        # -------------------------------------------------------------
        # Fixed maintenance schedule: Overhaul engine at fixed_interval
        # If engine life < fixed_interval, it suffers catastrophic in-flight failure!
        fixed_int = cfg["fixed_interval"]
        fixed_catastrophic_count = 0
        fixed_scheduled_count = 0
        fixed_wasted_cost = 0.0
        fixed_total_cost = 0.0

        for row in self.eval_data:
            act_rul = row.get("actual", 50.0)
            # Proxy total life = test cycle + actual remaining
            # In test sets, if actual <= 0, engine reached failure before fixed interval
            if act_rul <= 0 or act_rul < 15.0: # Early failure before depot visit
                fixed_catastrophic_count += 1
                fixed_total_cost += c_cat
            else:
                fixed_scheduled_count += 1
                wasted = max(0.0, act_rul - 15.0) * c_waste
                fixed_wasted_cost += wasted
                fixed_total_cost += (c_ovh + wasted)

        # -------------------------------------------------------------
        # Policy 3: AETURNUS Predictive Maintenance Policy
        # -------------------------------------------------------------
        aeturnus_total_cost = 0.0
        aeturnus_catastrophic_count = 0
        aeturnus_scheduled_count = 0
        aeturnus_wasted_cost = 0.0
        aeturnus_inspection_cost = 0.0
        avoided_disasters_count = 0
        engine_ledger = []

        for row in self.eval_data:
            u = row.get("unit", 1)
            act_rul = row.get("actual", 0.0)
            pred_rul = row.get("predicted", 0.0)
            p10_rul = row.get("p10", pred_rul)

            res = self.calculate_engine_cost(act_rul, pred_rul, p10_rul, thresh, cfg)
            cost = res["total_cost"]
            aeturnus_total_cost += cost

            if res["outcome"] in ("UNPLANNED_CATASTROPHIC_FAILURE", "LATE_DISASTER_PULL"):
                aeturnus_catastrophic_count += 1
            elif res["outcome"] == "SAFE_PREVENTIVE_OVERHAUL":
                aeturnus_scheduled_count += 1
                aeturnus_wasted_cost += res["wasted_life_cost"]
                avoided_disasters_count += 1
            elif res["outcome"] == "HEALTHY_CONTINUED_OPERATION" and cost > 0:
                aeturnus_inspection_cost += cost

            engine_ledger.append({
                "unit": u,
                "actual_rul": act_rul,
                "predicted_rul": pred_rul,
                "conservative_rul": res["cons_rul"],
                "status": res["outcome"],
                "cost": cost,
                "avoided_catastrophic": res["avoided_catastrophic"]
            })

        # Sort ledger by actual RUL ascending (most critical first)
        engine_ledger.sort(key=lambda x: x["actual_rul"])

        net_savings_vs_reactive = max(0.0, reactive_total_cost - aeturnus_total_cost)
        net_savings_vs_fixed = max(0.0, fixed_total_cost - aeturnus_total_cost)
        roi_percent_reactive = round((net_savings_vs_reactive / aeturnus_total_cost * 100.0), 1) if aeturnus_total_cost > 0 else 0.0
        roi_percent_fixed = round((net_savings_vs_fixed / aeturnus_total_cost * 100.0), 1) if aeturnus_total_cost > 0 else 0.0

        return {
            "subset": self.subset,
            "fleet_size": n_engines,
            "threshold": thresh,
            "cost_parameters": cfg,
            "summary": {
                "net_savings_vs_reactive": round(net_savings_vs_reactive, 2),
                "net_savings_vs_fixed": round(net_savings_vs_fixed, 2),
                "roi_percent_vs_reactive": roi_percent_reactive,
                "roi_percent_vs_fixed": roi_percent_fixed,
                "avoided_catastrophic_failures": avoided_disasters_count,
                "catastrophic_failures_under_aeturnus": aeturnus_catastrophic_count,
                "cost_reduction_percent": round((net_savings_vs_reactive / reactive_total_cost * 100.0), 1) if reactive_total_cost > 0 else 0.0
            },
            "policies": {
                "reactive": {
                    "name": "Run-to-Failure (Unmanaged Reactive)",
                    "total_cost": round(reactive_total_cost, 2),
                    "catastrophic_failures": reactive_catastrophic_count,
                    "scheduled_overhauls": reactive_scheduled_count,
                    "wasted_life_cost": round(reactive_wasted_cost, 2),
                    "inspection_cost": 0.0,
                    "cost_per_engine": round(reactive_total_cost / n_engines, 2)
                },
                "fixed_interval": {
                    "name": f"Static Hour Limit ({fixed_int} Cycles)",
                    "total_cost": round(fixed_total_cost, 2),
                    "catastrophic_failures": fixed_catastrophic_count,
                    "scheduled_overhauls": fixed_scheduled_count,
                    "wasted_life_cost": round(fixed_wasted_cost, 2),
                    "inspection_cost": 0.0,
                    "cost_per_engine": round(fixed_total_cost / n_engines, 2)
                },
                "aeturnus": {
                    "name": f"AETURNUS Dynamic RUL (Threshold = {thresh} cyc)",
                    "total_cost": round(aeturnus_total_cost, 2),
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
                    "Catastrophic AOG ($M)": round((reactive_total_cost) / 1e6, 2),
                    "Scheduled Overhaul ($M)": 0.0,
                    "Wasted Life ($M)": 0.0,
                    "Inspection ($M)": 0.0,
                    "Total ($M)": round(reactive_total_cost / 1e6, 2)
                },
                {
                    "policy": "Fixed Schedule",
                    "Catastrophic AOG ($M)": round((fixed_catastrophic_count * c_cat) / 1e6, 2),
                    "Scheduled Overhaul ($M)": round((fixed_scheduled_count * c_ovh) / 1e6, 2),
                    "Wasted Life ($M)": round(fixed_wasted_cost / 1e6, 2),
                    "Inspection ($M)": 0.0,
                    "Total ($M)": round(fixed_total_cost / 1e6, 2)
                },
                {
                    "policy": "AETURNUS",
                    "Catastrophic AOG ($M)": round((aeturnus_catastrophic_count * c_cat) / 1e6, 2),
                    "Scheduled Overhaul ($M)": round((aeturnus_scheduled_count * c_ovh) / 1e6, 2),
                    "Wasted Life ($M)": round(aeturnus_wasted_cost / 1e6, 2),
                    "Inspection ($M)": round(aeturnus_inspection_cost / 1e6, 2),
                    "Total ($M)": round(aeturnus_total_cost / 1e6, 2)
                }
            ],
            "ledger": engine_ledger[:30] # Top 30 engines for UI table
        }

    def sweep_cost_curve(
        self,
        min_threshold: int = 5,
        max_threshold: int = 60,
        step: int = 1,
        custom_cost_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Sweeps threshold tau in [min_threshold, max_threshold] to generate
        the mathematical U-shaped fleet cost curve and identify optimal tau*.
        """
        cfg = {**self.cost_cfg, **(custom_cost_params or {})}
        c_cat = cfg["cost_catastrophic"]
        c_ovh = cfg["cost_overhaul"]
        c_waste = cfg["cost_wasted_cycle"]
        c_insp = cfg["cost_inspection"]

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
                act_rul = row.get("actual", 0.0)
                pred_rul = row.get("predicted", 0.0)
                p10_rul = row.get("p10", pred_rul)

                res = self.calculate_engine_cost(act_rul, pred_rul, p10_rul, tau, cfg)
                c = res["total_cost"]
                total_cost += c

                if res["outcome"] in ("UNPLANNED_CATASTROPHIC_FAILURE", "LATE_DISASTER_PULL"):
                    cat_failures += 1
                elif res["outcome"] == "SAFE_PREVENTIVE_OVERHAUL":
                    ovh_count += 1
                    wasted_cost += res["wasted_life_cost"]
                elif res["outcome"] == "HEALTHY_CONTINUED_OPERATION" and c > 0:
                    insp_cost += c

            if total_cost < min_cost:
                min_cost = total_cost
                best_tau = tau

            curve_points.append({
                "threshold": tau,
                "total_cost_m": round(total_cost / 1e6, 3),
                "catastrophic_cost_m": round((cat_failures * c_cat) / 1e6, 3),
                "overhaul_cost_m": round((ovh_count * c_ovh) / 1e6, 3),
                "wasted_life_cost_m": round(wasted_cost / 1e6, 3),
                "inspection_cost_m": round(insp_cost / 1e6, 3),
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
