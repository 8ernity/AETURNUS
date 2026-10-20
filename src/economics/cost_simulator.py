import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.common.utils import get_root_dir
from src.eval.protocol import sample_truncation, calculate_metrics
from src.ingest.loader import CMAPSSLoader
from src.models.model_service import ModelService

DEFAULT_COST_CONFIG = {
    "cost_catastrophic": {"value": 2500000.0, "kind": "assumption", "source": "Aviation unplanned in-flight shutdown AOG emergency cost estimate ($)"},
    "cost_overhaul": {"value": 220000.0, "kind": "assumption", "source": "Depot scheduled preventive overhaul cost ($)"},
    "cost_wasted_cycle": {"value": 1200.0, "kind": "assumption", "source": "Unused RUL component depreciation ($/cycle)"},
    "cost_inspection": {"value": 12500.0, "kind": "assumption", "source": "Tarmac borescope line inspection triage ($)"},
    "cost_aog_day": {"value": 45000.0, "kind": "assumption", "source": "Daily flight cancellation penalty ($/day)"},
    "default_threshold": {"value": 30.0, "kind": "policy", "source": "Fleet flight turn-around scheduling window"},
    "fixed_interval": {"value": 140.0, "kind": "policy", "source": "Fixed cycle OEM depot overhaul interval"}
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
        
        # Normalize cost_cfg
        flat_cfg = {}
        provenance_cfg = {}
        for k, v in DEFAULT_COST_CONFIG.items():
            if isinstance(v, dict) and "value" in v:
                flat_cfg[k] = float(v["value"])
                provenance_cfg[k] = v
            else:
                flat_cfg[k] = float(v)
                provenance_cfg[k] = {"value": float(v), "kind": "assumption", "source": "Fleet economic parameter"}
                
        if cost_params:
            for k, v in cost_params.items():
                if isinstance(v, dict) and "value" in v:
                    flat_cfg[k] = float(v["value"])
                    provenance_cfg[k] = v
                elif v is not None:
                    flat_cfg[k] = float(v)
                    provenance_cfg[k] = {"value": float(v), "kind": "assumption", "source": "Custom user scenario parameter"}
                    
        self.cost_cfg = flat_cfg
        self.provenance_cfg = provenance_cfg
        self.is_stale, self.stale_reason = self._check_stale()
        self.eval_data = self._load_evaluation_data()

    def _check_stale(self) -> (bool, str):
        bundle_file = self.root_dir / "artifacts" / "models" / f"model_bundle_{self.subset}.joblib"
        manifest_file = self.root_dir / "artifacts" / "run_manifest.json"
        if not bundle_file.exists():
            return True, f"Bundle model_bundle_{self.subset}.joblib not found on disk"
        if not manifest_file.exists():
            return True, "artifacts/run_manifest.json missing"
            
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            expected_sha = manifest.get("subsets", {}).get(self.subset, {}).get("bundle_sha256")
            actual_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
            if expected_sha and expected_sha != actual_sha:
                return True, f"Bundle SHA256 mismatch (manifest={expected_sha[:8]}..., disk={actual_sha[:8]}...)"
        except Exception as e:
            return True, f"Error validating manifest: {e}"
        return False, ""

    def _load_evaluation_data(self) -> List[Dict[str, Any]]:
        """
        Loads authentic engine-level test predictions and ground truth.
        Uses cached evaluation report if available, else evaluates live from bundle+data.
        """
        test_path = self.root_dir / "data" / "processed" / self.subset / "test.parquet"
        if not test_path.exists():
            return []

        eval_f = self.root_dir / "artifacts" / "reports" / f"evaluation_{self.subset}.json"
        if eval_f.exists() and not self.is_stale:
            try:
                with open(eval_f, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    scatter = d.get("parity_scatter", [])
                    if len(scatter) > 0:
                        return scatter
            except Exception:
                pass

        test_df = pd.read_parquet(test_path)
        loader = CMAPSSLoader(subset=self.subset)
        _, _, rul_truth = loader.load_raw_data()
        ms = ModelService(subset=self.subset).load()
        res = ms.evaluate_test(test_df, rul_truth, save_report=False)
        return res.get("parity_scatter", [])


    def _load_holdout_lifespans(self) -> List[int]:
        """
        Loads real run-to-failure lifetimes from persistent holdout engines or train engines.
        Each engine ran continuously until functional failure (max cycle).
        """
        from src.common.splits import get_persistent_split
        try:
            train_path = self.root_dir / "data" / "processed" / self.subset / "train.parquet"
            if not train_path.exists():
                return []
            df = pd.read_parquet(train_path, columns=["unit", "cycle"])
            split = get_persistent_split(self.subset)
            holdout_units = set(split.get("lifecycle_holdout_units", []))
            if holdout_units:
                hdf = df[df["unit"].isin(holdout_units)]
                if not hdf.empty:
                    df = hdf
            lifespans = df.groupby("unit")["cycle"].max().tolist()
            return [int(x) for x in lifespans] if lifespans else []
        except Exception:
            return []

    def simulate_fleet(
        self,
        threshold: Optional[float] = None,
        use_conservative: bool = True,
        custom_cost_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simulates fleet-wide economics under 3 operational maintenance policies:
        1. Reactive (Run-to-Failure)
        2. Fixed Flight-Cycle Schedule (Static Overhaul Interval from real holdout lifespans)
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
            return {"error": "No evaluation data found", "subset": self.subset, "stale": self.is_stale, "stale_reason": self.stale_reason}

        # -------------------------------------------------------------
        # Policy 1: Reactive Run-to-Failure (No Predictive AI)
        # -------------------------------------------------------------
        reactive_total_cost = n_engines * c_cat
        reactive_catastrophic_count = n_engines

        # -------------------------------------------------------------
        # Policy 2: Static Fixed-Cycle Schedule (Overhaul at fixed_int)
        # Simulated from real run-to-failure holdout engines using cfg fixed_interval
        # -------------------------------------------------------------
        holdout_lifespans = self._load_holdout_lifespans()

        h_cat_count = 0
        h_sch_count = 0
        h_wasted_cost = 0.0
        h_total_cost = 0.0

        for T in holdout_lifespans:
            if T < fixed_int:
                # Engine fails before reaching scheduled depot overhaul
                h_cat_count += 1
                h_total_cost += c_cat
            else:
                # Engine safely reaches scheduled depot overhaul at fixed_int cycles
                h_sch_count += 1
                wasted = max(0.0, float(T - fixed_int)) * c_waste
                h_wasted_cost += wasted
                h_total_cost += (c_ovh + wasted)

        H = max(1, len(holdout_lifespans))
        # Scale holdout policy rates to fleet size n_engines
        fixed_catastrophic_count = int(round((h_cat_count / H) * n_engines))
        fixed_scheduled_count = n_engines - fixed_catastrophic_count
        fixed_wasted_cost = (h_wasted_cost / H) * n_engines
        fixed_total_cost = (h_total_cost / H) * n_engines

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
                avoided_catastrophic = False
            else:
                # Safe preventive pull: depot overhaul + wasted remaining life
                wasted_life = act_at_pull * c_waste
                cost = c_ovh + wasted_life
                outcome = "SAFE_PREVENTIVE_OVERHAUL"
                aeturnus_scheduled_count += 1
                aeturnus_wasted_cost += wasted_life
                # "Avoided failure" ONLY if engine would fail before next scheduled interval
                avoided_catastrophic = bool(act <= fixed_int)
                if avoided_catastrophic:
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
                "avoided_catastrophic": avoided_catastrophic
            })

        # Sort ledger by actual RUL ascending (most critical engines first)
        engine_ledger.sort(key=lambda x: x["actual_rul"])

        # Unclamped net savings (remove max(0, ...), show true negative savings when applicable)
        net_savings_vs_reactive = reactive_total_cost - aeturnus_total_cost
        net_savings_vs_fixed = fixed_total_cost - aeturnus_total_cost
        cost_reduction_pct = round((net_savings_vs_reactive / reactive_total_cost * 100.0), 1) if reactive_total_cost > 0 else 0.0
        roi_multiplier = round((net_savings_vs_reactive / aeturnus_total_cost), 2) if aeturnus_total_cost > 0 else 0.0

        # Break-even cost ratio: (C_cat / C_ovh)* at which AETURNUS matches Fixed Interval baseline
        delta_cat = fixed_catastrophic_count - aeturnus_catastrophic_count
        if delta_cat > 0:
            breakeven_ratio = (
                (aeturnus_scheduled_count - fixed_scheduled_count) * c_ovh
                + (aeturnus_wasted_cost - fixed_wasted_cost)
                + aeturnus_inspection_cost
            ) / (delta_cat * c_ovh)
            breakeven_cost_ratio = max(0.5, round(float(breakeven_ratio), 2))
        else:
            breakeven_cost_ratio = 1.0
        current_cost_ratio = round(c_cat / max(1.0, c_ovh), 2)

        return {
            "subset": self.subset,
            "stale": self.is_stale,
            "stale_reason": self.stale_reason,
            "fleet_size": n_engines,
            "threshold": tau,
            "use_conservative": use_conservative,
            "cost_parameters": cfg,
            "cost_provenance": self.provenance_cfg,
            "summary": {
                "breakeven_cost_ratio": breakeven_cost_ratio,
                "current_cost_ratio": current_cost_ratio,
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
