import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.common.utils import get_root_dir, load_yaml

class AlertManager:
    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = get_root_dir()
        cfg_path = Path(config_path) if config_path else self.root_dir / "configs" / "monitoring.yaml"
        if cfg_path.exists():
            cfg = load_yaml(str(cfg_path))
        else:
            cfg = {}
        
        self.alerts_cfg = cfg.get("alerts", {})
        self.psi_thresh = float(self.alerts_cfg.get("drift_psi_threshold", 0.25))
        self.key_sensors = self.alerts_cfg.get("key_drift_sensors", ["s3", "s7", "s11"])
        self.backlog_thresh = int(self.alerts_cfg.get("backlog_critical_high_threshold", 10))
        self.cov_min_thresh = float(self.alerts_cfg.get("coverage_min_threshold", 0.75))
        self.latency_thresh_ms = float(self.alerts_cfg.get("latency_p95_ms_threshold", 100.0))

    def evaluate_alerts(
        self,
        triage_agent = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates operational health rules and returns list of alert objects.
        """
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        ctx = context or {}
        alerts: List[Dict[str, Any]] = []

        # 1. Rule: Elevated Drift on Key Sensors
        drift_firing = False
        max_psi = 0.0
        drift_sensor = ""
        # Check context or drift_history.json
        if "drift_metrics" in ctx:
            for s, psi in ctx["drift_metrics"].items():
                if any(k in s for k in self.key_sensors) and psi > max_psi:
                    max_psi = psi
                    drift_sensor = s
        else:
            drift_file = self.root_dir / "artifacts" / "reports" / "drift_history.json"
            if drift_file.exists():
                try:
                    with open(drift_file, "r", encoding="utf-8") as f:
                        drift_data = json.load(f)
                    records = drift_data.get("history", [])
                    if records:
                        last_rec = records[-1]
                        for feat, psi_val in last_rec.get("feature_psi", {}).items():
                            if any(k in feat for k in self.key_sensors) and psi_val > max_psi:
                                max_psi = psi_val
                                drift_sensor = feat
                except Exception:
                    pass

        if max_psi >= self.psi_thresh:
            drift_firing = True

        alerts.append({
            "alert_id": "ALERT-DRIFT-001",
            "rule_name": "RULE_ELEVATED_FEATURE_DRIFT",
            "severity": "WARNING",
            "firing": drift_firing,
            "metric_value": round(float(max_psi), 3),
            "threshold": self.psi_thresh,
            "message": f"Elevated drift detected on {drift_sensor or 'key sensors'} (PSI={max_psi:.3f} >= {self.psi_thresh})" if drift_firing else "Telemetry distributions stable across fleet regimes.",
            "timestamp": now
        })

        # 2. Rule: Work Order Backlog Spike
        backlog_count = 0
        if triage_agent and hasattr(triage_agent, "open_orders"):
            backlog_count = sum(
                1 for o in triage_agent.open_orders.values()
                if o.get("severity") in ["CRITICAL", "HIGH"]
            )
        elif "backlog_count" in ctx:
            backlog_count = ctx["backlog_count"]

        backlog_firing = backlog_count > self.backlog_thresh
        alerts.append({
            "alert_id": "ALERT-BACKLOG-002",
            "rule_name": "RULE_MAINTENANCE_BACKLOG_SPIKE",
            "severity": "CRITICAL" if backlog_count > self.backlog_thresh * 1.5 else "WARNING",
            "firing": backlog_firing,
            "metric_value": backlog_count,
            "threshold": self.backlog_thresh,
            "message": f"Critical/High work orders backlog ({backlog_count}) exceeds capacity threshold ({self.backlog_thresh})" if backlog_firing else "Maintenance backlog within operational hangar capacity.",
            "timestamp": now
        })

        # 3. Rule: Conformal Coverage Dip
        min_cov = 1.0
        lowest_sub = "FD001"
        if "coverage_metrics" in ctx:
            for sub, cov in ctx["coverage_metrics"].items():
                if cov < min_cov:
                    min_cov = cov
                    lowest_sub = sub
        else:
            for sub in ["FD001", "FD002", "FD003", "FD004"]:
                eval_file = self.root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
                if eval_file.exists():
                    try:
                        with open(eval_file, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        cov = d.get("interval_coverage", 0.80)
                        if cov < min_cov:
                            min_cov = cov
                            lowest_sub = sub
                    except Exception:
                        pass

        cov_firing = min_cov < self.cov_min_thresh
        alerts.append({
            "alert_id": "ALERT-COVERAGE-003",
            "rule_name": "RULE_CONFORMAL_COVERAGE_DIP",
            "severity": "WARNING",
            "firing": cov_firing,
            "metric_value": round(float(min_cov), 3),
            "threshold": self.cov_min_thresh,
            "message": f"Interval coverage for {lowest_sub} ({min_cov*100:.1f}%) dropped below nominal threshold ({self.cov_min_thresh*100:.1f}%)" if cov_firing else "Conformal prediction intervals meet empirical coverage targets.",
            "timestamp": now
        })

        # 4. Rule: Latency Spike
        lat_ms = ctx.get("latency_p95_ms", 12.5)
        lat_firing = lat_ms > self.latency_thresh_ms
        alerts.append({
            "alert_id": "ALERT-LATENCY-004",
            "rule_name": "RULE_INFERENCE_LATENCY_SPIKE",
            "severity": "WARNING",
            "firing": lat_firing,
            "metric_value": round(float(lat_ms), 1),
            "threshold": self.latency_thresh_ms,
            "message": f"P95 prediction latency ({lat_ms:.1f}ms) exceeds SLO threshold ({self.latency_thresh_ms:.1f}ms)" if lat_firing else "Inference latency satisfies real-time flight replay SLO (<100ms).",
            "timestamp": now
        })

        return alerts
