import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests

from src.common.utils import get_root_dir, load_yaml

logger = logging.getLogger("aeturnus.alerts")

RULE_METADATA = {
    "M1": {
        "name": "RULE_M1_RAPID_RUL_DROP",
        "title": "Rapid RUL Drop",
        "description": "Steep degradation slope: RUL dropped faster than normal physical wear threshold.",
        "default_severity": "HIGH",
    },
    "M2": {
        "name": "RULE_M2_HIGH_ANOMALY_SCORE",
        "title": "High Anomaly Score",
        "description": "Isolation Forest multivariate sensor anomaly score breached healthy envelope threshold.",
        "default_severity": "MEDIUM",
    },
    "M3": {
        "name": "RULE_M3_MULTI_HORIZON_PROBABILITY",
        "title": "Multi-Horizon Failure Probability",
        "description": "Calibrated failure probability classifier breached operating threshold (H=10, 30, or 60 cycles).",
        "default_severity": "CRITICAL",
    },
}

class WebhookDispatcher:
    """
    Configurable external webhook dispatcher for operational alerting across Slack,
    Microsoft Teams, PagerDuty, and Generic JSON endpoints.
    """
    def __init__(self, alerts_cfg: Optional[Dict[str, Any]] = None):
        cfg = alerts_cfg or {}
        wh_cfg = cfg.get("webhook", {})
        
        # Allow environment variable override
        env_enabled = os.getenv("ALERT_WEBHOOK_ENABLED")
        if env_enabled is not None:
            self.enabled = env_enabled.lower() in ("true", "1", "yes")
        else:
            self.enabled = bool(wh_cfg.get("enabled", False))
            
        self.url = os.getenv("ALERT_WEBHOOK_URL", wh_cfg.get("url", ""))
        self.service = os.getenv("ALERT_WEBHOOK_SERVICE", wh_cfg.get("service", "slack")).lower()
        self.timeout_sec = float(wh_cfg.get("timeout_sec") or 5.0)
        self.retry_count = int(wh_cfg.get("retry_count") or 2)
        self.notify_rules = [r.upper() for r in wh_cfg.get("notify_rules", ["M1", "M2", "M3"])]
        self.recent_dispatches: List[Dict[str, Any]] = []

    def format_payload(
        self,
        rule_id: str,
        engine_id: Any,
        cycle: int,
        severity: str,
        conservative_rul: float,
        details: Optional[Dict[str, Any]] = None,
        suspected_subsystem: str = "HPC Compressor",
        recommended_action: str = "Inspect turbofan subsystem"
    ) -> Dict[str, Any]:
        """
        Formats alert payload according to target service schema (Slack, Teams, PagerDuty, Generic).
        """
        r_meta = RULE_METADATA.get(rule_id.upper(), {
            "name": f"RULE_{rule_id.upper()}",
            "title": f"Rule {rule_id.upper()}",
            "description": "Custom operational rule triggered."
        })
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rul_val = round(float(conservative_rul), 1)

        if self.service == "slack":
            sev_emoji = "🔴" if severity == "CRITICAL" else ("🟠" if severity == "HIGH" else "🟡")
            return {
                "text": f"{sev_emoji} [AETURNUS Alert] {r_meta['title']} on Engine #{engine_id} ({severity})",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{sev_emoji} AETURNUS Turbofan Alert: {r_meta['title']}",
                            "emoji": True
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Engine ID:*\n#{engine_id}"},
                            {"type": "mrkdwn", "text": f"*Cycle:*\n{cycle}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n`{severity}`"},
                            {"type": "mrkdwn", "text": f"*Conservative RUL:*\n{rul_val} cyc (p10)"},
                            {"type": "mrkdwn", "text": f"*Suspected Subsystem:*\n{suspected_subsystem}"},
                            {"type": "mrkdwn", "text": f"*Triggered Rule:*\n{rule_id} ({r_meta['name']})"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Recommended Action:*\n_{recommended_action}_\n\n*Condition Description:*\n{r_meta['description']}"
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": f"Platform: *AETURNUS Self-Healing Engine* | Timestamp: {now_str}"}
                        ]
                    }
                ]
            }

        elif self.service == "teams":
            theme_color = "D73A49" if severity == "CRITICAL" else ("FF9800" if severity == "HIGH" else "4CAF50")
            return {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": f"[AETURNUS Alert] Rule {rule_id} on Engine #{engine_id}",
                "themeColor": theme_color,
                "title": f"🚨 AETURNUS Turbofan Alert: Rule {rule_id} Triggered",
                "sections": [
                    {
                        "activityTitle": f"Engine #{engine_id} Escalated to {severity}",
                        "activitySubtitle": f"Evaluated at cycle {cycle} ({now_str})",
                        "facts": [
                            {"name": "Rule", "value": f"{rule_id} ({r_meta['title']})"},
                            {"name": "Severity", "value": severity},
                            {"name": "Conservative RUL (p10)", "value": f"{rul_val} cycles"},
                            {"name": "Suspected Subsystem", "value": suspected_subsystem},
                            {"name": "Action", "value": recommended_action}
                        ],
                        "text": r_meta["description"],
                        "markdown": True
                    }
                ]
            }

        elif self.service == "pagerduty":
            pd_severity = "critical" if severity == "CRITICAL" else ("error" if severity == "HIGH" else "warning")
            routing_key = os.getenv("PAGERDUTY_ROUTING_KEY", "aeturnus-turbofan-fleet")
            return {
                "routing_key": routing_key,
                "event_action": "trigger",
                "dedup_key": f"aeturnus-engine-{engine_id}-rule-{rule_id.lower()}",
                "payload": {
                    "summary": f"[{severity}] Rule {rule_id} on Engine #{engine_id}: {r_meta['title']} (RUL={rul_val} cyc)",
                    "source": "aeturnus-self-healing-engine",
                    "severity": pd_severity,
                    "timestamp": now_iso,
                    "component": suspected_subsystem,
                    "group": "turbofan-fleet",
                    "custom_details": {
                        "rule_id": rule_id,
                        "rule_name": r_meta["name"],
                        "engine_id": str(engine_id),
                        "cycle": cycle,
                        "severity": severity,
                        "conservative_rul": rul_val,
                        "suspected_subsystem": suspected_subsystem,
                        "recommended_action": recommended_action,
                        **(details or {})
                    }
                }
            }

        else:  # Generic JSON
            return {
                "event_type": "RULE_TRIGGERED",
                "rule_id": rule_id,
                "rule_name": r_meta["name"],
                "rule_title": r_meta["title"],
                "engine_id": str(engine_id),
                "cycle": cycle,
                "severity": severity,
                "conservative_rul": rul_val,
                "suspected_subsystem": suspected_subsystem,
                "recommended_action": recommended_action,
                "timestamp": now_iso,
                "details": details or {}
            }

    def dispatch_m_rule(
        self,
        rule_id: str,
        engine_id: Any,
        cycle: int,
        severity: str,
        conservative_rul: float,
        details: Optional[Dict[str, Any]] = None,
        suspected_subsystem: str = "HPC Compressor",
        recommended_action: str = "Inspect turbofan subsystem",
        mock_sender: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Dispatches webhook alert when Rule M1, M2, or M3 is triggered.
        Safe against network errors, timeout failures, or misconfigured URLs.
        """
        rid = rule_id.upper()
        if rid not in self.notify_rules:
            res = {
                "status": "skipped",
                "rule_id": rid,
                "reason": f"Rule {rid} not in notify_rules ({self.notify_rules})"
            }
            return res

        payload = self.format_payload(
            rule_id=rid,
            engine_id=engine_id,
            cycle=cycle,
            severity=severity,
            conservative_rul=conservative_rul,
            details=details,
            suspected_subsystem=suspected_subsystem,
            recommended_action=recommended_action
        )

        if not self.enabled or not self.url:
            record = {
                "dispatch_id": f"WH-DIS-{int(time.time()*1000)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "disabled",
                "rule_id": rid,
                "engine_id": str(engine_id),
                "severity": severity,
                "service": self.service,
                "url": self.url,
                "payload": payload,
                "reason": "Webhook dispatcher is disabled or URL is not configured."
            }
            self.recent_dispatches.append(record)
            if len(self.recent_dispatches) > 50:
                self.recent_dispatches.pop(0)
            return record

        status_code = None
        error_msg = None
        success = False

        if mock_sender is not None:
            try:
                res = mock_sender(self.url, payload)
                success = True
                status_code = getattr(res, "status_code", 200)
            except Exception as e:
                error_msg = str(e)
        else:
            for attempt in range(max(1, self.retry_count + 1)):
                try:
                    resp = requests.post(
                        self.url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout_sec
                    )
                    status_code = resp.status_code
                    if 200 <= resp.status_code < 300:
                        success = True
                        break
                    else:
                        error_msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
                except Exception as e:
                    error_msg = str(e)
                time.sleep(0.05)

        record = {
            "dispatch_id": f"WH-{int(time.time()*1000)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "sent" if success else "failed",
            "rule_id": rid,
            "engine_id": str(engine_id),
            "severity": severity,
            "service": self.service,
            "url": self.url,
            "status_code": status_code,
            "payload": payload,
            "error": error_msg
        }
        self.recent_dispatches.append(record)
        if len(self.recent_dispatches) > 50:
            self.recent_dispatches.pop(0)

        if not success:
            logger.warning(f"Failed to dispatch alert webhook for Rule {rid} (Engine #{engine_id}): {error_msg}")

        return record

    def dispatch_alert(self, alert_obj: Dict[str, Any], mock_sender: Optional[Any] = None) -> Dict[str, Any]:
        """
        Dispatches operational system alert (Drift, Backlog, Coverage, Latency).
        """
        if not self.enabled or not self.url:
            return {"status": "disabled"}
        
        if mock_sender is not None:
            try:
                mock_sender(self.url, alert_obj)
                return {"status": "sent", "status_code": 200}
            except Exception as e:
                return {"status": "failed", "error": str(e)}

        try:
            resp = requests.post(
                self.url,
                json=alert_obj,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_sec
            )
            return {"status": "sent" if 200 <= resp.status_code < 300 else "failed", "status_code": resp.status_code}
        except Exception as e:
            return {"status": "failed", "error": str(e)}


class AlertManager:
    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = get_root_dir()
        cfg_path = Path(config_path) if config_path else self.root_dir / "configs" / "monitoring.yaml"
        if cfg_path.exists():
            cfg = load_yaml(str(cfg_path))
        else:
            cfg = {}
        
        self.alerts_cfg = cfg.get("alerts", {})
        def _get_val(obj, key):
            item = obj.get(key)
            if isinstance(item, dict) and "value" in item:
                return item["value"]
            return item

        self.psi_thresh = float(_get_val(self.alerts_cfg, "drift_psi_threshold"))
        self.key_sensors = list(self.alerts_cfg["key_drift_sensors"])
        self.backlog_thresh = int(_get_val(self.alerts_cfg, "backlog_critical_high_threshold"))
        self.cov_min_thresh = float(_get_val(self.alerts_cfg, "coverage_min_threshold"))
        self.latency_thresh_ms = float(_get_val(self.alerts_cfg, "latency_p95_ms_threshold"))

        # External Alert Webhook Dispatcher
        self.webhook_dispatcher = WebhookDispatcher(self.alerts_cfg)

    def dispatch_m_rule(
        self,
        rule_id: str,
        engine_id: Any,
        cycle: int,
        severity: str,
        conservative_rul: float,
        details: Optional[Dict[str, Any]] = None,
        suspected_subsystem: str = "HPC Compressor",
        recommended_action: str = "Inspect turbofan subsystem",
        mock_sender: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Dispatches webhook alert when Rule M1, M2, or M3 triggers.
        """
        return self.webhook_dispatcher.dispatch_m_rule(
            rule_id=rule_id,
            engine_id=engine_id,
            cycle=cycle,
            severity=severity,
            conservative_rul=conservative_rul,
            details=details,
            suspected_subsystem=suspected_subsystem,
            recommended_action=recommended_action,
            mock_sender=mock_sender
        )

    def get_webhook_status(self) -> Dict[str, Any]:
        """
        Returns the current configuration and recent dispatch metrics for external webhooks.
        """
        return {
            "enabled": self.webhook_dispatcher.enabled,
            "service": self.webhook_dispatcher.service,
            "url_configured": bool(self.webhook_dispatcher.url),
            "timeout_sec": self.webhook_dispatcher.timeout_sec,
            "notify_rules": self.webhook_dispatcher.notify_rules,
            "total_dispatches": len(self.webhook_dispatcher.recent_dispatches),
            "recent_dispatches": self.webhook_dispatcher.recent_dispatches[-10:]
        }


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
        # Default subset is derived from the pipeline config (never a typed fallback).
        sub = ctx.get("subset") or load_yaml("configs/pipeline.yaml")["dataset"]["default_subset"]
        drift_firing = False
        max_psi = 0.0
        drift_sensor = ""

        if "drift_metrics" in ctx:
            for s, psi in ctx["drift_metrics"].items():
                val = float(psi["psi"]) if isinstance(psi, dict) and "psi" in psi else float(psi)
                if any(k in s for k in self.key_sensors) and val > max_psi:
                    max_psi = val
                    drift_sensor = s
            drift_firing = max_psi >= self.psi_thresh
            alerts.append({
                "alert_id": "ALERT-DRIFT-001",
                "rule_name": "RULE_ELEVATED_FEATURE_DRIFT",
                "severity": "WARNING",
                "status": "firing" if drift_firing else "nominal",
                "firing": drift_firing,
                "metric_value": round(float(max_psi), 3),
                "threshold": self.psi_thresh,
                "message": f"Elevated drift detected on {drift_sensor or 'key sensors'} (PSI={max_psi:.3f} >= {self.psi_thresh})" if drift_firing else "Telemetry distributions stable across fleet regimes.",
                "timestamp": now
            })
        else:
            from src.mlops.drift_and_retrain import get_latest_drift_run
            latest_drift = get_latest_drift_run(sub)
            if latest_drift is None:
                alerts.append({
                    "alert_id": "ALERT-DRIFT-001",
                    "rule_name": "RULE_ELEVATED_FEATURE_DRIFT",
                    "severity": "WARNING",
                    "status": "no data",
                    "state": "no data",
                    "firing": False,
                    "metric_value": None,
                    "threshold": self.psi_thresh,
                    "message": "no data",
                    "timestamp": now
                })
            else:
                sensor_metrics = latest_drift.get("sensor_metrics", {})
                for s, s_data in sensor_metrics.items():
                    psi = float(s_data["psi"]) if isinstance(s_data, dict) and "psi" in s_data else float(s_data)
                    if any(k in s for k in self.key_sensors) and psi > max_psi:
                        max_psi = psi
                        drift_sensor = s
                drift_firing = max_psi >= self.psi_thresh
                alerts.append({
                    "alert_id": "ALERT-DRIFT-001",
                    "rule_name": "RULE_ELEVATED_FEATURE_DRIFT",
                    "severity": "WARNING",
                    "status": "firing" if drift_firing else "nominal",
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

        # 3. Rule: Conformal Coverage Dip. Read the policy metric named in the
        # manifest (configs/targets.yaml coverage_metric), never a hard-coded
        # all-engine coverage key.
        cov_map: Dict[str, float] = {}
        targets_cfg = load_yaml("configs/targets.yaml", raw=True)
        coverage_metric = str(targets_cfg["coverage_metric"]["value"])
        if "coverage_metrics" in ctx:
            coverage_metric = str(ctx.get("coverage_metric", coverage_metric))
            cov_map = ctx["coverage_metrics"]
        else:
            manifest_file = self.root_dir / "artifacts" / "run_manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    coverage_metric = str(
                        m_data.get("coverage_metric", {}).get("value", coverage_metric)
                    )
                    for s_k, s_v in m_data.get("subsets", {}).items():
                        m_obj = s_v.get("metrics")
                        if isinstance(m_obj, dict) and m_obj.get(coverage_metric) is not None:
                            cov_map[s_k] = float(m_obj[coverage_metric])
                except Exception:
                    cov_map = {}

        if not cov_map:
            alerts.append({
                "alert_id": "ALERT-COVERAGE-003",
                "rule_name": "RULE_CONFORMAL_COVERAGE_DIP",
                "severity": "INFO",
                "status": "no data",
                "state": "no data",
                "firing": False,
                "metric_value": None,
                "metric_name": coverage_metric,
                "threshold": self.cov_min_thresh,
                "message": "no data",
                "timestamp": now
            })
        else:
            lowest_sub = min(cov_map, key=cov_map.get)
            min_cov = cov_map[lowest_sub]
            cov_firing = min_cov < self.cov_min_thresh
            alerts.append({
                "alert_id": "ALERT-COVERAGE-003",
                "rule_name": "RULE_CONFORMAL_COVERAGE_DIP",
                "severity": "WARNING",
                "status": "firing" if cov_firing else "nominal",
                "state": "firing" if cov_firing else "passing",
                "firing": cov_firing,
                "metric_value": round(float(min_cov), 3),
                "metric_name": coverage_metric,
                "threshold": self.cov_min_thresh,
                "message": (
                    f"Policy coverage {coverage_metric} for {lowest_sub} "
                    f"({min_cov*100:.1f}%) dropped below nominal threshold "
                    f"({self.cov_min_thresh*100:.1f}%)"
                    if cov_firing
                    else f"Policy coverage {coverage_metric} meets empirical coverage targets."
                ),
                "timestamp": now
            })

        # 4. Rule: Latency Spike
        if "latency_p95_ms" not in ctx:
            alerts.append({
                "alert_id": "ALERT-LATENCY-004",
                "rule_name": "RULE_INFERENCE_LATENCY_SPIKE",
                "severity": "INFO",
                "firing": None,
                "metric_value": None,
                "threshold": self.latency_thresh_ms,
                "message": "no data",
                "timestamp": now
            })
        else:
            lat_ms = float(ctx["latency_p95_ms"])
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
