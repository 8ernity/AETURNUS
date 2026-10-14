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
        self.timeout_sec = float(wh_cfg.get("timeout_sec", 5.0))
        self.retry_count = int(wh_cfg.get("retry_count", 2))
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
        self.psi_thresh = float(self.alerts_cfg.get("drift_psi_threshold", 0.25))
        self.key_sensors = self.alerts_cfg.get("key_drift_sensors", ["s3", "s7", "s11"])
        self.backlog_thresh = int(self.alerts_cfg.get("backlog_critical_high_threshold", 10))
        self.cov_min_thresh = float(self.alerts_cfg.get("coverage_min_threshold", 0.75))
        self.latency_thresh_ms = float(self.alerts_cfg.get("latency_p95_ms_threshold", 100.0))

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
