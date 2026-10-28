import json
import time

import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.mlops.alerts import AlertManager

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)

def test_prometheus_metrics_endpoint():
    """Verify GET /metrics returns Prometheus format with pdm_ metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "pdm_predictions_total" in text
    assert "pdm_model_test_rmse" in text
    assert "pdm_conformal_coverage" in text
    assert "pdm_active_alerts" in text
    assert "pdm_artifact_hash_info" in text
    assert "pdm_artifact_mtime_seconds" in text


def test_metrics_scrape_is_fast_and_hashes_are_stable():
    client.get("/metrics")
    start = time.perf_counter()
    resp = client.get("/metrics")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed < 0.5, f"/metrics scrape took {elapsed:.3f}s, expected < 0.5s"
    assert resp.text.count("pdm_artifact_hash_info") >= 5


def test_artifact_hash_cache_invalidates_on_mtime(tmp_path, monkeypatch):
    import os
    import src.api.server as server

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first")
    monkeypatch.setattr(server, "_artifact_paths", lambda: {"test_artifact": artifact})

    first = server.refresh_artifact_metrics(force=True)["test_artifact"]
    assert server.refresh_artifact_metrics()["test_artifact"] == first, (
        "Unchanged mtime must reuse the cached digest"
    )

    artifact.write_bytes(b"second")
    stat = artifact.stat()
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = server.refresh_artifact_metrics()["test_artifact"]
    assert second != first, "Changed mtime must invalidate the cached digest"

def test_monitoring_alerts_endpoint():
    """Verify GET /api/v1/monitoring/alerts returns structured active alerts."""
    resp = client.get("/api/v1/monitoring/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_alerts_count" in data
    assert "total_rules_evaluated" in data
    assert "status" in data
    assert "alerts" in data
    assert len(data["alerts"]) >= 4
    
    for a in data["alerts"]:
        assert "alert_id" in a
        assert "rule_name" in a
        assert "severity" in a
        assert "firing" in a
        assert "metric_value" in a
        assert "threshold" in a

def test_alert_manager_synthetic_triggers():
    """Verify alert rules trigger accurately under specific anomaly conditions."""
    mgr = AlertManager()
    
    # 1. Nominal conditions
    nominal_ctx = {
        "drift_metrics": {"s3_dev_base": 0.05, "s7_dev_base": 0.08},
        "backlog_count": 3,
        "coverage_metrics": {"FD001": 0.85, "FD002": 0.80},
        "latency_p95_ms": 15.0
    }
    nominal_alerts = mgr.evaluate_alerts(context=nominal_ctx)
    assert all(not a["firing"] for a in nominal_alerts)
    
    # 2. Elevated drift condition
    drift_ctx = {
        "drift_metrics": {"s3_dev_base": 0.35, "s7_dev_base": 0.08},
        "backlog_count": 3,
        "coverage_metrics": {"FD001": 0.85},
        "latency_p95_ms": 15.0
    }
    drift_alerts = mgr.evaluate_alerts(context=drift_ctx)
    drift_alert = next(a for a in drift_alerts if a["rule_name"] == "RULE_ELEVATED_FEATURE_DRIFT")
    assert drift_alert["firing"] is True
    assert drift_alert["severity"] == "WARNING"
    
    # 3. Work order backlog spike condition
    backlog_ctx = {
        "drift_metrics": {"s3_dev_base": 0.05},
        "backlog_count": 15,
        "coverage_metrics": {"FD001": 0.85},
        "latency_p95_ms": 15.0
    }
    backlog_alerts = mgr.evaluate_alerts(context=backlog_ctx)
    backlog_alert = next(a for a in backlog_alerts if a["rule_name"] == "RULE_MAINTENANCE_BACKLOG_SPIKE")
    assert backlog_alert["firing"] is True
    
    # 4. Coverage dip condition
    cov_ctx = {
        "drift_metrics": {"s3_dev_base": 0.05},
        "backlog_count": 3,
        "coverage_metrics": {"FD001": 0.68},
        "latency_p95_ms": 15.0
    }
    cov_alerts = mgr.evaluate_alerts(context=cov_ctx)
    cov_alert = next(a for a in cov_alerts if a["rule_name"] == "RULE_CONFORMAL_COVERAGE_DIP")
    assert cov_alert["firing"] is True
    assert cov_alert["metric_name"] == "coverage_le_cap"
    
    # 5. Latency spike condition
    lat_ctx = {
        "drift_metrics": {"s3_dev_base": 0.05},
        "backlog_count": 3,
        "coverage_metrics": {"FD001": 0.85},
        "latency_p95_ms": 145.0
    }
    lat_alerts = mgr.evaluate_alerts(context=lat_ctx)
    lat_alert = next(a for a in lat_alerts if a["rule_name"] == "RULE_INFERENCE_LATENCY_SPIKE")
    assert lat_alert["firing"] is True

def test_conformal_coverage_alert_manifest():
    mgr = AlertManager()
    mgr.cov_min_thresh = 0.75

    # 1. With coverage 0.63 and threshold 0.75, the alert fires
    alerts_firing = mgr.evaluate_alerts(context={"coverage_metrics": {"FD001": 0.63}})
    cov_alert = next(a for a in alerts_firing if a["rule_name"] == "RULE_CONFORMAL_COVERAGE_DIP")
    assert cov_alert["firing"] is True
    assert cov_alert["metric_value"] == 0.63
    assert cov_alert["state"] == "firing"

    # 2. With manifest removed, says 'no data' (not firing, not passing)
    manifest_path = mgr.root_dir / "artifacts" / "run_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest_path.unlink()
        alerts_nodata = mgr.evaluate_alerts(context={})
        nodata_alert = next(a for a in alerts_nodata if a["rule_name"] == "RULE_CONFORMAL_COVERAGE_DIP")
        assert nodata_alert["state"] == "no data"
        assert nodata_alert["status"] == "no data"
        assert nodata_alert["firing"] is False
    finally:
        manifest_path.write_bytes(manifest_bytes)


def test_conformal_coverage_alert_uses_manifest_policy_metric():
    """The coverage rule must evaluate the policy metric, not interval_coverage."""
    mgr = AlertManager()
    mgr.cov_min_thresh = 0.75
    manifest_path = mgr.root_dir / "artifacts" / "run_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    metric_name = manifest["coverage_metric"]["value"]
    assert metric_name == "coverage_le_cap"
    expected = min(
        float(sub["metrics"][metric_name])
        for sub in manifest["subsets"].values()
        if sub["metrics"].get(metric_name) is not None
    )

    alerts = mgr.evaluate_alerts(context={})
    cov_alert = next(a for a in alerts if a["rule_name"] == "RULE_CONFORMAL_COVERAGE_DIP")
    assert cov_alert["metric_name"] == metric_name
    assert cov_alert["metric_value"] == round(expected, 3)
    assert cov_alert["firing"] is (expected < mgr.cov_min_thresh)


def test_webhook_dispatcher_payload_formatting():
    """Verify webhook dispatcher formats payloads correctly for Slack, Teams, PagerDuty, and Generic schemas."""
    from src.mlops.alerts import WebhookDispatcher

    # 1. Slack format for Rule M1
    slack_dispatcher = WebhookDispatcher({"webhook": {"service": "slack"}})
    slack_payload = slack_dispatcher.format_payload("M1", 42, 120, "CRITICAL", 12.5)
    assert "blocks" in slack_payload
    assert "text" in slack_payload
    assert any("M1" in str(b) for b in slack_payload["blocks"])
    assert any("#42" in str(b) for b in slack_payload["blocks"])

    # 2. Teams format for Rule M2
    teams_dispatcher = WebhookDispatcher({"webhook": {"service": "teams"}})
    teams_payload = teams_dispatcher.format_payload("M2", 15, 85, "HIGH", 28.0)
    assert teams_payload["@type"] == "MessageCard"
    assert "sections" in teams_payload
    assert any(f["name"] == "Rule" for s in teams_payload["sections"] for f in s.get("facts", []))

    # 3. PagerDuty format for Rule M3
    pd_dispatcher = WebhookDispatcher({"webhook": {"service": "pagerduty"}})
    pd_payload = pd_dispatcher.format_payload("M3", 7, 160, "CRITICAL", 8.4)
    assert "payload" in pd_payload
    assert pd_payload["payload"]["severity"] == "critical"
    assert pd_payload["payload"]["custom_details"]["rule_id"] == "M3"

    # 4. Generic format
    gen_dispatcher = WebhookDispatcher({"webhook": {"service": "generic"}})
    gen_payload = gen_dispatcher.format_payload("M1", 99, 45, "HIGH", 22.0)
    assert gen_payload["event_type"] == "RULE_TRIGGERED"
    assert gen_payload["rule_id"] == "M1"
    assert gen_payload["engine_id"] == "99"


def test_webhook_dispatcher_dispatch_execution():
    """Verify webhook dispatcher handles mock dispatch, disabled state, and rule filtering."""
    from src.mlops.alerts import WebhookDispatcher

    sent_calls = []
    def mock_sender(url, payload):
        sent_calls.append({"url": url, "payload": payload})
        class MockResp:
            status_code = 200
        return MockResp()

    # Enabled dispatcher
    enabled_cfg = {
        "webhook": {
            "enabled": True,
            "url": "https://hooks.slack.com/services/TEST/123",
            "service": "slack",
            "notify_rules": ["M1", "M3"]
        }
    }
    dispatcher = WebhookDispatcher(enabled_cfg)

    # Dispatched Rule M1
    res_m1 = dispatcher.dispatch_m_rule("M1", "E-101", 150, "CRITICAL", 11.2, mock_sender=mock_sender)
    assert res_m1["status"] == "sent"
    assert len(sent_calls) == 1
    assert "E-101" in str(sent_calls[0]["payload"])

    # Skipped Rule M2 (not in notify_rules)
    res_m2 = dispatcher.dispatch_m_rule("M2", "E-102", 90, "HIGH", 25.0, mock_sender=mock_sender)
    assert res_m2["status"] == "skipped"
    assert len(sent_calls) == 1

    # Disabled dispatcher
    disabled_cfg = {"webhook": {"enabled": False, "url": ""}}
    disabled_dispatcher = WebhookDispatcher(disabled_cfg)
    res_dis = disabled_dispatcher.dispatch_m_rule("M1", "E-103", 50, "CRITICAL", 10.0, mock_sender=mock_sender)
    assert res_dis["status"] == "disabled"


def test_monitoring_webhooks_api_endpoints():
    """Verify GET /api/v1/monitoring/webhooks and POST /api/v1/monitoring/webhooks/test."""
    get_resp = client.get("/api/v1/monitoring/webhooks")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert "enabled" in data
    assert "service" in data
    assert "notify_rules" in data
    assert "M1" in data["notify_rules"]
    assert "recent_dispatches" in data

    post_resp = client.post("/api/v1/monitoring/webhooks/test?rule_id=M1&engine_id=12&cycle=135&severity=CRITICAL&conservative_rul=14.2")
    assert post_resp.status_code == 200
    test_data = post_resp.json()
    assert test_data["status"] in ["disabled", "sent", "failed"]

