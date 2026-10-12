import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.mlops.alerts import AlertManager

client = TestClient(app)

def test_prometheus_metrics_endpoint():
    """Verify GET /metrics returns Prometheus format with pdm_ metrics."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "pdm_predictions_total" in text
    assert "pdm_model_test_rmse" in text
    assert "pdm_conformal_coverage" in text
    assert "pdm_active_alerts" in text

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
