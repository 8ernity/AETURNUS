import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.model_service import ModelService
from src.agent.triage_agent import TriageAgent
from src.common.utils import get_root_dir

root_dir = get_root_dir()

def test_multi_horizon_probabilities_bounded():
    """Verify fail_prob_10, fail_prob_30, fail_prob_60 are within [0, 1]."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        dummy_row = {c: 0.5 for c in ms.feature_cols}
        dummy_row["unit"] = 1
        dummy_row["cycle"] = 100
        pred = ms.predict_point(dummy_row)
        
        for h in [10, 30, 60]:
            k = f"fail_prob_{h}"
            assert k in pred, f"{k} must be in prediction for {sub}"
            p = pred[k]
            assert 0.0 <= p <= 1.0, f"{k} out of range [0, 1]: {p}"
        
        # Risk generally monotonic: P(fail <= 10) <= P(fail <= 30) <= P(fail <= 60)
        p10 = pred["fail_prob_10"]
        p30 = pred["fail_prob_30"]
        p60 = pred["fail_prob_60"]
        assert p10 <= p30 + 0.15, f"Expected p10 <= p30 roughly, got {p10} and {p30}"
        assert p30 <= p60 + 0.15, f"Expected p30 <= p60 roughly, got {p30} and {p60}"

def test_validation_tuned_threshold_provenance():
    """Verify validation-tuned threshold is present and valid."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        ms = ModelService(sub).load()
        thresh = getattr(ms, "classifier_threshold_30", None)
        assert thresh is not None, f"classifier_threshold_30 must exist in bundle for {sub}"
        assert 0.10 <= thresh <= 0.90, f"Threshold {thresh} should be reasonable [0.10, 0.90]"

def test_triage_rule_m3_escalation():
    """Verify rule M3 escalates when fail_prob_30 >= tau_30 even if RUL is medium."""
    agent = TriageAgent(db_path=":memory:")
    
    # Engine with RUL = 50 (base MEDIUM), but high fail_prob_30 (0.65) and tau_30 = 0.45
    prediction = {
        "engine_id": "FD001-042",
        "unit": 42,
        "subset": "FD001",
        "cycle": 120,
        "conservative_rul": 50.0,
        "rul_p10": 45.0,
        "rul_p50": 55.0,
        "rul_p90": 65.0,
        "fail_prob_10": 0.05,
        "fail_prob_30": 0.65,
        "fail_prob_60": 0.90,
        "classifier_threshold_30": 0.45,
        "anomaly_score": 0.10,
        "top_sensors": ["s11", "s12", "s13"]
    }
    
    order, trace = agent.evaluate_engine_cycle(prediction)
    assert any("M3:fired" in t for t in trace), f"M3 should fire, trace: {trace}"
    assert order is not None
    assert order["severity"] in ["HIGH", "CRITICAL"], f"Severity should escalate to at least HIGH, got {order['severity']}"

def test_triage_rule_m3_fires_critical_on_imminent_failure():
    """Verify rule M3 escalates to CRITICAL if fail_prob_10 >= 0.70."""
    agent = TriageAgent(db_path=":memory:")
    prediction = {
        "engine_id": "FD001-099",
        "unit": 99,
        "subset": "FD001",
        "cycle": 150,
        "conservative_rul": 40.0,
        "rul_p10": 35.0,
        "rul_p50": 45.0,
        "rul_p90": 55.0,
        "fail_prob_10": 0.82,
        "fail_prob_30": 0.95,
        "fail_prob_60": 0.99,
        "classifier_threshold_30": 0.45,
        "anomaly_score": 0.20,
        "top_sensors": ["s11", "s12"]
    }
    order, trace = agent.evaluate_engine_cycle(prediction)
    assert any("fail_prob_10" in t and "CRITICAL" in t for t in trace), f"Trace should log CRITICAL for imminent failure: {trace}"
    assert order is not None
    assert order["severity"] == "CRITICAL"

def test_multi_horizon_metrics_reported_in_all_subsets():
    """Verify evaluation reports contain complete multi-horizon metrics."""
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        report_file = root_dir / "artifacts" / "reports" / f"evaluation_{sub}.json"
        assert report_file.exists()
        with open(report_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        mhm = d.get("multi_horizon_metrics")
        assert mhm is not None, f"multi_horizon_metrics missing in evaluation_{sub}.json"
        for h_str in ["10", "30", "60"]:
            assert h_str in mhm, f"Horizon {h_str} missing in evaluation_{sub}.json"
            h_data = mhm[h_str]
            assert "pr_auc" in h_data
            assert "threshold" in h_data
            assert "precision" in h_data
            assert "recall" in h_data
            assert h_data["pr_auc"] >= 0.70, f"PR-AUC for {sub} horizon {h_str} should be >= 0.70, got {h_data['pr_auc']}"
