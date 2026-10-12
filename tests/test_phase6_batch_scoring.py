import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_predict_batch_csv_with_processed_features():
    """Verify batch scoring endpoint works on CSV with features."""
    csv_content = (
        "unit,cycle,s2_norm,s2_ewma,s3_norm,s3_ewma,s4_norm,s4_ewma,s7_norm,s7_ewma\n"
        "1,30,0.5,0.48,0.6,0.58,0.4,0.38,0.7,0.68\n"
        "2,50,0.8,0.78,0.9,0.88,0.6,0.58,0.9,0.88\n"
    )
    files = {"file": ("batch_features.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    resp = client.post("/api/v1/predict/batch?subset=FD001", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["subset"] == "FD001"
    assert data["engines_scored"] == 2
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    first = data["predictions"][0]
    assert first["unit"] == 1
    assert "conservative_rul" in first
    assert "rul_p10" in first
    assert "rul_p50" in first
    assert "rul_p90" in first
    assert "fail_prob_30" in first
    assert "suspected_subsystem" in first
    assert "summary" in data
    assert "critical_count" in data["summary"]

def test_predict_batch_cmapss_raw_txt():
    """Verify batch scoring handles whitespace-delimited raw C-MAPSS format."""
    # 26 standard C-MAPSS columns for unit 1, cycle 1 and 2
    row1 = "1 1 0.001 0.002 100.0 " + " ".join([str(100.0 + i) for i in range(21)]) + "\n"
    row2 = "1 2 0.001 0.002 100.0 " + " ".join([str(101.0 + i) for i in range(21)]) + "\n"
    txt_content = (row1 + row2).encode("utf-8")
    
    files = {"file": ("test_FD001.txt", io.BytesIO(txt_content), "text/plain")}
    resp = client.post("/api/v1/predict/batch?subset=FD001", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["subset"] == "FD001"
    assert data["engines_scored"] == 1
    assert data["predictions"][0]["unit"] == 1

def test_predict_batch_validation_errors():
    """Verify empty file and missing schema trigger 400 errors."""
    # Empty file
    empty_files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    resp_empty = client.post("/api/v1/predict/batch?subset=FD001", files=empty_files)
    assert resp_empty.status_code == 400
    
    # Missing unit/cycle
    bad_csv = b"s2,s3\n1.0,2.0\n"
    bad_files = {"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")}
    resp_bad = client.post("/api/v1/predict/batch?subset=FD001", files=bad_files)
    assert resp_bad.status_code == 400
