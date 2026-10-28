import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.api.server import app

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)

def test_predict_batch_csv_with_processed_features():
    """Verify batch scoring endpoint works on CSV with features."""
    from src.api.server import get_test_df
    full_df = get_test_df("FD001")
    df_u1 = full_df[full_df["unit"] == 1].iloc[-1:]
    df_u2 = full_df[full_df["unit"] == 2].iloc[-1:]
    df_sample = pd.concat([df_u1, df_u2], ignore_index=True)
    csv_content = df_sample.to_csv(index=False)
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

def test_predict_batch_missing_columns_422():
    """Verify batch scoring returns 422 with missing columns when required features are missing."""
    csv_content = (
        "unit,cycle,s2_norm,s2_ewma\n"
        "1,30,0.5,0.48\n"
    )
    files = {"file": ("missing_cols.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    resp = client.post("/api/v1/predict/batch?subset=FD001", files=files)
    assert resp.status_code == 422
    assert "missing" in resp.json()["detail"].lower()
    assert "required feature" in resp.json()["detail"].lower()

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
