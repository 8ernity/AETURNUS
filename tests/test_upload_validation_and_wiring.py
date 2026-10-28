import io
import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.common.utils import get_root_dir

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)

@pytest.fixture(autouse=True)
def isolate_uploads(tmp_path, monkeypatch):
    import src.api.server as server
    # Redirect ROOT_DIR to tmp_path during upload tests
    monkeypatch.setattr(server, "ROOT_DIR", tmp_path)
    (tmp_path / "data" / "uploads" / "FD001").mkdir(parents=True, exist_ok=True)
    yield

def test_upload_filename_sanitization():
    # Attempt directory traversal in filename
    csv_bytes = b"unit,cycle,setting_1\n1,1,0.001\n"
    files = {"file": ("../../malicious_file.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = client.post("/api/v1/data/upload?subset=FD001", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "uploaded"
    # Filename must be sanitized to pure basename without '../'
    assert data["filename"] == "malicious_file.csv"
    assert ".." not in data["saved_path"]
    assert "malicious_file.csv" in data["saved_path"]

def test_upload_extension_and_size_validation():
    # Disallowed extension
    script_bytes = b"echo 'hack'"
    files = {"file": ("evil.sh", io.BytesIO(script_bytes), "text/plain")}
    resp = client.post("/api/v1/data/upload?subset=FD001", files=files)
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]

    # Empty file
    files_empty = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    resp_empty = client.post("/api/v1/data/upload?subset=FD001", files=files_empty)
    assert resp_empty.status_code == 400

def test_upload_schema_validation():
    # Missing unit or cycle columns
    bad_csv = b"sensor_a,sensor_b\n1.2,3.4\n"
    files = {"file": ("bad_schema.csv", io.BytesIO(bad_csv), "text/csv")}
    resp = client.post("/api/v1/data/upload?subset=FD001", files=files)
    assert resp.status_code == 400
    assert "Schema validation failed" in resp.json()["detail"]

def test_upload_wired_into_pipeline_and_retrain():
    valid_csv = b"unit,cycle,setting_1,setting_2,setting_3\n1,1,0.001,0.002,100.0\n1,2,0.001,0.002,100.0\n"
    files = {"file": ("valid_data.csv", io.BytesIO(valid_csv), "text/csv")}
    resp = client.post("/api/v1/data/upload?subset=FD001&trigger_pipeline=false", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "uploaded"
    assert data["rows"] == 2
    assert "unit" in data["columns"]
    assert "cycle" in data["columns"]
