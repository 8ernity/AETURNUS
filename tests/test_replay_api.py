import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from src.api.server import app, replay_state, current_subset
from src.common.utils import get_root_dir
from src.features.streaming import StreamingEnginePipeline

client = TestClient(app)


@pytest.fixture(autouse=True)
def restore_shipped_model_artifacts():
    """Promote/rollback endpoints mutate the served bundle and registry; restore them after each test."""
    root = get_root_dir()
    bundle = root / "artifacts" / "models" / "model_bundle_FD001.joblib"
    registry = root / "artifacts" / "reports" / "model_registry.json"
    bundle_bytes = bundle.read_bytes()
    registry_text = registry.read_text(encoding="utf-8") if registry.exists() else None
    yield
    bundle.write_bytes(bundle_bytes)
    if registry_text is not None:
        registry.write_text(registry_text, encoding="utf-8")

def test_fleet_endpoint_non_mutating_and_all_engines():
    """Verify GET /fleet returns all engines, does not mutate server state, and handles subsets."""
    init_subset = current_subset
    response = client.get("/api/v1/fleet?subset=FD001")
    assert response.status_code == 200
    data = response.json()
    assert data["subset"] == "FD001"
    # All 100 engines in FD001 test set must be present
    assert data["kpis"]["engines_monitored"] == 100
    assert len(data["engines"]) == 100
    # Global state must remain unmutated
    from src.api.server import current_subset as after_subset
    assert after_subset == init_subset

def test_invalid_subset_returns_400():
    """Verify invalid subset returns HTTP 400, never 500 error."""
    resp = client.get("/api/v1/fleet?subset=FD999")
    assert resp.status_code == 400
    assert "Invalid or unsupported subset" in resp.json()["detail"]

    resp_detail = client.get("/api/v1/engines/ENG-001?subset=FD_INVALID")
    assert resp_detail.status_code == 400

    resp_models = client.get("/api/v1/models/FD999/details")
    assert resp_models.status_code == 400

def test_engine_detail_and_404_handling():
    """Verify GET /engines/{id} returns details or proper 404 for missing engine."""
    # Valid engine
    resp = client.get("/api/v1/engines/ENG-001?subset=FD001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_id"] == "ENG-001"
    assert "trajectory" in data
    assert len(data["trajectory"]) > 0

    # Non-existent engine in FD001
    resp_missing = client.get("/api/v1/engines/ENG-999?subset=FD001")
    assert resp_missing.status_code == 404

    # Malformed ID
    resp_bad = client.get("/api/v1/engines/bad_id?subset=FD001")
    assert resp_bad.status_code == 400

def test_streaming_engine_pipeline():
    """Verify StreamingEnginePipeline streams raw cycle through clean -> features -> predict -> triage."""
    pipeline = StreamingEnginePipeline(subset="FD001")
    raw_cycle_1 = {
        "unit": 1,
        "cycle": 1,
        "setting_1": -0.0007,
        "setting_2": -0.0004,
        "setting_3": 100.0,
        "s1": 518.67, "s2": 641.82, "s3": 1589.70, "s4": 1400.60,
        "s5": 14.62, "s6": 21.61, "s7": 554.36, "s8": 2388.06,
        "s9": 9046.19, "s10": 1.30, "s11": 47.47, "s12": 521.66,
        "s13": 2388.02, "s14": 8138.62, "s15": 8.4195, "s16": 0.03,
        "s17": 392, "s18": 2388, "s19": 100.0, "s20": 39.06, "s21": 23.4190
    }
    res = pipeline.stream_cycle(raw_cycle_1)
    assert res["unit"] == 1
    assert res["cycle"] == 1
    assert "prediction" in res
    assert "rul_p50" in res["prediction"]
    assert "conservative_rul" in res["prediction"]

def test_replay_controls():
    """Test start, pause, reset replay endpoints."""
    resp = client.post("/api/v1/replay/start?speed=5")
    assert resp.status_code == 200
    assert resp.json()["status"] in ["started", "running"]
    assert resp.json()["speed"] == 5.0

    resp = client.post("/api/v1/replay/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    resp = client.post("/api/v1/replay/reset")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reset"

def test_missing_endpoints_reload_promote_rollback_pipeline_upload():
    """Test promote, rollback, reload, pipeline run, and upload endpoints."""
    # 1. Reload models
    resp_reload = client.post("/api/v1/models/reload?subset=FD001")
    assert resp_reload.status_code == 200
    assert "reloaded" in resp_reload.json()["status"]

    # 2. Promote model endpoint
    resp_promote = client.post("/api/v1/models/v1.0/promote?subset=FD001")
    assert resp_promote.status_code == 200
    assert resp_promote.json()["status"] == "promoted"
    assert resp_promote.json()["version"] == "v1.0"

    # 3. Rollback model endpoint
    resp_rb = client.post("/api/v1/models/v1.0/rollback?subset=FD001")
    assert resp_rb.status_code == 200
    assert resp_rb.json()["status"] == "rolled_back"

    # 4. Pipeline run endpoint
    resp_pipe = client.post("/api/v1/pipeline/run?subset=FD001")
    assert resp_pipe.status_code == 200
    assert resp_pipe.json()["status"] == "success"
    assert "report" in resp_pipe.json()

    # 5. Data upload endpoint
    dummy_csv = b"unit,cycle,setting_1\n1,1,0.001\n"
    files = {"file": ("test_upload.csv", BytesIO(dummy_csv), "text/csv")}
    resp_upload = client.post("/api/v1/data/upload?subset=FD001", files=files)
    assert resp_upload.status_code == 200
    assert resp_upload.json()["status"] == "uploaded"
    assert resp_upload.json()["filename"] == "test_upload.csv"
