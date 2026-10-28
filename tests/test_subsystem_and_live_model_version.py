import pytest
from fastapi.testclient import TestClient
from src.api.server import app, get_mlops_manager
from src.common.utils import get_root_dir, load_yaml

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)
root_dir = get_root_dir()

def test_suspected_subsystem_and_live_model_version():
    subset = "FD001"
    resp = client.get(f"/api/v1/fleet?subset={subset}")
    assert resp.status_code == 200
    data = resp.json()

    # 1. live_model_version must be derived from registry active_champion
    mgr = get_mlops_manager(subset)
    reg = mgr.get_registry()
    champ_ver = reg.get("active_champion", "v1.0")
    expected_version = f"rul-{subset.lower()}@{champ_ver}"
    assert data["kpis"]["live_model_version"] == expected_version

    # 2. suspected_subsystem must be a genuine turbofan subsystem, not a raw sensor id like "s11"
    valid_subsystems = {"HPC", "LPC", "LPT", "HPT", "Fan", "Combustor", "Bleed/Cooling", "Core", "Bleed"}
    for eng in data["engines"][:10]:
        subsystem = eng["suspected_subsystem"]
        assert not subsystem.startswith("s"), f"suspected_subsystem '{subsystem}' looks like a raw sensor id"
        assert subsystem in valid_subsystems

def test_engine_detail_sensor_attributions():
    subset = "FD001"
    resp = client.get(f"/api/v1/engines/FD001-001?subset={subset}")
    assert resp.status_code == 200
    data = resp.json()

    assert "sensor_attributions" in data
    assert len(data["sensor_attributions"]) > 0
    valid_subsystems = {"HPC", "LPC", "LPT", "HPT", "Fan", "Combustor", "Bleed/Cooling", "Core", "Bleed"}
    for attr in data["sensor_attributions"]:
        assert "sensor" in attr
        assert "subsystem" in attr
        assert attr["subsystem"] in valid_subsystems
        assert "contribution" in attr
        assert 0.0 <= attr["contribution"] <= 1.0
