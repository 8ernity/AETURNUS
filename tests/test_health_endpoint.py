"""Unit tests for the lightweight /api/v1/health keep-alive endpoint."""

import time
from datetime import datetime
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_endpoint_public_and_schema():
    """Verify GET /api/v1/health returns 200 OK without credentials and with expected fields."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data.get("status") == "online"
    assert data.get("service") == "AETURNUS"
    assert data.get("version") == "1.0.0"
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0

    # Ensure timestamp is valid ISO format
    ts_str = data.get("timestamp")
    assert ts_str is not None
    # datetime.fromisoformat parses ISO-8601 strings
    parsed = datetime.fromisoformat(ts_str)
    assert parsed is not None


def test_health_alias_endpoint():
    """Verify GET /health alias works identically for generic monitors."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "online"
    assert data.get("service") == "AETURNUS"


def test_health_endpoint_is_lightweight():
    """Verify that /api/v1/health responds with sub-millisecond to low-millisecond latency."""
    t0 = time.perf_counter()
    response = client.get("/api/v1/health")
    duration = time.perf_counter() - t0

    assert response.status_code == 200
    # Must respond well under 100ms even in unoptimized test harness
    assert duration < 0.100, f"Health check took too long: {duration:.4f}s"
