import sqlite3
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from src.common.utils import get_root_dir

client = TestClient(app)

def test_routes_unique_method_and_path():
    """Assert unique (method, path) per route across entire FastAPI app."""
    routes = []
    for r in app.routes:
        methods = getattr(r, "methods", None)
        path = getattr(r, "path", None)
        if methods and path:
            for m in methods:
                # Exclude HEAD auto-added by starlette for GET endpoints
                if m == "HEAD":
                    continue
                routes.append((m, path))
    duplicates = [item for item in routes if routes.count(item) > 1]
    assert len(duplicates) == 0, f"Duplicate routes found: {set(duplicates)}"

def test_scoring_creates_drift_runs_row_and_get_only_reads():
    """Assert that a scoring run creates a drift_runs row, while GET only reads."""
    root_dir = get_root_dir()
    db_path = root_dir / "artifacts" / "drift_monitoring.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_drift_runs_count():
        if not db_path.exists():
            return 0
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='drift_runs'")
            if cur.fetchone()[0] == 0:
                return 0
            cur.execute("SELECT count(*) FROM drift_runs")
            return cur.fetchone()[0]
        finally:
            conn.close()

    count_before = get_drift_runs_count()

    # 1. GET /api/v1/monitoring/drift must only read and NOT create a row
    resp_get = client.get("/api/v1/monitoring/drift?subset=FD001")
    assert resp_get.status_code == 200
    count_after_get = get_drift_runs_count()
    assert count_after_get == count_before, "GET endpoint must not persist drift runs"

    # 2. POST /api/v1/predict/batch (scoring run) MUST create a drift_runs row
    test_parquet = root_dir / "data" / "processed" / "FD001" / "test.parquet"
    if not test_parquet.exists():
        pytest.skip("FD001 processed test.parquet not found")

    with open(test_parquet, "rb") as f:
        resp_post = client.post(
            "/api/v1/predict/batch?subset=FD001",
            files={"file": ("batch.parquet", f, "application/octet-stream")}
        )
    assert resp_post.status_code == 200
    count_after_score = get_drift_runs_count()
    assert count_after_score > count_before, (
        f"Expected drift_runs row count to increase after scoring, before={count_before}, after={count_after_score}"
    )


def test_first_start_seeds_empty_work_orders_db(tmp_path):
    """A fresh install seeds an empty work_orders.db schema; no state is shipped."""
    from src.agent.triage_agent import TriageAgent

    db_file = tmp_path / "artifacts" / "sqlite" / "work_orders.db"
    assert not db_file.exists()

    TriageAgent(db_path=db_file)

    assert db_file.exists(), "First start must create work_orders.db"
    conn = sqlite3.connect(str(db_file))
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM work_orders")
        assert cur.fetchone()[0] == 0, "Seeded work_orders table must start empty"
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='audit_events'")
        assert cur.fetchone()[0] == 1, "First start must create the audit_events table"
    finally:
        conn.close()
