import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import src.api.server as server
from src.mlops.drift_and_retrain import RetrainJobStore
from tests.auth_helpers import ADMIN_HEADERS


def test_retrain_job_store_releases_lock_on_terminal_states(tmp_path):
    store = RetrainJobStore(tmp_path / "retrain_jobs.db")

    first = store.start_job("FD001", "B", "operator")
    assert first is not None
    assert first["status"] == "RUNNING"
    assert store.start_job("FD001", "A", "operator") is None

    interrupted = store.finish_job(
        first["job_id"], "INTERRUPTED", error="simulated worker interruption"
    )
    assert interrupted["status"] == "INTERRUPTED"

    retry = store.start_job("FD001", "B", "operator")
    assert retry is not None
    assert retry["job_id"] != first["job_id"]

    completed = store.finish_job(
        retry["job_id"], "COMPLETED", result={"gate_passed": False}
    )
    assert completed["status"] == "COMPLETED"
    assert completed["result"] == {"gate_passed": False}

    # Different subsets have independent single-flight slots.
    other_subset = store.start_job("FD002", "B", "operator")
    assert other_subset is not None


def test_concurrent_retrain_requests_return_409(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingManager:
        def run_retrain_demo(self, scenario, replay_window_df, progress_callback):
            started.set()
            assert release.wait(timeout=10), "test did not release the retrain worker"
            return {
                "gate_passed": False,
                "champion_metrics": {},
                "challenger_metrics": {},
                "gate_reasons": ["FAIL: simulated gate"],
            }

    monkeypatch.setattr(server, "retrain_jobs", RetrainJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(server, "get_mlops_manager", lambda subset: BlockingManager())

    url = "/api/v1/mlops/retrain?subset=FD001&scenario=B"

    def request():
        return TestClient(server.app, headers=ADMIN_HEADERS).post(url)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(request)
        assert started.wait(timeout=10)
        second_future = pool.submit(request)
        second = second_future.result(timeout=10)
        assert second.status_code == 409
        assert "already running" in second.json()["detail"]
        release.set()
        first = first_future.result(timeout=20)

    assert first.status_code == 200
    jobs = server.retrain_jobs.list_jobs(subset="FD001")
    assert jobs[0]["status"] == "COMPLETED"


def test_failed_retrain_persists_failure_and_releases_lock(monkeypatch, tmp_path):
    class FailingManager:
        def run_retrain_demo(self, scenario, replay_window_df, progress_callback):
            raise RuntimeError("simulated worker interruption")

    monkeypatch.setattr(server, "retrain_jobs", RetrainJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(server, "get_mlops_manager", lambda subset: FailingManager())

    response = TestClient(server.app, headers=ADMIN_HEADERS).post(
        "/api/v1/mlops/retrain?subset=FD001&scenario=A"
    )
    assert response.status_code == 500
    assert "simulated worker interruption" in response.json()["detail"]

    job = server.retrain_jobs.latest_job("FD001")
    assert job["status"] == "FAILED"
    assert "simulated worker interruption" in job["error"]

    # A terminal FAILED row must not hold the single-flight lock.
    replacement = server.retrain_jobs.start_job("FD001", "A", "operator")
    assert replacement is not None
    server.retrain_jobs.finish_job(replacement["job_id"], "INTERRUPTED")
