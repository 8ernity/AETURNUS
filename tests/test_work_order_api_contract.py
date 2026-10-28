from fastapi.testclient import TestClient

import src.api.server as server
from src.agent.triage_agent import TriageAgent, get_next_lifecycle_stage


from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(server.app, headers=ADMIN_HEADERS)


def _seed_work_order(monkeypatch) -> tuple[TriageAgent, dict]:
    fresh_agent = TriageAgent(db_path=":memory:")
    monkeypatch.setattr(server, "triage_agent", fresh_agent)

    pred = {
        "engine_id": "ENG-FD001-API-CONTRACT",
        "unit": 1,
        "subset": "FD001",
        "cycle": 50,
        "conservative_rul": 12.0,
        "rul_p10": 10.0,
        "rul_p50": 15.0,
        "rul_p90": 22.0,
        "fail_prob_30": 0.88,
        "anomaly_score": 0.75,
        "model_version": "rul-fd001@champion",
    }
    order, _ = fresh_agent.evaluate_engine_cycle(pred)
    assert order is not None, "Expected a work order for a critical prediction"
    return fresh_agent, order


def test_human_action_returns_updated_work_order(monkeypatch):
    fresh_agent, order = _seed_work_order(monkeypatch)

    response = client.post(
        f"/api/v1/work-orders/{order['work_order_id']}/action",
        json={"action": "APPROVE", "reason": "Approved by API contract test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["work_order_id"] == order["work_order_id"]
    assert payload["work_order"]["status"] == "APPROVED"

    with fresh_agent._get_db() as conn:
        audit = conn.execute(
            "SELECT reason, actor FROM audit_events WHERE work_order_id = ? AND action = 'APPROVE'",
            (order["work_order_id"],),
        ).fetchone()
    assert audit is not None
    assert audit["reason"] == "Approved by API contract test"
    assert audit["actor"] == "test-admin"


def test_advance_accepts_note_json_and_returns_updated_work_order(monkeypatch):
    fresh_agent, order = _seed_work_order(monkeypatch)
    expected_status = get_next_lifecycle_stage(order["status"])

    response = client.post(
        f"/api/v1/work-orders/{order['work_order_id']}/advance",
        json={"note": "Advanced by API contract test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["work_order_id"] == order["work_order_id"]
    assert payload["work_order"]["status"] == expected_status

    with fresh_agent._get_db() as conn:
        audit = conn.execute(
            "SELECT reason, actor FROM audit_events WHERE work_order_id = ? AND action = ?",
            (order["work_order_id"], f"STAGE_{expected_status}"),
        ).fetchone()
    assert audit is not None
    assert audit["reason"] == "Advanced by API contract test"
    assert audit["actor"] == "test-admin"


def test_failed_action_returns_server_detail(monkeypatch):
    _seed_work_order(monkeypatch)

    response = client.post(
        "/api/v1/work-orders/WO-DOES-NOT-EXIST/action",
        json={"action": "APPROVE", "reason": "Missing order"},
    )

    assert response.status_code == 400
    assert "Unable to apply action" in response.json()["detail"]


def test_unknown_action_returns_failure(monkeypatch):
    _seed_work_order(monkeypatch)

    response = client.post(
        "/api/v1/work-orders/WO-DOES-NOT-EXIST/advance",
        json={"note": "Missing order"},
    )

    assert response.status_code == 400
    assert "Unable to advance" in response.json()["detail"]
