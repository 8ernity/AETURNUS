import pytest
import io
from pathlib import Path
from src.agent.triage_agent import TriageAgent
from src.agent.aerospace_pdm import (
    calculate_throttle_derating,
    generate_parts_bom,
    get_next_lifecycle_stage,
    is_valid_lifecycle_transition,
    generate_airworthiness_pdf,
    COMPONENT_CATALOG,
    LIFECYCLE_STAGES
)
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

@pytest.fixture
def agent(tmp_path):
    db_file = tmp_path / "test_aerospace_pdm.db"
    return TriageAgent(db_path=db_file)

def test_engine_derating_critical_vs_nominal():
    # Critical RUL <= 15
    derate_crit = calculate_throttle_derating(conservative_rul=11.2, severity="CRITICAL", subsystem="HPC")
    assert derate_crit["derate_recommended"] is True
    assert 4.0 <= derate_crit["derate_percentage"] <= 7.0
    assert 15 <= derate_crit["life_extension_cycles"] <= 20
    assert derate_crit["effective_rul_with_derate"] >= 11.2 + 15
    assert "FADEC" in derate_crit["fadec_mode"]
    assert "JFK" in derate_crit["mission_replanning"]["target_hub"]

    # Nominal RUL > 15
    derate_nom = calculate_throttle_derating(conservative_rul=55.0, severity="MEDIUM", subsystem="HPC")
    assert derate_nom["derate_recommended"] is False
    assert derate_nom["derate_percentage"] == 0.0
    assert derate_nom["life_extension_cycles"] == 0

def test_parts_requisition_and_bom_mapping():
    # Sensor 9 (Core speed) -> HPC Blades
    bom_s9 = generate_parts_bom("WO-TEST-0001", ["s9"], "HPC")
    assert bom_s9["bom_id"].startswith("BOM-")
    pns = [item["part_number"] for item in bom_s9["items"]]
    assert "HPC-BLD-8821" in pns
    assert bom_s9["total_estimated_cost_usd"] > 4000.0

    # Sensor 14 (Corrected core speed) -> Core Roller Bearing
    bom_s14 = generate_parts_bom("WO-TEST-0002", ["s14"], "HPC")
    pns_14 = [item["part_number"] for item in bom_s14["items"]]
    assert "BRG-COR-4402" in pns_14

    # Default fallback
    bom_def = generate_parts_bom("WO-TEST-0003", ["s99"], "Fan")
    assert len(bom_def["items"]) >= 1

def test_multistage_lifecycle_progression(agent):
    pred = {
        "engine_id": "ENG-AERO-001",
        "unit": 1,
        "subset": "FD001",
        "cycle": 100,
        "conservative_rul": 12.0,
        "rul_p10": 10.0,
        "rul_p50": 14.0,
        "rul_p90": 20.0,
        "fail_prob_30": 0.88,
        "anomaly_score": 0.82,
        "top_sensors": ["s9", "s14"],
        "model_version": "rul-fd001@champion"
    }
    order, _ = agent.evaluate_engine_cycle(pred)
    assert order is not None
    wo_id = order["work_order_id"]
    assert order["status"] == "PROPOSED"
    assert "derate_plan" in order and order["derate_plan"]["derate_recommended"] is True
    assert "parts_bom" in order and len(order["parts_bom"]["items"]) >= 1

    # Stage 1: Advance to DISPATCHED
    res1 = agent.human_action(wo_id, "ADVANCE", reason="Crew dispatched to flight line")
    assert res1 is True
    assert agent.open_orders["ENG-AERO-001"]["status"] == "DISPATCHED"

    # Stage 2: Advance to PARTS_ORDERED
    res2 = agent.human_action(wo_id, "ADVANCE", reason="Requisitioned HPC titanium blades")
    assert res2 is True
    assert agent.open_orders["ENG-AERO-001"]["status"] == "PARTS_ORDERED"

    # Stage 3: Advance to IN_PROGRESS
    res3 = agent.human_action(wo_id, "ADVANCE", reason="Technicians on wing")
    assert res3 is True
    assert agent.open_orders["ENG-AERO-001"]["status"] == "IN_PROGRESS"

    # Apply De-rate
    res_derate = agent.human_action(wo_id, "APPLY_DERATE", reason="Pilot accepted FADEC D1 derate")
    assert res_derate is True
    assert agent.open_orders["ENG-AERO-001"]["derate_plan"]["derate_applied"] is True

    # Stage 4: Advance to POST_MAINTENANCE_RUNUP
    res4 = agent.human_action(wo_id, "ADVANCE", reason="Ground test cell runup nominal")
    assert res4 is True
    assert agent.open_orders["ENG-AERO-001"]["status"] == "POST_MAINTENANCE_RUNUP"

    # Stage 5: Advance to AIRWORTHY_CLOSED
    res5 = agent.human_action(wo_id, "ADVANCE", reason="Signed off FAA Form 8130-3 release")
    assert res5 is True
    assert "ENG-AERO-001" not in agent.open_orders
    closed_orders = agent.closed_orders_history["ENG-AERO-001"]
    assert closed_orders[-1]["status"] == "AIRWORTHY_CLOSED"

def test_airworthiness_pdf_generator():
    wo = {
        "work_order_id": "WO-CERT-8899",
        "engine_id": "ENG-FD001-019",
        "current_cycle": 145,
        "created_at_cycle": 140,
        "severity": "CRITICAL",
        "status": "PARTS_ORDERED",
        "suspected_subsystem": "HPC",
        "conservative_rul": 13.0,
        "rul_p10": 10.5,
        "rul_p50": 13.5,
        "rul_p90": 19.0,
        "top_sensors": ["s9", "s14"],
        "derate_plan": calculate_throttle_derating(13.0, "CRITICAL", "HPC"),
        "parts_bom": generate_parts_bom("WO-CERT-8899", ["s9", "s14"], "HPC"),
        "events": [
            {"cycle": 140, "actor": "TriageAgent", "event": "Auto-created in PROPOSED", "reason": "RUL <= 15"},
            {"cycle": 142, "actor": "OpsDispatcher", "event": "Advanced to DISPATCHED", "reason": "Hangar assigned"},
            {"cycle": 145, "actor": "SparesLogistics", "event": "Advanced to PARTS_ORDERED", "reason": "Blades staged"}
        ]
    }
    pdf_bytes = generate_airworthiness_pdf("ENG-FD001-019", wo)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000
    assert pdf_bytes.startswith(b"%PDF")

def test_api_advance_derate_and_pdf_endpoints():
    # Fetch work orders
    res = client.get("/api/v1/work-orders")
    assert res.status_code == 200
    orders = res.json().get("work_orders", [])
    
    # If no work order exists, trigger one via evaluation
    if not orders:
        # Evaluate engine to create order
        client.get("/api/v1/fleet?subset=FD001")
        res = client.get("/api/v1/work-orders")
        orders = res.json().get("work_orders", [])

    if orders:
        wo_id = orders[0]["work_order_id"]
        # Advance stage
        adv_res = client.post(f"/api/v1/work-orders/{wo_id}/advance?reason=CI_Test_Progression")
        assert adv_res.status_code == 200

        # Apply derate
        derate_res = client.post(f"/api/v1/work-orders/{wo_id}/derate")
        assert derate_res.status_code == 200

        # Download PDF by WO
        pdf_res = client.get(f"/api/v1/work-orders/{wo_id}/airworthiness-report.pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert len(pdf_res.content) > 5000

    # Download PDF by engine
    eng_pdf_res = client.get("/api/v1/engines/ENG-FD001-001/airworthiness-report.pdf")
    assert eng_pdf_res.status_code == 200
    assert eng_pdf_res.headers["content-type"] == "application/pdf"
    assert len(eng_pdf_res.content) > 5000
