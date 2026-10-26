"""Create a small, deterministic, non-alarming demo state.

The seed intentionally leaves zero open work orders.  It writes two historical
closed orders so the demo UI has an auditable example, but no active queue or
drift rows that could fire alerts on a fresh checkout.

Usage:
    python scripts/seed_demo_state.py [--reset]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.triage_agent import TriageAgent
from src.common.state import get_state_dir


def _demo_order(
    work_order_id: str,
    engine_id: str,
    subset: str,
    unit: int,
    severity: str,
    p10: float,
    p50: float,
    p90: float,
) -> Dict[str, Any]:
    return {
        "work_order_id": work_order_id,
        "engine_id": engine_id,
        "unit": unit,
        "subset": subset,
        "created_at_cycle": 120,
        "current_cycle": 150,
        "closed_at_cycle": 150,
        "severity": severity,
        "raw_severity": severity,
        "status": "CLOSED",
        "conservative_rul": p10,
        "predicted_rul": {"p10": p10, "p50": p50, "p90": p90},
        "fail_prob_30": 0.12,
        "anomaly_score": 0.18,
        "due_by_cycle": 160,
        "suspected_subsystem": "HPC",
        "top_sensors": ["s3", "s7", "s9"],
        "recommended_action": "Historical demo: inspection completed and closed.",
        "decision_trace": ["demo:seeded_historical_order", "status:CLOSED"],
        "model_version": "demo-state-v1",
        "rules_version": "demo-state-v1",
        "exception": None,
        "events": [
            {"cycle": 120, "actor": "DemoSeeder", "event": "Historical demo order created"},
            {"cycle": 150, "actor": "DemoSeeder", "event": "Historical demo order closed"},
        ],
    }


def seed_demo_state(reset: bool = False) -> Dict[str, Any]:
    state_dir = get_state_dir()
    if reset:
        for path in state_dir.glob("*.db*"):
            if path.is_file():
                path.unlink()
        shutil.rmtree(state_dir / "mlruns", ignore_errors=True)

    agent = TriageAgent()
    with agent._get_db() as conn:
        conn.execute("DELETE FROM work_orders")
        conn.execute("DELETE FROM audit_events")
    # Drop any in-memory orders loaded before the reset so the seeded DB and the
    # reported active queue cannot diverge.
    agent.open_orders.clear()

    # Seeded demo state must not inherit stale drift rows that could fire alerts.
    for suffix in ("", "-wal", "-shm", "-journal"):
        drift_db = state_dir / f"drift_monitoring.db{suffix}"
        if drift_db.exists():
            drift_db.unlink()

    orders: List[Dict[str, Any]] = [
        _demo_order("WO-DEMO-0001", "ENG-FD001-0001", "FD001", 1, "MEDIUM", 28.0, 41.0, 55.0),
        _demo_order("WO-DEMO-0002", "ENG-FD002-0042", "FD002", 42, "HIGH", 17.0, 24.0, 33.0),
    ]
    for order in orders:
        agent._persist_order(order)

    return {
        "state_dir": str(state_dir),
        "open_work_orders": len(agent.get_active_work_orders()),
        "seeded_closed_orders": [o["work_order_id"] for o in orders],
        "reset": reset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Delete existing state DBs before seeding")
    args = parser.parse_args()
    result = seed_demo_state(reset=args.reset)
    print(
        f"Demo state ready at {result['state_dir']}: "
        f"{result['open_work_orders']} open work orders, "
        f"{len(result['seeded_closed_orders'])} historical closed orders."
    )


if __name__ == "__main__":
    main()
