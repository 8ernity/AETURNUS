import os
import json
import uuid
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.common.utils import get_root_dir, load_yaml

SUBSYSTEM_PARTS_CATALOG = {
    "HPC": {
        "part_name": "High-Pressure Compressor Stage 4-9 Rotor Blisk",
        "part_number": "P/N: CFM56-HPC-BLISK-4421",
        "lead_time_hours": 18,
        "recommended_facility": "AeroTurbine Service Depot Hub A"
    },
    "HPT": {
        "part_name": "High-Pressure Turbine Single-Crystal Vane Set",
        "part_number": "P/N: CFM56-HPT-BLADE-8812",
        "lead_time_hours": 12,
        "recommended_facility": "AeroTurbine Overhaul Center B"
    },
    "Combustor": {
        "part_name": "Low-Emission Swirler Fuel Nozzle Assembly",
        "part_number": "P/N: CFM56-CMB-NOZZLE-9014",
        "lead_time_hours": 6,
        "recommended_facility": "Fleet Line Maintenance Bay 3"
    },
    "LPT": {
        "part_name": "Low-Pressure Turbine Rotor Stage 2 Disk Assembly",
        "part_number": "P/N: CFM56-LPT-DISK-3041",
        "lead_time_hours": 24,
        "recommended_facility": "AeroTurbine Overhaul Center B"
    },
    "Fan": {
        "part_name": "Titanium Wide-Chord Fan Blade Pair",
        "part_number": "P/N: CFM56-FAN-BLD-1008",
        "lead_time_hours": 8,
        "recommended_facility": "Line Maintenance Quick-Turn Bay 1"
    }
}

class SelfHealingEngine:
    def __init__(self, db_path: Optional[Path] = None):
        self.root_dir = get_root_dir()
        self.db_path = db_path or (self.root_dir / "data" / "work_orders.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS self_healing_actions (
                    action_id TEXT PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    derate_pct REAL NOT NULL,
                    life_extension_pct REAL NOT NULL,
                    original_rul REAL NOT NULL,
                    extended_rul REAL NOT NULL,
                    status TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    logistics_reservation TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sh_engine ON self_healing_actions(engine_id);
                CREATE INDEX IF NOT EXISTS idx_sh_status ON self_healing_actions(status);
            """)

    def generate_prescriptions(self, engine_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synthesizes automated cyber-physical self-healing prescriptions
        based on active RUL, failure probability, and subsystem degradation.
        """
        engine_id = str(engine_status.get("engine_id", "unknown"))
        cycle = int(engine_status.get("cycle", 1))
        cons_rul = float(engine_status.get("conservative_rul", 50.0))
        severity = str(engine_status.get("severity", "MEDIUM")).upper()
        subsystem = str(engine_status.get("suspected_subsystem", "HPC"))

        prescriptions = []

        # 1. Operational Dynamic Derating
        derate_pct = 0.0
        life_extension = 0.0
        if severity == "CRITICAL":
            derate_pct = 20.0
            life_extension = 38.0
        elif severity == "HIGH":
            derate_pct = 12.0
            life_extension = 24.0
        elif severity == "MEDIUM":
            derate_pct = 6.0
            life_extension = 14.0
        else:
            derate_pct = 3.0
            life_extension = 8.0

        extended_rul = round(cons_rul * (1.0 + life_extension / 100.0), 1)

        prescriptions.append({
            "prescription_id": f"RX-DERATE-{engine_id}-{cycle}",
            "action_type": "DYNAMIC_DERATE",
            "title": f"Dynamic Operational Derating (-{derate_pct:.0f}%)",
            "description": f"Reduce maximum continuous thrust rating by {derate_pct:.0f}% to de-stress {subsystem}. Lowers peak EGT thermal load by 35°C.",
            "derate_pct": derate_pct,
            "life_extension_pct": life_extension,
            "original_rul": cons_rul,
            "extended_rul": extended_rul,
            "subsystem": subsystem,
            "safety_level": "AUTONOMOUS_APPROVED",
            "parameters": {
                "thrust_limit_ratio": round(1.0 - derate_pct / 100.0, 3),
                "egt_margin_gain_c": 35.0,
                "fan_speed_n1_trim_pct": -derate_pct * 0.4
            }
        })

        # 2. Bleed Valve & Active Thermal Cooling Actuation
        prescriptions.append({
            "prescription_id": f"RX-BLEED-{engine_id}-{cycle}",
            "action_type": "COOLING_BLEED_MODULATION",
            "title": f"Active {subsystem} Clearance Bleed Actuation",
            "description": "Increase compressor variable stator vane clearance cooling by +15%. Dampens high-frequency thermal stress wave.",
            "derate_pct": 2.5,
            "life_extension_pct": 12.0,
            "original_rul": cons_rul,
            "extended_rul": round(cons_rul * 1.12, 1),
            "subsystem": subsystem,
            "safety_level": "AUTONOMOUS_APPROVED",
            "parameters": {
                "bleed_valve_offset_pct": 15.0,
                "vane_angle_trim_deg": -1.8,
                "acoustic_flutter_suppression": True
            }
        })

        # 3. Supply Chain Parts Auto-Reservation
        part_info = SUBSYSTEM_PARTS_CATALOG.get(subsystem, SUBSYSTEM_PARTS_CATALOG["HPC"])
        prescriptions.append({
            "prescription_id": f"RX-PARTS-{engine_id}-{cycle}",
            "action_type": "LOGISTICS_PARTS_RESERVATION",
            "title": f"Autonomous Spare Reservation ({part_info['part_name']})",
            "description": f"Pre-allocate certified {part_info['part_number']} at {part_info['recommended_facility']} for zero-AOG turnaround.",
            "derate_pct": 0.0,
            "life_extension_pct": 0.0,
            "original_rul": cons_rul,
            "extended_rul": cons_rul,
            "subsystem": subsystem,
            "safety_level": "AUTONOMOUS_APPROVED",
            "logistics_details": part_info
        })

        return prescriptions

    def execute_action(
        self,
        engine_id: str,
        action_type: str,
        cycle: int,
        cons_rul: float,
        subsystem: str = "HPC",
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a self-healing action, records in the immutable ledger,
        and computes the resulting life extension.
        """
        status_dict = {
            "engine_id": engine_id,
            "cycle": cycle,
            "conservative_rul": cons_rul,
            "suspected_subsystem": subsystem,
            "severity": "CRITICAL" if cons_rul <= 15 else ("HIGH" if cons_rul <= 30 else "MEDIUM")
        }
        candidates = self.generate_prescriptions(status_dict)
        chosen = next((c for c in candidates if c["action_type"] == action_type), candidates[0])

        action_id = f"ACT-{uuid.uuid4()}"
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        params = chosen.get("parameters", custom_params or {})
        logistics = chosen.get("logistics_details", SUBSYSTEM_PARTS_CATALOG.get(subsystem))

        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO self_healing_actions (
                    action_id, engine_id, cycle, action_type, subsystem,
                    derate_pct, life_extension_pct, original_rul, extended_rul,
                    status, parameters, logistics_reservation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_id,
                engine_id,
                int(cycle),
                chosen["action_type"],
                subsystem,
                float(chosen["derate_pct"]),
                float(chosen["life_extension_pct"]),
                float(chosen["original_rul"]),
                float(chosen["extended_rul"]),
                "ACTIVE",
                json.dumps(params),
                json.dumps(logistics),
                now
            ))

            # Also log audit event in audit_events table
            event_id = f"EVT-{uuid.uuid4()}"
            conn.execute("""
                INSERT INTO audit_events (
                    event_id, work_order_id, engine_id, cycle, action, actor, reason, trace, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                None,
                engine_id,
                int(cycle),
                f"SELF_HEAL_{chosen['action_type']}",
                "SelfHealingEngine",
                f"Applied {chosen['action_type']} on {subsystem}: extended RUL from {chosen['original_rul']} to {chosen['extended_rul']} cycles (+{chosen['life_extension_pct']:.0f}%).",
                json.dumps([f"derate:{chosen['derate_pct']}%", f"extended_rul:{chosen['extended_rul']}"]),
                now
            ))

        return {
            "action_id": action_id,
            "engine_id": engine_id,
            "cycle": cycle,
            "action_type": chosen["action_type"],
            "title": chosen["title"],
            "subsystem": subsystem,
            "original_rul": chosen["original_rul"],
            "extended_rul": chosen["extended_rul"],
            "life_extension_pct": chosen["life_extension_pct"],
            "derate_pct": chosen["derate_pct"],
            "status": "ACTIVE",
            "parameters": params,
            "logistics": logistics,
            "timestamp": now
        }

    def get_history(self, engine_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_db() as conn:
            if engine_id:
                rows = conn.execute(
                    "SELECT * FROM self_healing_actions WHERE engine_id = ? ORDER BY cycle DESC, created_at DESC LIMIT ?",
                    (engine_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM self_healing_actions ORDER BY cycle DESC, created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            history = []
            for r in rows:
                item = dict(r)
                item["parameters"] = json.loads(item["parameters"])
                item["logistics_reservation"] = json.loads(item["logistics_reservation"]) if item["logistics_reservation"] else None
                history.append(item)
            return history
