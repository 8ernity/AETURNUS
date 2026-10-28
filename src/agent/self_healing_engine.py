import os
import json
import uuid
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.common.utils import get_root_dir, load_yaml
from src.common.state import state_db_path

class SelfHealingEngine:
    def __init__(self, db_path: Optional[Path] = None):
        self.root_dir = get_root_dir()
        self.rules_cfg = load_yaml("configs/agent_rules.yaml")
        self.sensor_cfg = load_yaml("configs/sensor_map.yaml")
        self.db_path = db_path or state_db_path("self_healing.db", self.root_dir)
        self.is_memory_db = str(self.db_path) == ":memory:"
        if not self.is_memory_db:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        else:
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_db(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id TEXT PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    unit INTEGER NOT NULL,
                    subset TEXT NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    current_cycle INTEGER NOT NULL,
                    closed_at_cycle INTEGER,
                    severity TEXT NOT NULL,
                    raw_severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    conservative_rul REAL NOT NULL,
                    rul_p10 REAL NOT NULL,
                    rul_p50 REAL NOT NULL,
                    rul_p90 REAL NOT NULL,
                    fail_prob_30 REAL NOT NULL,
                    anomaly_score REAL NOT NULL,
                    due_by_cycle INTEGER NOT NULL,
                    suspected_subsystem TEXT NOT NULL,
                    top_sensors TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    decision_trace TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    rules_version TEXT NOT NULL,
                    exception TEXT,
                    events TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS self_healing_actions (
                    action_id TEXT PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    work_order_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    completion_note TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    work_order_id TEXT,
                    engine_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT,
                    trace TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sh_engine ON self_healing_actions(engine_id);
                CREATE INDEX IF NOT EXISTS idx_sh_status ON self_healing_actions(status);
            """)
            # Migrate pre-existing ledgers that predate external completion
            # tracking. SQLite cannot add multiple columns in one statement.
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(self_healing_actions)"
                ).fetchall()
            }
            for column, ddl in (
                ("work_order_id", "TEXT"),
                ("completed_at", "TEXT"),
                ("completion_note", "TEXT"),
            ):
                if column not in columns:
                    conn.execute(
                        f"ALTER TABLE self_healing_actions ADD COLUMN {column} {ddl}"
                    )

    def _severity_bands(self) -> Dict[str, float]:
        """Severity thresholds are policy: every band must be present in config."""
        sev_bands = self.rules_cfg.get("severity_bands")
        if not isinstance(sev_bands, dict):
            raise ValueError(
                "configs/agent_rules.yaml must define severity_bands with "
                "critical/high/medium/low thresholds; numeric defaults are not allowed."
            )
        missing = [key for key in ("critical", "high", "medium", "low") if key not in sev_bands]
        if missing:
            raise ValueError(f"configs/agent_rules.yaml severity_bands missing keys: {missing}")
        return {key: float(sev_bands[key]) for key in ("critical", "high", "medium", "low")}

    def get_base_severity(self, cons_rul: float) -> str:
        bands = self._severity_bands()
        if cons_rul <= bands["critical"]:
            return "CRITICAL"
        if cons_rul <= bands["high"]:
            return "HIGH"
        if cons_rul <= bands["medium"]:
            return "MEDIUM"
        if cons_rul <= bands["low"]:
            return "LOW"
        return "HEALTHY"

    def generate_prescriptions(self, engine_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synthesizes rule-based recommended maintenance actions based on
        active RUL, severity, and suspected subsystem degradation.
        Contains no invented numbers or synthetic physics parameters.
        """
        engine_id = str(engine_status.get("engine_id", "unknown"))
        cycle = int(engine_status["cycle"])
        cons_rul = float(engine_status["conservative_rul"])
        severity = str(engine_status.get("severity", "MEDIUM")).upper()
        subsystem = str(engine_status.get("suspected_subsystem", "Unknown"))

        rule_actions = {
            "CRITICAL": f"Immediate shutdown and remove from service. Perform emergent overhaul of {subsystem}.",
            "HIGH": f"Schedule urgent {subsystem} borescope inspection at next turnaround window.",
            "MEDIUM": f"Plan non-destructive evaluation of {subsystem} during upcoming scheduled maintenance.",
            "LOW": f"Continue standard monitoring and schedule routine {subsystem} check.",
            "HEALTHY": f"Nominal operations. Routine {subsystem} telemetry monitoring."
        }
        primary_action = rule_actions.get(severity, rule_actions["MEDIUM"])
        prescriptions = [
            {
                "prescription_id": f"RX-MAINT-{engine_id}-{cycle}",
                "action_type": "MAINTENANCE_INSPECTION",
                "title": f"Rule-Based {subsystem} Maintenance Order",
                "description": primary_action,
                "subsystem": subsystem,
                "severity": severity,
                "conservative_rul": cons_rul,
                "status": "RECOMMENDED"
            },
            {
                "prescription_id": f"RX-LOGISTICS-{engine_id}-{cycle}",
                "action_type": "LOGISTICS_DISPATCH",
                "title": f"Queue {subsystem} Inspection Kit",
                "description": f"Stage standard inspection tooling and calibration equipment for {subsystem}.",
                "subsystem": subsystem,
                "severity": severity,
                "conservative_rul": cons_rul,
                "status": "RECOMMENDED"
            },
            {
                "prescription_id": f"RX-TELEMETRY-{engine_id}-{cycle}",
                "action_type": "TELEMETRY_CALIBRATION",
                "title": f"Enhanced {subsystem} Telemetry Capture",
                "description": f"Increase telemetry sampling frequency on {subsystem} sensor channels for granular degradation tracking.",
                "subsystem": subsystem,
                "severity": severity,
                "conservative_rul": cons_rul,
                "status": "RECOMMENDED"
            }
        ]
        return prescriptions

    def execute_action(
        self,
        engine_id: str,
        action_type: str,
        cycle: int,
        cons_rul: float,
        subsystem: str = "HPC",
        custom_params: Optional[Dict[str, Any]] = None,
        actor: str = "SelfHealingEngine",
        work_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Records a rule-based maintenance *recommendation* in the immutable
        ledger. The action remains RECOMMENDED until an external completion is
        recorded through complete_action(); this method never claims that
        physical work was executed.
        """
        status_dict = {
            "engine_id": engine_id,
            "cycle": cycle,
            "conservative_rul": cons_rul,
            "suspected_subsystem": subsystem,
            "severity": self.get_base_severity(cons_rul)
        }
        candidates = self.generate_prescriptions(status_dict)
        chosen = next((c for c in candidates if c["action_type"] == action_type), candidates[0])

        action_id = f"ACT-{uuid.uuid4()}"
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS self_healing_actions (
                    action_id TEXT PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    subsystem TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    work_order_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    completion_note TEXT
                );
            """)
            conn.execute("""
                INSERT INTO self_healing_actions (
                    action_id, engine_id, cycle, action_type, subsystem, status,
                    recommended_action, work_order_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_id,
                engine_id,
                int(cycle),
                chosen["action_type"],
                subsystem,
                "RECOMMENDED",
                chosen["description"],
                work_order_id,
                now
            ))

            event_id = f"EVT-{uuid.uuid4()}"
            conn.execute("""
                INSERT INTO audit_events (
                    event_id, work_order_id, engine_id, cycle, action, actor, reason, trace, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                work_order_id,
                engine_id,
                int(cycle),
                f"ACTION_RECOMMENDED_{chosen['action_type']}",
                actor,
                f"Recommended rule-based action for {subsystem}: {chosen['description']}",
                json.dumps([f"action:{chosen['action_type']}", "status:RECOMMENDED"]),
                now
            ))

        return {
            "action_id": action_id,
            "engine_id": engine_id,
            "cycle": cycle,
            "action_type": chosen["action_type"],
            "title": chosen["title"],
            "subsystem": subsystem,
            "conservative_rul": cons_rul,
            "recommended_action": chosen["description"],
            "status": "RECOMMENDED",
            "work_order_id": work_order_id,
            "timestamp": now
        }

    def complete_action(
        self,
        action_id: str,
        actor: str = "ExternalSystem",
        note: str = "External completion recorded",
    ) -> Optional[Dict[str, Any]]:
        """Record an external completion against a recommendation.

        Returns ``None`` when the action does not exist. Re-completing an
        already completed action is idempotent and returns the stored row.
        """
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM self_healing_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if row is None:
                return None

            if row["status"] != "COMPLETED":
                conn.execute(
                    """
                    UPDATE self_healing_actions
                    SET status = 'COMPLETED', completed_at = ?, completion_note = ?
                    WHERE action_id = ?
                    """,
                    (now, note, action_id),
                )
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, work_order_id, engine_id, cycle, action,
                        actor, reason, trace, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"EVT-{uuid.uuid4()}",
                        row["work_order_id"],
                        row["engine_id"],
                        int(row["cycle"]),
                        "ACTION_COMPLETED",
                        actor,
                        note,
                        json.dumps([f"action_id:{action_id}", "status:COMPLETED"]),
                        now,
                    ),
                )

            completed = conn.execute(
                "SELECT * FROM self_healing_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            result = dict(completed)
            result["timestamp"] = result.get("created_at")
            result["completed_at"] = result.get("completed_at") or now
            return result

    def get_history(self, engine_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_db() as conn:
            # Check if recommended_action column exists
            columns = [col[1] for col in conn.execute("PRAGMA table_info(self_healing_actions)").fetchall()]
            if not columns:
                return []
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
                if "parameters" in item and item["parameters"]:
                    try:
                        item["parameters"] = json.loads(item["parameters"])
                    except Exception:
                        pass
                if "logistics_reservation" in item and item["logistics_reservation"]:
                    try:
                        item["logistics_reservation"] = json.loads(item["logistics_reservation"])
                    except Exception:
                        pass
                history.append(item)
            return history
