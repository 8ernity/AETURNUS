import os
import json
import uuid
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from src.common.utils import get_root_dir, load_yaml

SEVERITY_ORDER = ["HEALTHY", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
HUMAN_STATUSES = {
    "APPROVED", "REJECTED", "OVERRIDDEN", "DISPATCHED",
    "PARTS_ORDERED", "IN_PROGRESS", "CLOSED"
}

MAINTENANCE_STAGES = [
    "PROPOSED", "APPROVED", "DISPATCHED", "PARTS_ORDERED", "IN_PROGRESS", "CLOSED"
]
STAGE_PROGRESSION = {
    "PROPOSED": "APPROVED",
    "AUTO_CREATED": "APPROVED",
    "NEEDS_REVIEW": "APPROVED",
    "APPROVED": "DISPATCHED",
    "DISPATCHED": "PARTS_ORDERED",
    "PARTS_ORDERED": "IN_PROGRESS",
    "IN_PROGRESS": "CLOSED"
}

def get_next_lifecycle_stage(current: str) -> Optional[str]:
    return STAGE_PROGRESSION.get(current)

class TriageAgent:
    def __init__(self, db_path: Optional[Path] = None, rules_cfg: Optional[Dict[str, Any]] = None):
        self.root_dir = get_root_dir()
        self.rules_cfg = rules_cfg or load_yaml("configs/agent_rules.yaml")
        self.sensor_cfg = load_yaml("configs/sensor_map.yaml")
        self.rules_version = self.rules_cfg.get("rules_version", "2026-10-07.1")
        
        # SQLite Database Path (Primary: artifacts/sqlite/work_orders.db)
        default_sqlite = self.root_dir / "artifacts" / "sqlite" / "work_orders.db"
        legacy_sqlite = self.root_dir / "data" / "work_orders.db"
        if not db_path:
            default_sqlite.parent.mkdir(parents=True, exist_ok=True)
            if not default_sqlite.exists() and legacy_sqlite.exists():
                import shutil
                try:
                    shutil.copy2(legacy_sqlite, default_sqlite)
                except Exception:
                    pass
            self.db_path = default_sqlite
        else:
            self.db_path = Path(db_path) if str(db_path) != ":memory:" else db_path

        self.is_memory_db = str(self.db_path) == ":memory:"
        if not self.is_memory_db:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        else:
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        
        # In-memory caches (synchronized with SQLite)
        self.open_orders: Dict[str, Dict[str, Any]] = {} # engine_id -> order
        self.closed_orders_history: Dict[str, List[Dict[str, Any]]] = {} # engine_id -> list of closed orders
        self.engine_state_history: Dict[str, List[Dict[str, Any]]] = {} # engine_id -> list of past evaluations
        self.engine_hysteresis: Dict[str, Dict[str, Any]] = {} # engine_id -> hysteresis tracking
        self.suppressed_engines: Dict[str, int] = {} # engine_id -> suppressed_until_cycle

        self._init_db()
        self._load_from_db()

    def get_base_severity(self, cons_rul: float) -> str:
        sev_bands = self.rules_cfg.get("severity_bands", {})
        crit_thresh = sev_bands.get("critical", 15)
        high_thresh = sev_bands.get("high", 30)
        med_thresh = sev_bands.get("medium", 60)
        low_thresh = sev_bands.get("low", 100)
        if cons_rul <= crit_thresh:
            return "CRITICAL"
        elif cons_rul <= high_thresh:
            return "HIGH"
        elif cons_rul <= med_thresh:
            return "MEDIUM"
        elif cons_rul <= low_thresh:
            return "LOW"
        return "HEALTHY"

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

                CREATE TABLE IF NOT EXISTS engine_hysteresis (
                    engine_id TEXT PRIMARY KEY,
                    held_severity TEXT NOT NULL,
                    raw_severity TEXT NOT NULL,
                    consecutive_lower_cycles INTEGER NOT NULL,
                    last_cycle INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS engine_state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    conservative_rul REAL NOT NULL,
                    raw_severity TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    exceptions TEXT,
                    recorded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_wo_engine ON work_orders(engine_id);
                CREATE INDEX IF NOT EXISTS idx_wo_status ON work_orders(status);
                CREATE INDEX IF NOT EXISTS idx_audit_engine ON audit_events(engine_id);
                CREATE INDEX IF NOT EXISTS idx_hist_engine ON engine_state_history(engine_id, cycle);
            """)

    def _load_from_db(self):
        try:
            with self._get_db() as conn:
                rows = conn.execute("SELECT * FROM work_orders").fetchall()
                for r in rows:
                    order = dict(r)
                    order["top_sensors"] = json.loads(order["top_sensors"]) if isinstance(order["top_sensors"], str) else order["top_sensors"]
                    order["decision_trace"] = json.loads(order["decision_trace"]) if isinstance(order["decision_trace"], str) else order["decision_trace"]
                    order["events"] = json.loads(order["events"]) if isinstance(order["events"], str) else order["events"]
                    order["predicted_rul"] = {
                        "p10": order["rul_p10"],
                        "p50": order["rul_p50"],
                        "p90": order["rul_p90"]
                    }

                    eid = order["engine_id"]
                    if order["status"] in ["CLOSED", "REJECTED", "CANCELLED"]:
                        if eid not in self.closed_orders_history:
                            self.closed_orders_history[eid] = []
                        self.closed_orders_history[eid].append(order)
                    else:
                        self.open_orders[eid] = order

                hyst_rows = conn.execute("SELECT * FROM engine_hysteresis").fetchall()
                for r in hyst_rows:
                    self.engine_hysteresis[r["engine_id"]] = {
                        "held_severity": r["held_severity"],
                        "raw_severity": r["raw_severity"],
                        "consecutive_lower_cycles": int(r["consecutive_lower_cycles"]),
                        "last_cycle": int(r["last_cycle"])
                    }

                hist_rows = conn.execute("SELECT * FROM engine_state_history ORDER BY engine_id, cycle ASC").fetchall()
                for r in hist_rows:
                    eid = r["engine_id"]
                    if eid not in self.engine_state_history:
                        self.engine_state_history[eid] = []
                    self.engine_state_history[eid].append({
                        "cycle": int(r["cycle"]),
                        "conservative_rul": float(r["conservative_rul"]),
                        "raw_severity": r["raw_severity"],
                        "severity": r["severity"],
                        "anomaly_score": float(r["anomaly_score"]),
                        "exceptions": json.loads(r["exceptions"]) if r["exceptions"] else []
                    })
        except Exception as e:
            print(f"Error loading work orders from SQLite: {e}")

    def _persist_hysteresis(self, engine_id: str, hyst: Dict[str, Any]):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO engine_hysteresis (
                    engine_id, held_severity, raw_severity, consecutive_lower_cycles, last_cycle, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                engine_id,
                hyst["held_severity"],
                hyst["raw_severity"],
                int(hyst["consecutive_lower_cycles"]),
                int(hyst["last_cycle"]),
                now
            ))

    def _persist_state_history_entry(self, engine_id: str, entry: Dict[str, Any]):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO engine_state_history (
                    engine_id, cycle, conservative_rul, raw_severity, severity, anomaly_score, exceptions, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                engine_id,
                int(entry["cycle"]),
                float(entry["conservative_rul"]),
                entry["raw_severity"],
                entry["severity"],
                float(entry["anomaly_score"]),
                json.dumps(entry.get("exceptions", [])),
                now
            ))

    def _persist_order(self, order: Dict[str, Any]):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        pred_rul = order.get("predicted_rul", {})
        with self._get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO work_orders (
                    work_order_id, engine_id, unit, subset, created_at_cycle, current_cycle,
                    closed_at_cycle, severity, raw_severity, status, conservative_rul,
                    rul_p10, rul_p50, rul_p90, fail_prob_30, anomaly_score,
                    due_by_cycle, suspected_subsystem, top_sensors, recommended_action,
                    decision_trace, model_version, rules_version, exception, events, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order["work_order_id"],
                order["engine_id"],
                int(order.get("unit", 1)),
                order.get("subset", "FD001"),
                int(order.get("created_at_cycle", 1)),
                int(order.get("current_cycle", order.get("created_at_cycle", 1))),
                order.get("closed_at_cycle"),
                order["severity"],
                order.get("raw_severity", order["severity"]),
                order["status"],
                float(order.get("conservative_rul", 0.0)),
                float(pred_rul.get("p10", 0.0)),
                float(pred_rul.get("p50", 0.0)),
                float(pred_rul.get("p90", 0.0)),
                float(order.get("fail_prob_30", 0.0)),
                float(order.get("anomaly_score", 0.0)),
                int(order.get("due_by_cycle", 1)),
                order.get("suspected_subsystem", "HPC"),
                json.dumps(order.get("top_sensors", [])),
                order.get("recommended_action", ""),
                json.dumps(order.get("decision_trace", [])),
                order.get("model_version") or "unknown",
                order.get("rules_version", self.rules_version),
                order.get("exception"),
                json.dumps(order.get("events", [])),
                now
            ))

        legacy_sqlite = self.root_dir / "data" / "work_orders.db"
        if not self.is_memory_db and self.db_path != legacy_sqlite:
            try:
                legacy_sqlite.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(str(legacy_sqlite)) as leg_conn:
                    leg_conn.execute("""
                        INSERT OR REPLACE INTO work_orders (
                            work_order_id, engine_id, unit, subset, created_at_cycle, current_cycle,
                            closed_at_cycle, severity, raw_severity, status, conservative_rul,
                            rul_p10, rul_p50, rul_p90, fail_prob_30, anomaly_score,
                            due_by_cycle, suspected_subsystem, top_sensors, recommended_action,
                            decision_trace, model_version, rules_version, exception, events, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        order["work_order_id"],
                        order["engine_id"],
                        int(order.get("unit", 1)),
                        order.get("subset", "FD001"),
                        int(order.get("created_at_cycle", 1)),
                        int(order.get("current_cycle", order.get("created_at_cycle", 1))),
                        order.get("closed_at_cycle"),
                        order["severity"],
                        order.get("raw_severity", order["severity"]),
                        order["status"],
                        float(order.get("conservative_rul", 0.0)),
                        float(pred_rul.get("p10", 0.0)),
                        float(pred_rul.get("p50", 0.0)),
                        float(pred_rul.get("p90", 0.0)),
                        float(order.get("fail_prob_30", 0.0)),
                        float(order.get("anomaly_score", 0.0)),
                        int(order.get("due_by_cycle", 1)),
                        order.get("suspected_subsystem", "HPC"),
                        json.dumps(order.get("top_sensors", [])),
                        order.get("recommended_action", ""),
                        json.dumps(order.get("decision_trace", [])),
                        order.get("model_version") or "unknown",
                        order.get("rules_version", self.rules_version),
                        order.get("exception"),
                        json.dumps(order.get("events", [])),
                        now
                    ))
            except Exception:
                pass

    def _log_audit_event(self, engine_id: str, action: str, actor: str, cycle: int, work_order_id: Optional[str] = None, reason: str = "", trace: Optional[List[str]] = None):
        event_id = f"EVT-{uuid.uuid4()}"
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO audit_events (
                    event_id, work_order_id, engine_id, cycle, action, actor, reason, trace, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                work_order_id,
                engine_id,
                int(cycle),
                action,
                actor,
                reason,
                json.dumps(trace or []),
                now
            ))

    def evaluate_and_persist(
        self,
        engine: Dict[str, Any],
        db_session: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Executes the 9-step decision pipeline on an engine prediction/state and guarantees
        immediate persistence into the SQLite database (artifacts/sqlite/work_orders.db).
        
        Args:
            engine: Engine prediction dictionary containing keys like engine_id, unit, cycle,
                    conservative_rul, rul_p10, fail_prob_30, anomaly_score.
            db_session: Optional database session object (e.g. SQLAlchemy session or connection).
        
        Returns:
            Tuple of (work_order, decision_trace).
        """
        if "conservative_rul" not in engine and "rul_p10" in engine:
            engine["conservative_rul"] = engine["rul_p10"]

        order, trace = self.evaluate_engine_cycle(engine)
        if order is not None:
            self._persist_order(order)
            if db_session is not None:
                try:
                    if hasattr(db_session, "commit"):
                        db_session.commit()
                except Exception:
                    pass
        return order, trace

    def evaluate_engine_cycle(
        self,
        prediction: Optional[Dict[str, Any]],
        recent_history: Optional[List[Dict[str, Any]]] = None,
        data_quality_flags: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Executes the 9-step decision pipeline:
        1. Input validation (gracefully handles None or missing keys)
        2. Conservative RUL & uncertainty extraction
        3. Degradation trend calculation
        4. Base severity classification
        5. Escalation modifiers (M1, M2, M3)
        6. Hysteresis (stores raw severity, permits de-escalation after N cycles)
        7. Independent exception checks (X1 - X7)
        8. Cooldown & suppression window verification
        9. Work order creation / refresh (UUIDs, live RUL refresh, human status preservation)
        """
        trace: List[str] = []

        # Step 1: Input Validation (handle None)
        if prediction is None:
            trace.append("X0:prediction_is_none")
            return None, trace

        if "engine_id" not in prediction or "cycle" not in prediction:
            trace.append("X0:invalid_prediction_format")
            return None, trace

        engine_id = str(prediction["engine_id"])
        cycle = int(prediction["cycle"])
        subset = str(prediction.get("subset", "FD001"))

        if engine_id not in self.engine_state_history:
            self.engine_state_history[engine_id] = []
        hist = self.engine_state_history[engine_id]

        if "conservative_rul" not in prediction:
            trace.append("X0:missing_conservative_rul->exception")
            order = self._create_exception_order(prediction, "Missing conservative_rul in prediction input", trace)
            return order, trace

        # Read thresholds dynamically from agent_rules.yaml
        sev_bands = self.rules_cfg.get("severity_bands", {})
        crit_thresh = sev_bands.get("critical", 15)
        high_thresh = sev_bands.get("high", 30)
        med_thresh = sev_bands.get("medium", 60)
        low_thresh = sev_bands.get("low", 100)

        esc_cfg = self.rules_cfg.get("escalation", {})
        drop_window = esc_cfg.get("rul_drop_window", 10)
        drop_thresh = esc_cfg.get("rul_drop_threshold", 15)
        anom_thresh = esc_cfg.get("anomaly_threshold", 0.80)
        fail_thresh = esc_cfg.get("fail_prob_threshold", 0.70)
        deescalate_cycles = esc_cfg.get("deescalate_after_cycles", 5)

        exc_cfg = self.rules_cfg.get("exceptions", {})
        max_interval_width = exc_cfg.get("max_interval_width", 40)
        max_missing_frac = exc_cfg.get("max_missing_fraction", 0.30)
        min_hist_cycles = exc_cfg.get("min_history_cycles", 15)
        data_gap_cycles = exc_cfg.get("data_gap_cycles", 5)

        life_cfg = self.rules_cfg.get("lifecycle", {})
        cooldown_cycles = life_cfg.get("cooldown_cycles", 20)
        due_by_factor = life_cfg.get("default_due_by_factor", 0.5)
        suppression_cycles = life_cfg.get("suppression_window_cycles", 10)

        # Step 2: Conservative RUL & uncertainty
        cons_rul = float(prediction["conservative_rul"])
        rul_p10 = float(prediction.get("rul_p10", cons_rul))
        rul_p90 = float(prediction.get("rul_p90", cons_rul + 20.0))
        fail_prob = float(prediction.get("fail_prob_30", 0.0))
        anomaly_score = float(prediction.get("anomaly_score", 0.0))

        # Step 3: Degradation trend
        rul_drop = 0.0
        if len(hist) >= 5:
            past_rul = hist[-min(drop_window, len(hist))]["conservative_rul"]
            rul_drop = past_rul - cons_rul

        # Step 4: Base severity assignment
        if cons_rul <= crit_thresh:
            base_severity = "CRITICAL"
        elif cons_rul <= high_thresh:
            base_severity = "HIGH"
        elif cons_rul <= med_thresh:
            base_severity = "MEDIUM"
        elif cons_rul <= low_thresh:
            base_severity = "LOW"
        else:
            base_severity = "HEALTHY"
        trace.append(f"base:{base_severity}(rul={cons_rul})")

        # Step 5: Escalation Modifiers
        sev_idx = SEVERITY_ORDER.index(base_severity)

        # M1: Rapid RUL drop
        if rul_drop >= drop_thresh:
            sev_idx = min(len(SEVERITY_ORDER) - 1, sev_idx + 1)
            trace.append(f"M1:fired(drop={rul_drop:.1f}->{SEVERITY_ORDER[sev_idx]})")
            self._dispatch_rule_webhook("M1", engine_id, cycle, SEVERITY_ORDER[sev_idx], cons_rul, {"rul_drop": round(float(rul_drop), 1), "drop_thresh": drop_thresh})
        else:
            trace.append("M1:not_fired")

        # M2: High anomaly score on Medium or lower
        if anomaly_score >= anom_thresh and sev_idx <= SEVERITY_ORDER.index("MEDIUM"):
            sev_idx = min(len(SEVERITY_ORDER) - 1, sev_idx + 1)
            trace.append(f"M2:fired(anomaly={anomaly_score:.2f}->{SEVERITY_ORDER[sev_idx]})")
            self._dispatch_rule_webhook("M2", engine_id, cycle, SEVERITY_ORDER[sev_idx], cons_rul, {"anomaly_score": round(float(anomaly_score), 2), "anom_thresh": anom_thresh})
        else:
            trace.append("M2:not_fired")

        # M3: Multi-horizon failure probability with validation-tuned threshold
        effective_tau_30 = float(prediction.get("classifier_threshold_30", fail_thresh))
        fail_prob_10 = float(prediction.get("fail_prob_10", 0.0))
        if fail_prob_10 >= 0.70:
            sev_idx = max(sev_idx, SEVERITY_ORDER.index("CRITICAL"))
            trace.append(f"M3:fired(fail_prob_10={fail_prob_10:.2f}->CRITICAL)")
            self._dispatch_rule_webhook("M3", engine_id, cycle, "CRITICAL", cons_rul, {"fail_prob_10": round(float(fail_prob_10), 2), "threshold": 0.70})
        elif fail_prob >= effective_tau_30:
            sev_idx = max(sev_idx, SEVERITY_ORDER.index("HIGH"))
            trace.append(f"M3:fired(fail_prob={fail_prob:.2f}>={effective_tau_30:.2f}->HIGH)")
            self._dispatch_rule_webhook("M3", engine_id, cycle, "HIGH", cons_rul, {"fail_prob_30": round(float(fail_prob), 2), "threshold": round(float(effective_tau_30), 2)})
        else:
            trace.append("M3:not_fired")

        raw_severity = SEVERITY_ORDER[sev_idx]

        # Step 6: Hysteresis (store raw severity, allow de-escalation after N cycles)
        if engine_id not in self.engine_hysteresis:
            self.engine_hysteresis[engine_id] = {
                "held_severity": raw_severity,
                "raw_severity": raw_severity,
                "consecutive_lower_cycles": 0,
                "last_cycle": cycle
            }

        hyst = self.engine_hysteresis[engine_id]
        held_idx = SEVERITY_ORDER.index(hyst["held_severity"])
        raw_idx = SEVERITY_ORDER.index(raw_severity)

        if raw_idx > held_idx:
            # Immediate escalation
            hyst["held_severity"] = raw_severity
            hyst["consecutive_lower_cycles"] = 0
            trace.append(f"escalated:{raw_severity}")
        elif raw_idx < held_idx:
            hyst["consecutive_lower_cycles"] += 1
            if hyst["consecutive_lower_cycles"] >= deescalate_cycles:
                hyst["held_severity"] = raw_severity
                hyst["consecutive_lower_cycles"] = 0
                trace.append(f"hysteresis:de_escalated_to_{raw_severity}(after_{deescalate_cycles}_cycles)")
            else:
                trace.append(f"hysteresis:held_at_{hyst['held_severity']}(lower_count={hyst['consecutive_lower_cycles']}/{deescalate_cycles})")
        else:
            hyst["consecutive_lower_cycles"] = 0

        hyst["raw_severity"] = raw_severity
        hyst["last_cycle"] = cycle
        self._persist_hysteresis(engine_id, hyst)
        final_severity = hyst["held_severity"]

        # Step 7: Independent Exception Checks (X1 - X7)
        triggered_exceptions: List[str] = []

        # X1: High prediction uncertainty
        interval_width = rul_p90 - rul_p10
        if interval_width > max_interval_width and SEVERITY_ORDER.index(final_severity) >= SEVERITY_ORDER.index("MEDIUM"):
            triggered_exceptions.append(f"X1: High prediction uncertainty (interval width = {interval_width:.1f} > {max_interval_width})")

        # X2: Model disagreement
        if anomaly_score >= anom_thresh and cons_rul > low_thresh:
            triggered_exceptions.append(f"X2: Model disagreement (high anomaly {anomaly_score:.2f} with healthy RUL {cons_rul:.1f} > {low_thresh})")
        elif cons_rul <= high_thresh and anomaly_score < 0.20:
            triggered_exceptions.append(f"X2: Model disagreement (near-failure RUL {cons_rul:.1f} <= {high_thresh} with low anomaly {anomaly_score:.2f})")

        # X3: Sensor flatlines
        flatlines_count = (data_quality_flags or {}).get("flatlines_count", 0)
        has_flatline = (data_quality_flags or {}).get("has_flatline", False)
        if flatlines_count >= 3 or has_flatline:
            triggered_exceptions.append(f"X3: Suspected sensor fault (flatline detected on {flatlines_count} sensors)")

        # X4: High missing telemetry fraction
        missing_frac = (data_quality_flags or {}).get("missing_fraction", 0.0)
        if missing_frac > max_missing_frac:
            triggered_exceptions.append(f"X4: High missing telemetry fraction ({missing_frac:.1%} > {max_missing_frac:.1%})")

        # X5: Sudden severity jump >= 2 levels in 1 cycle
        if len(hist) > 0:
            last_raw = hist[-1].get("raw_severity", "HEALTHY")
            jump = SEVERITY_ORDER.index(raw_severity) - SEVERITY_ORDER.index(last_raw)
            if jump >= 2:
                triggered_exceptions.append(f"X5: Sudden severity jump of {jump} levels in 1 cycle")

        # X6: Insufficient telemetry history with High/Critical
        if cycle < min_hist_cycles and final_severity in ["HIGH", "CRITICAL"]:
            triggered_exceptions.append(f"X6: Insufficient telemetry history (cycle {cycle} < {min_hist_cycles}) for high-urgency alert")

        # X7: Telemetry data gap, out-of-order cycles, or missing model specification
        last_recorded_cycle = hist[-1].get("cycle", 0) if len(hist) > 0 else 0
        if last_recorded_cycle > 0 and (cycle - last_recorded_cycle) > data_gap_cycles:
            triggered_exceptions.append(f"X7: Telemetry data gap detected ({cycle - last_recorded_cycle} cycles > {data_gap_cycles})")
        elif last_recorded_cycle > 0 and cycle < last_recorded_cycle:
            triggered_exceptions.append(f"X7: Telemetry time regression / out-of-order cycle ({cycle} < {last_recorded_cycle})")
        elif (data_quality_flags or {}).get("has_gap", False) or (data_quality_flags or {}).get("cycle_gap", 0) > data_gap_cycles:
            gap_len = (data_quality_flags or {}).get("cycle_gap", data_gap_cycles + 1)
            triggered_exceptions.append(f"X7: Telemetry data gap detected ({gap_len} cycles > {data_gap_cycles})")

        model_ver = prediction.get("model_version")
        if not model_ver or str(model_ver).strip() in ["unknown", "None", ""]:
            triggered_exceptions.append("X7: Missing model version specification")

        for exc in triggered_exceptions:
            trace.append(f"exception:{exc.split(':')[0]}")

        # Update historical evaluations and persist to SQLite
        hist_entry = {
            "cycle": cycle,
            "conservative_rul": cons_rul,
            "raw_severity": raw_severity,
            "severity": final_severity,
            "anomaly_score": anomaly_score,
            "exceptions": triggered_exceptions
        }
        hist.append(hist_entry)
        self._persist_state_history_entry(engine_id, hist_entry)

        # Step 8: Suppression window, recovery, & cooldown check
        existing_order = self.open_orders.get(engine_id)

        # Auto-close on sustained recovery to HEALTHY or LOW
        if existing_order and final_severity in ["HEALTHY", "LOW"] and not triggered_exceptions:
            if existing_order.get("status") not in HUMAN_STATUSES:
                existing_order["status"] = "CLOSED"
                existing_order["closed_at_cycle"] = cycle
                existing_order["current_cycle"] = cycle
                existing_order["conservative_rul"] = cons_rul
                existing_order["events"].append({
                    "cycle": cycle,
                    "event": f"Auto-closed on sustained recovery to {final_severity} (RUL: {cons_rul:.1f})",
                    "actor": "TriageAgent"
                })
                if engine_id not in self.closed_orders_history:
                    self.closed_orders_history[engine_id] = []
                self.closed_orders_history[engine_id].append(existing_order)
                del self.open_orders[engine_id]
                self._persist_order(existing_order)
                self._log_audit_event(
                    engine_id=engine_id,
                    action="AUTO_CLOSE",
                    actor="TriageAgent",
                    cycle=cycle,
                    work_order_id=existing_order["work_order_id"],
                    reason=f"Sustained recovery: severity dropped to {final_severity} for {deescalate_cycles} consecutive cycles",
                    trace=trace
                )
                trace.append(f"auto_close:sustained_recovery_to_{final_severity}")
                return existing_order, trace

        # Check human REJECT suppression window
        suppress_until = self.suppressed_engines.get(engine_id, 0)
        if cycle <= suppress_until and final_severity != "CRITICAL":
            trace.append(f"suppression:active_until_cycle_{suppress_until}")
            return None, trace

        # If healthy or low and no exception, no work order needed
        if final_severity in ["HEALTHY", "LOW"] and not triggered_exceptions:
            trace.append("severity:no_work_order_required")
            return None, trace

        # Check real closure cooldown
        closed_orders = self.closed_orders_history.get(engine_id, [])
        if closed_orders:
            last_closure_cycle = max(o.get("closed_at_cycle", o.get("created_at_cycle", 0)) for o in closed_orders)
            if (cycle - last_closure_cycle) < cooldown_cycles and final_severity != "CRITICAL":
                trace.append(f"cooldown:active({cycle - last_closure_cycle}/{cooldown_cycles}_cycles)")
                return None, trace

        # Subsystem & action mapping
        top_sensors = prediction.get("top_sensors", ["s2", "s3", "s4"])
        suspected_subsystem = "HPC"
        sensor_info = self.sensor_cfg.get("sensors", {})
        if top_sensors and top_sensors[0] in sensor_info:
            suspected_subsystem = sensor_info[top_sensors[0]].get("subsystem", "HPC")

        due_by_cycle = int(cycle + max(1, int(due_by_factor * cons_rul)))
        actions = {
            "CRITICAL": f"Immediate shutdown and remove from service. Perform emergent overhaul of {suspected_subsystem}.",
            "HIGH": f"Schedule urgent {suspected_subsystem} borescope inspection at next turnaround window.",
            "MEDIUM": f"Plan non-destructive evaluation of {suspected_subsystem} during upcoming scheduled maintenance."
        }
        recommended_action = actions.get(final_severity, f"Inspect {suspected_subsystem}")
        status = "NEEDS_REVIEW" if triggered_exceptions else "PROPOSED"

        # Step 9: Create or update work order
        if existing_order:
            # Refresh live RUL and cycle on open orders
            existing_order["current_cycle"] = cycle
            existing_order["conservative_rul"] = cons_rul
            existing_order["predicted_rul"] = {
                "p10": rul_p10,
                "p50": prediction.get("rul_p50", cons_rul + 10.0),
                "p90": rul_p90
            }
            existing_order["fail_prob_30"] = fail_prob
            existing_order["anomaly_score"] = anomaly_score
            existing_order["decision_trace"] = trace

            # NEVER overwrite human status
            if existing_order.get("status") in HUMAN_STATUSES:
                trace.append(f"human_status_preserved:{existing_order['status']}")
            else:
                if triggered_exceptions:
                    existing_order["status"] = "NEEDS_REVIEW"
                    existing_order["exception"] = "; ".join(triggered_exceptions)

            # Escalate or downgrade severity on sustained recovery
            existing_idx = SEVERITY_ORDER.index(existing_order["severity"])
            new_idx = SEVERITY_ORDER.index(final_severity)
            if new_idx > existing_idx:
                existing_order["severity"] = final_severity
                existing_order["recommended_action"] = recommended_action
                existing_order["due_by_cycle"] = due_by_cycle
                existing_order["events"].append({
                    "cycle": cycle,
                    "event": f"Escalated to {final_severity}",
                    "trace": trace
                })
                trace.append(f"dedupe:escalated_to_{final_severity}")
            elif new_idx < existing_idx:
                if existing_order.get("status") not in HUMAN_STATUSES:
                    existing_order["severity"] = final_severity
                    existing_order["recommended_action"] = recommended_action
                    existing_order["due_by_cycle"] = due_by_cycle
                    existing_order["events"].append({
                        "cycle": cycle,
                        "event": f"Downgraded to {final_severity} on sustained recovery",
                        "trace": trace
                    })
                    trace.append(f"dedupe:downgraded_to_{final_severity}")
                    self._log_audit_event(
                        engine_id=engine_id,
                        action="DOWNGRADE",
                        actor="TriageAgent",
                        cycle=cycle,
                        work_order_id=existing_order["work_order_id"],
                        reason=f"Sustained recovery: severity downgraded to {final_severity}",
                        trace=trace
                    )
            else:
                trace.append("dedupe:open_order_refreshed")

            self._persist_order(existing_order)
            return existing_order, trace

        # Generate new work order with UUID
        work_order_id = f"WO-{uuid.uuid4()}"
        work_order = {
            "work_order_id": work_order_id,
            "engine_id": engine_id,
            "unit": int(prediction.get("unit", 1)),
            "subset": subset,
            "created_at_cycle": cycle,
            "current_cycle": cycle,
            "closed_at_cycle": None,
            "severity": final_severity,
            "raw_severity": raw_severity,
            "status": status,
            "predicted_rul": {
                "p10": rul_p10,
                "p50": float(prediction.get("rul_p50", cons_rul + 10.0)),
                "p90": rul_p90
            },
            "conservative_rul": cons_rul,
            "fail_prob_30": fail_prob,
            "anomaly_score": anomaly_score,
            "due_by_cycle": due_by_cycle,
            "suspected_subsystem": suspected_subsystem,
            "top_sensors": top_sensors,
            "recommended_action": recommended_action,
            "decision_trace": trace,
            "model_version": prediction.get("model_version") or "unknown",
            "rules_version": self.rules_version,
            "exception": "; ".join(triggered_exceptions) if triggered_exceptions else None,
            "events": [{
                "cycle": cycle,
                "event": f"Work order created with severity {final_severity} ({status})",
                "actor": "TriageAgent"
            }]
        }

        self.open_orders[engine_id] = work_order
        self._persist_order(work_order)
        self._log_audit_event(
            engine_id=engine_id,
            action="CREATE",
            actor="TriageAgent",
            cycle=cycle,
            work_order_id=work_order_id,
            reason=f"Initial alert: {final_severity}",
            trace=trace
        )
        trace.append("dedupe:new_order_created")
        return work_order, trace

    def _create_exception_order(self, pred: Dict[str, Any], reason: str, trace: List[str]) -> Dict[str, Any]:
        engine_id = pred.get("engine_id", "UNKNOWN")
        cycle = int(pred.get("cycle", 1))
        work_order_id = f"WO-{uuid.uuid4()}"
        wo = {
            "work_order_id": work_order_id,
            "engine_id": engine_id,
            "unit": int(pred.get("unit", 0)),
            "subset": pred.get("subset", "FD001"),
            "created_at_cycle": cycle,
            "current_cycle": cycle,
            "closed_at_cycle": None,
            "severity": "CRITICAL",
            "raw_severity": "CRITICAL",
            "status": "NEEDS_REVIEW",
            "predicted_rul": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "conservative_rul": 0.0,
            "fail_prob_30": 1.0,
            "anomaly_score": 1.0,
            "due_by_cycle": cycle + 1,
            "suspected_subsystem": "Unknown",
            "top_sensors": ["s2", "s3"],
            "recommended_action": "Urgent manual review required due to telemetry exception.",
            "decision_trace": trace,
            "model_version": pred.get("model_version", "unknown"),
            "rules_version": self.rules_version,
            "exception": reason,
            "events": [{"cycle": cycle, "event": f"Exception created: {reason}", "actor": "TriageAgent"}]
        }
        self.open_orders[engine_id] = wo
        self._persist_order(wo)
        self._log_audit_event(engine_id=engine_id, action="EXCEPTION_CREATE", actor="TriageAgent", cycle=cycle, work_order_id=work_order_id, reason=reason, trace=trace)
        return wo

    def get_active_work_orders(self) -> List[Dict[str, Any]]:
        """Returns a list of all currently active/open work orders."""
        return list(self.open_orders.values())

    def human_action(self, work_order_id: str, action: str, new_severity: Optional[str] = None, reason: str = "", current_cycle: Optional[int] = None) -> bool:
        """
        Human-in-the-loop exception handling:
        Actions: APPROVE, REJECT, OVERRIDE, CLOSE
        Persists action in SQLite audit_events and updates work order state.
        """
        target_order = None
        target_engine = None
        for eid, order in self.open_orders.items():
            if order["work_order_id"] == work_order_id:
                target_order = order
                target_engine = eid
                break

        if not target_order:
            return False

        cycle = current_cycle if current_cycle is not None else target_order.get("current_cycle", target_order.get("created_at_cycle", 1))

        if action == "ADVANCE":
            next_stage = get_next_lifecycle_stage(target_order.get("status", "PROPOSED"))
            if not next_stage:
                return False
            target_order["status"] = next_stage
            target_order["events"].append({
                "cycle": cycle,
                "event": f"Advanced to lifecycle stage {next_stage}",
                "reason": reason or "Standard maintenance progression",
                "actor": "MaintenanceEngineer"
            })
            if next_stage == "CLOSED":
                target_order["closed_at_cycle"] = cycle
                if target_engine not in self.closed_orders_history:
                    self.closed_orders_history[target_engine] = []
                self.closed_orders_history[target_engine].append(target_order)
                del self.open_orders[target_engine]
            self._persist_order(target_order)
            self._log_audit_event(
                engine_id=target_engine,
                action=f"STAGE_{next_stage}",
                actor="MaintenanceEngineer",
                cycle=cycle,
                work_order_id=work_order_id,
                reason=reason or f"Advanced to {next_stage}"
            )
            return True

        elif action in MAINTENANCE_STAGES:
            target_order["status"] = action
            target_order["events"].append({
                "cycle": cycle,
                "event": f"Lifecycle stage set to {action}",
                "reason": reason or "Operator stage transition",
                "actor": "HumanOperator"
            })
            if action == "CLOSED":
                target_order["closed_at_cycle"] = cycle
                if target_engine not in self.closed_orders_history:
                    self.closed_orders_history[target_engine] = []
                self.closed_orders_history[target_engine].append(target_order)
                del self.open_orders[target_engine]
            self._persist_order(target_order)
            self._log_audit_event(
                engine_id=target_engine,
                action=f"STAGE_{action}",
                actor="HumanOperator",
                cycle=cycle,
                work_order_id=work_order_id,
                reason=reason or f"Status set to {action}"
            )
            return True

        elif action == "APPROVE":
            target_order["status"] = "APPROVED"
            target_order["events"].append({"cycle": cycle, "event": "Approved by human engineer", "reason": reason, "actor": "HumanOperator"})
            self._persist_order(target_order)
            self._log_audit_event(engine_id=target_engine, action="APPROVE", actor="HumanOperator", cycle=cycle, work_order_id=work_order_id, reason=reason)

        elif action == "REJECT":
            target_order["status"] = "REJECTED"
            target_order["closed_at_cycle"] = cycle
            target_order["events"].append({"cycle": cycle, "event": "Rejected by human engineer", "reason": reason, "actor": "HumanOperator"})
            # Apply suppression window
            suppression_cycles = self.rules_cfg.get("lifecycle", {}).get("suppression_window_cycles", 10)
            self.suppressed_engines[target_engine] = cycle + suppression_cycles
            
            # Archive order
            if target_engine not in self.closed_orders_history:
                self.closed_orders_history[target_engine] = []
            self.closed_orders_history[target_engine].append(target_order)
            del self.open_orders[target_engine]

            self._persist_order(target_order)
            self._log_audit_event(
                engine_id=target_engine,
                action="REJECT",
                actor="HumanOperator",
                cycle=cycle,
                work_order_id=work_order_id,
                reason=reason
            )

        elif action == "OVERRIDE" and new_severity in SEVERITY_ORDER:
            target_order["severity"] = new_severity
            target_order["status"] = "OVERRIDDEN"
            target_order["events"].append({"cycle": cycle, "event": f"Severity manually overridden to {new_severity}", "reason": reason, "actor": "HumanOperator"})
            if target_engine in self.engine_hysteresis:
                self.engine_hysteresis[target_engine]["held_severity"] = new_severity
                self._persist_hysteresis(target_engine, self.engine_hysteresis[target_engine])
            self._persist_order(target_order)
            self._log_audit_event(engine_id=target_engine, action="OVERRIDE", actor="HumanOperator", cycle=cycle, work_order_id=work_order_id, reason=reason)

        elif action == "CLOSE":
            target_order["status"] = "CLOSED"
            target_order["closed_at_cycle"] = cycle
            target_order["events"].append({"cycle": cycle, "event": "Work order closed", "reason": reason, "actor": "HumanOperator"})
            
            if target_engine not in self.closed_orders_history:
                self.closed_orders_history[target_engine] = []
            self.closed_orders_history[target_engine].append(target_order)
            del self.open_orders[target_engine]

            self._persist_order(target_order)
            self._log_audit_event(engine_id=target_engine, action="CLOSE", actor="HumanOperator", cycle=cycle, work_order_id=work_order_id, reason=reason)

        return True

    def _dispatch_rule_webhook(
        self,
        rule_id: str,
        engine_id: Any,
        cycle: int,
        severity: str,
        cons_rul: float,
        details: Optional[Dict[str, Any]] = None,
        suspected_subsystem: str = "HPC Compressor"
    ):
        """
        Dispatches external webhook alert (Slack/Teams/PagerDuty) when Rule M1, M2, or M3 fires.
        """
        try:
            if not hasattr(self, "_alert_manager") or self._alert_manager is None:
                from src.mlops.alerts import AlertManager
                self._alert_manager = AlertManager()
            if self._alert_manager:
                self._alert_manager.dispatch_m_rule(
                    rule_id=rule_id,
                    engine_id=engine_id,
                    cycle=cycle,
                    severity=severity,
                    conservative_rul=cons_rul,
                    details=details,
                    suspected_subsystem=suspected_subsystem
                )
        except Exception:
            pass

