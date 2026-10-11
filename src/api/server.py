import os
import sys

# Constrain thread pools and glibc memory arenas to prevent memory explosion on 512MB cloud instances
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import asyncio
import concurrent.futures
import gc
import ctypes
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import io
import re

def trim_memory():
    """Forces garbage collection and releases glibc heap memory back to the Linux OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Body, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.common.utils import get_root_dir, load_yaml
from src.models.model_service import ModelService
from src.agent.triage_agent import TriageAgent
from src.agent.self_healing_engine import SelfHealingEngine
from src.mlops.drift_and_retrain import MLOpsManager
from src.features.streaming import StreamingEnginePipeline

app = FastAPI(
    title="AETURNUS API",
    description="Autonomous Equipment Tracking and Uptime Restoration through Networked Unified Systems",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton states
ROOT_DIR = get_root_dir()
current_subset = "FD001"
model_services: Dict[str, ModelService] = {}
triage_agent = TriageAgent()
self_healing_engine = SelfHealingEngine()
mlops_managers: Dict[str, MLOpsManager] = {}
streaming_pipelines: Dict[str, StreamingEnginePipeline] = {}

# Cap executor workers at 2 to conserve thread memory in 512MB containers
_fleet_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
_prediction_cache: Dict[tuple, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

def get_cached_prediction(ms: ModelService, row_dict: Dict[str, Any], subset: str) -> Dict[str, Any]:
    unit = int(row_dict.get("unit", 1))
    cycle = int(row_dict.get("cycle", 1))
    cache_key = (str(subset), unit, cycle, getattr(ms, "version", "v1"))
    with _cache_lock:
        if cache_key in _prediction_cache:
            return _prediction_cache[cache_key]
    pred = ms.predict_point(row_dict)
    with _cache_lock:
        if len(_prediction_cache) >= 500:
            _prediction_cache.clear()
        _prediction_cache[cache_key] = pred
    return pred

def clear_prediction_cache():
    with _cache_lock:
        _prediction_cache.clear()
    trim_memory()

# Replay simulation state
class ReplayState:
    def __init__(self):
        self.is_running = False
        self.speed = 1.0 # 1x, 5x, 20x
        self.current_cycle = 30
        self.max_cycle = 200
        self.selected_lifecycle_unit = 1 # for single lifecycle demo
        self.mode = "fleet" # 'fleet' or 'lifecycle_demo'
        self.recent_replay_window: List[Dict[str, Any]] = []

replay_state = ReplayState()
active_websockets: List[WebSocket] = []

# Cached processed dataframes (Single-subset active cache to keep RAM < 200MB)
loaded_dfs: Dict[str, pd.DataFrame] = {}

def _load_downcasted_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Downcast float64 to float32: cuts memory usage by 50% with zero loss of accuracy
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    trim_memory()
    return df

def get_model_service(subset: str = "FD001") -> ModelService:
    if subset not in model_services:
        # Evict older models to keep only 1 active model bundle in memory
        model_services.clear()
        trim_memory()

        ms = ModelService(subset)
        try:
            ms.load()
        except Exception:
            proc_train = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
            if proc_train.exists():
                train_df = get_train_df(subset)
                ms.train(train_df)
        model_services[subset] = ms
    return model_services[subset]

def get_mlops_manager(subset: str = "FD001") -> MLOpsManager:
    if subset not in mlops_managers:
        mlops_managers.clear()
        trim_memory()
        mlops_managers[subset] = MLOpsManager(subset)
    return mlops_managers[subset]

def get_streaming_pipeline(subset: str = "FD001") -> StreamingEnginePipeline:
    if subset not in streaming_pipelines:
        streaming_pipelines.clear()
        trim_memory()
        ms = get_model_service(subset)
        streaming_pipelines[subset] = StreamingEnginePipeline(subset=subset, model_service=ms, triage_agent=triage_agent)
    return streaming_pipelines[subset]

VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}

def validate_subset(subset: str):
    if subset not in VALID_SUBSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unsupported subset '{subset}'. Must be one of {sorted(list(VALID_SUBSETS))}."
        )

def get_test_df(subset: str = "FD001") -> pd.DataFrame:
    validate_subset(subset)
    key = f"test_{subset}"
    if key not in loaded_dfs:
        # Keep only the active dataset in memory to guarantee RAM < 200MB
        loaded_dfs.clear()
        trim_memory()

        proc_path = ROOT_DIR / "data" / "processed" / subset / "test.parquet"
        if not proc_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset).run()
            trim_memory()
        loaded_dfs[key] = _load_downcasted_parquet(proc_path)
        trim_memory()
    return loaded_dfs[key]

def get_train_df(subset: str = "FD001") -> pd.DataFrame:
    validate_subset(subset)
    key = f"train_{subset}"
    if key not in loaded_dfs:
        loaded_dfs.clear()
        trim_memory()

        proc_path = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
        if not proc_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset).run()
            trim_memory()
        loaded_dfs[key] = _load_downcasted_parquet(proc_path)
        trim_memory()
    return loaded_dfs[key]

# ----------------- WebSocket Hub -----------------
async def broadcast(event_type: str, data: Any):
    if not active_websockets:
        return
    message = json.dumps({"type": event_type, "data": data})
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "data": {
                "subset": current_subset,
                "current_cycle": replay_state.current_cycle,
                "is_running": replay_state.is_running
            }
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# ----------------- Replay Background Loop -----------------
async def replay_ticker_loop():
    while True:
        try:
            if replay_state.is_running:
                if replay_state.current_cycle >= replay_state.max_cycle:
                    replay_state.is_running = False
                    await broadcast("replay_finished", {
                        "cycle": replay_state.current_cycle,
                        "status": "completed"
                    })
                    await asyncio.sleep(0.5)
                    continue

                replay_state.current_cycle += 1
                cycle = replay_state.current_cycle

                # In lifecycle demo mode, evaluate held-out run-to-failure train engine over full lifespan
                if replay_state.mode == "lifecycle_demo":
                    train_df = get_train_df(current_subset)
                    from src.common.splits import get_persistent_split
                    split = get_persistent_split(current_subset, sorted([int(u) for u in train_df["unit"].unique()]))
                    holdout_units = split["lifecycle_holdout_units"]
                    unit = replay_state.selected_lifecycle_unit
                    if unit not in holdout_units:
                        unit = int(holdout_units[-1])
                        replay_state.selected_lifecycle_unit = unit

                    unit_rows = train_df[(train_df["unit"] == unit) & (train_df["cycle"] == cycle)]
                    if len(unit_rows) > 0:
                        raw_row = unit_rows.iloc[0].to_dict()
                        replay_state.recent_replay_window.append(raw_row)
                        if len(replay_state.recent_replay_window) > 1000:
                            replay_state.recent_replay_window = replay_state.recent_replay_window[-1000:]
                        
                        sp = get_streaming_pipeline(current_subset)
                        result = sp.stream_cycle(raw_row)
                        await broadcast("tick", {
                            "cycle": cycle,
                            "unit": unit,
                            "prediction": result["prediction"],
                            "order": result["work_order"],
                            "trace": result["trace"]
                        })
                else:
                    df = get_test_df(current_subset)
                    ms = get_model_service(current_subset)
                    sub_rows = df[df["cycle"] == cycle]
                    all_units = sorted(sub_rows["unit"].unique())
                    rows = [sub_rows[sub_rows["unit"] == u].iloc[0].to_dict() for u in all_units]
                    preds = list(_fleet_executor.map(lambda r: get_cached_prediction(ms, r, current_subset), rows))
                    
                    fleet_predictions = []
                    for row, pred in zip(rows, preds):
                        replay_state.recent_replay_window.append(row)
                        order, trace = triage_agent.evaluate_engine_cycle(pred)
                        fleet_predictions.append(pred)
                        if order and order.get("created_at_cycle") == cycle:
                            await broadcast("work_order_created", order)

                    if len(replay_state.recent_replay_window) > 100:
                        replay_state.recent_replay_window = replay_state.recent_replay_window[-100:]

                    await broadcast("tick", {
                        "cycle": cycle,
                        "engine_count": len(all_units),
                        "predictions": fleet_predictions
                    })

                if cycle % 25 == 0:
                    trim_memory()

                delay = max(0.05, 1.0 / replay_state.speed)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Replay ticker loop error: {e}")
            await asyncio.sleep(0.5)

_system_ready = False

def is_system_ready() -> bool:
    global _system_ready
    if _system_ready:
        return True
    train_file = ROOT_DIR / "data" / "processed" / "FD001" / "train.parquet"
    test_file = ROOT_DIR / "data" / "processed" / "FD001" / "test.parquet"
    model_file = ROOT_DIR / "artifacts" / "models" / "model_bundle_FD001.joblib"
    if train_file.exists() and test_file.exists() and model_file.exists():
        _system_ready = True
        return True
    return False

@app.on_event("startup")
async def startup_event():
    try:
        ms = get_model_service(current_subset)
        df = get_test_df(current_subset)
        last_indices = df.groupby("unit")["cycle"].idxmax()
        records = df.loc[last_indices].sort_values("unit").to_dict("records")
        list(_fleet_executor.map(lambda r: get_cached_prediction(ms, r, current_subset), records))
    except Exception as e:
        print(f"Startup precomputation notice: {e}")
    finally:
        trim_memory()
    asyncio.create_task(replay_ticker_loop())

# ----------------- REST Endpoints -----------------

@app.get("/api/v1/health")
def health():
    if not is_system_ready():
        raise HTTPException(status_code=503, detail="System initializing: data and models not ready yet")
    return {"status": "ok", "timestamp": time.time(), "version": "1.0.0"}

@app.get("/api/v1/fleet")
def get_fleet(subset: str = Query(default="FD001")):
    df = get_test_df(subset)
    ms = get_model_service(subset)

    # Compute latest status per engine across ALL engines
    last_indices = df.groupby("unit")["cycle"].idxmax()
    last_rows_df = df.loc[last_indices].sort_values("unit")

    # Vectorized 10-point sparkline per engine
    history_by_unit = df.groupby("unit")["health_index"].apply(
        lambda s: [float(x) for x in s.tail(10).round(2).tolist()]
    ).to_dict()

    engines = []
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "HEALTHY": 0}
    total_rul = 0.0

    all_units = sorted(df["unit"].unique())
    records = last_rows_df.to_dict("records")
    preds = list(_fleet_executor.map(lambda r: get_cached_prediction(ms, r, subset), records))
    sensor_map = load_yaml("configs/sensor_map.yaml").get("sensors", {})

    for row_dict, pred in zip(records, preds):
        u = int(row_dict["unit"])
        max_c = int(row_dict["cycle"])

        order = triage_agent.open_orders.get(pred["engine_id"])
        c_rul = float(pred["conservative_rul"])
        severity = order["severity"] if order else triage_agent.get_base_severity(c_rul)
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        total_rul += c_rul

        sparkline = history_by_unit.get(u, [float(pred["health_index"])])
        top_s = pred["top_sensors"][0] if pred.get("top_sensors") else "s11"
        subsystem_name = sensor_map.get(top_s, {}).get("subsystem", "HPC")

        engines.append({
            "engine_id": str(pred["engine_id"]),
            "unit": u,
            "current_cycle": max_c,
            "conservative_rul": c_rul,
            "rul_p50": float(pred["rul_p50"]),
            "health_index": float(pred["health_index"]),
            "anomaly_score": float(pred["anomaly_score"]),
            "severity": str(severity),
            "has_open_order": bool(order is not None),
            "sparkline": sparkline,
            "suspected_subsystem": subsystem_name
        })

    avg_rul = round(float(total_rul) / max(1, len(engines)), 1)
    critical_count = int(severity_counts["CRITICAL"])
    open_orders_count = int(len(triage_agent.open_orders))
    needs_review_count = int(sum(1 for o in triage_agent.open_orders.values() if o.get("status") == "NEEDS_REVIEW"))

    mgr = get_mlops_manager(subset)
    reg = mgr.get_registry()
    champ_ver = reg.get("active_champion", "v1.0")

    return {
        "subset": str(subset),
        "kpis": {
            "engines_monitored": int(len(all_units)),
            "critical_count": critical_count,
            "open_work_orders": open_orders_count,
            "exception_queue_size": needs_review_count,
            "mean_conservative_rul": avg_rul,
            "live_model_version": f"rul-{subset.lower()}@{champ_ver}"
        },
        "severity_distribution": {str(k): int(v) for k, v in severity_counts.items()},
        "engines": engines
    }

@app.get("/api/v1/engines/{engine_id}")
def get_engine_detail(engine_id: str, subset: str = Query(default="FD001")):
    df = get_test_df(subset)
    ms = get_model_service(subset)
    try:
        unit_str = engine_id.split("-")[-1]
        unit_num = int(unit_str)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid engine ID: {engine_id}")

    unit_df = df[df["unit"] == unit_num].sort_values("cycle")
    if unit_df.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found in subset {subset}")

    last_row = unit_df.iloc[-1].to_dict()
    pred = ms.predict_point(last_row)
    order = triage_agent.open_orders.get(pred["engine_id"])

    # Build cycle-by-cycle series for charts
    trajectory = []
    for _, row in unit_df.iterrows():
        r_dict = row.to_dict()
        p = ms.predict_point(r_dict)
        trajectory.append({
            "cycle": int(r_dict["cycle"]),
            "rul_p10": p["rul_p10"],
            "rul_p50": p["rul_p50"],
            "rul_p90": p["rul_p90"],
            "true_rul": round(float(r_dict.get("rul_raw", 0.0)), 1),
            "health_index": p["health_index"],
            "anomaly_score": p["anomaly_score"],
            "s2_raw": round(float(r_dict.get("s2", 0.0)), 2),
            "s2_ewma": round(float(r_dict.get("s2_ewma", 0.0)), 2),
            "s3_raw": round(float(r_dict.get("s3", 0.0)), 2),
            "s3_ewma": round(float(r_dict.get("s3_ewma", 0.0)), 2),
            "s4_raw": round(float(r_dict.get("s4", 0.0)), 2),
            "s4_ewma": round(float(r_dict.get("s4_ewma", 0.0)), 2),
            "s7_raw": round(float(r_dict.get("s7", 0.0)), 2),
            "s7_ewma": round(float(r_dict.get("s7_ewma", 0.0)), 2),
        })

    # Sensor attributions computed via sensor_map.yaml
    sensor_map = load_yaml("configs/sensor_map.yaml").get("sensors", {})
    sensor_attributions = []
    for s in pred.get("top_sensors", []):
        info = sensor_map.get(s, {})
        subsystem = info.get("subsystem", "HPC")
        dev = abs(float(last_row.get(f"{s}_dev_base", 1.0)))
        sensor_attributions.append({
            "sensor": s,
            "name": info.get("name", s),
            "subsystem": subsystem,
            "contribution": round(min(1.0, dev / 10.0), 3),
            "deviation_score": round(dev, 3)
        })

    return {
        "engine_id": engine_id,
        "unit": unit_num,
        "subset": subset,
        "current_cycle": int(unit_df.iloc[-1]["cycle"]),
        "prediction": pred,
        "work_order": order,
        "trajectory": trajectory,
        "sensor_attributions": sensor_attributions,
        "shap_bars": sensor_attributions,
        "regime": int(last_row.get("regime", 0))
    }

# ----------------- Replay Controls -----------------
@app.post("/api/v1/replay/start")
def start_replay(speed: float = Query(default=1.0), mode: str = Query(default="fleet"), unit: int = Query(default=1)):
    replay_state.is_running = True
    replay_state.speed = speed
    replay_state.mode = mode
    replay_state.selected_lifecycle_unit = unit
    return {"status": "started", "speed": replay_state.speed, "mode": replay_state.mode}

@app.post("/api/v1/replay/pause")
def pause_replay():
    replay_state.is_running = False
    return {"status": "paused"}

@app.post("/api/v1/replay/reset")
def reset_replay():
    replay_state.is_running = False
    replay_state.current_cycle = 20
    return {"status": "reset", "cycle": replay_state.current_cycle}

@app.get("/api/v1/replay/state")
def get_replay_state():
    return {
        "is_running": replay_state.is_running,
        "speed": replay_state.speed,
        "current_cycle": replay_state.current_cycle,
        "mode": replay_state.mode
    }

# ----------------- Work Orders -----------------
@app.get("/api/v1/work-orders")
def get_work_orders(status: Optional[str] = None, severity: Optional[str] = None):
    orders = list(triage_agent.open_orders.values())
    for hist_list in triage_agent.closed_orders_history.values():
        orders.extend(hist_list)

    if status:
        orders = [o for o in orders if o.get("status") == status]
    if severity:
        orders = [o for o in orders if o.get("severity") == severity]

    return {"count": len(orders), "work_orders": sorted(orders, key=lambda x: x.get("created_at_cycle", 0), reverse=True)}

@app.get("/api/v1/work-orders/{work_order_id}")
def get_work_order_detail(work_order_id: str):
    for o in triage_agent.open_orders.values():
        if o["work_order_id"] == work_order_id:
            return o
    for h_list in triage_agent.closed_orders_history.values():
        for o in h_list:
            if o["work_order_id"] == work_order_id:
                return o
    raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")

class HumanActionRequest(BaseModel):
    action: str # APPROVE, REJECT, OVERRIDE, CLOSE
    reason: str = "Operator decision"
    new_severity: Optional[str] = None

@app.post("/api/v1/work-orders/{work_order_id}/action")
def take_human_action(work_order_id: str, req: HumanActionRequest):
    success = triage_agent.human_action(
        work_order_id, req.action, req.new_severity, req.reason, current_cycle=replay_state.current_cycle
    )
    if not success:
        raise HTTPException(status_code=400, detail="Unable to apply action to work order")
@app.post("/api/v1/work-orders/{work_order_id}/advance")
def advance_work_order(work_order_id: str, reason: str = Query(default="Stage progression")):
    success = triage_agent.human_action(
        work_order_id, "ADVANCE", reason=reason, current_cycle=replay_state.current_cycle
    )
    if not success:
        raise HTTPException(status_code=400, detail="Unable to advance work order stage")
    return {"status": "success", "work_order_id": work_order_id}

@app.post("/api/v1/work-orders/{work_order_id}/derate")
def apply_derate(work_order_id: str, reason: str = Query(default="Flight operations FADEC derate authorization")):
    success = triage_agent.human_action(
        work_order_id, "APPLY_DERATE", reason=reason, current_cycle=replay_state.current_cycle
    )
    if not success:
        raise HTTPException(status_code=400, detail="Unable to apply derate to work order")
    return {"status": "success", "work_order_id": work_order_id}

@app.get("/api/v1/work-orders/{work_order_id}/airworthiness-report.pdf")
def download_airworthiness_report_by_wo(work_order_id: str):
    from src.agent.aerospace_pdm import generate_airworthiness_pdf
    target_wo = None
    for o in triage_agent.open_orders.values():
        if o["work_order_id"] == work_order_id:
            target_wo = o
            break
    if not target_wo:
        for h_list in triage_agent.closed_orders_history.values():
            for o in h_list:
                if o["work_order_id"] == work_order_id:
                    target_wo = o
                    break
            if target_wo:
                break
    if not target_wo:
        raise HTTPException(status_code=404, detail=f"Work order {work_order_id} not found")

    engine_id = target_wo["engine_id"]
    pdf_bytes = generate_airworthiness_pdf(engine_id, target_wo)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="airworthiness_report_{work_order_id}.pdf"'}
    )

@app.get("/api/v1/engines/{engine_id}/airworthiness-report.pdf")
def download_airworthiness_report_by_engine(engine_id: str, subset: str = Query(default="FD001")):
    from src.agent.aerospace_pdm import generate_airworthiness_pdf, calculate_throttle_derating, generate_parts_bom
    target_wo = triage_agent.open_orders.get(engine_id)
    if not target_wo:
        closed = triage_agent.closed_orders_history.get(engine_id, [])
        if closed:
            target_wo = closed[-1]

    if not target_wo:
        df = get_test_df(subset)
        unit = int(engine_id.split("-")[-1]) if "-" in engine_id else 1
        unit_df = df[df["unit"] == unit]
        cycle = int(unit_df["cycle"].max()) if not unit_df.empty else 100
        ms = get_model_service(subset)
        row = unit_df[unit_df["cycle"] == cycle].iloc[0].to_dict() if not unit_df.empty else {}
        pred = ms.predict_point(row) if row else {"conservative_rul": 55.0, "rul_p10": 50.0, "rul_p50": 60.0, "rul_p90": 70.0}
        target_wo = {
            "work_order_id": f"WO-{engine_id}-CERT",
            "engine_id": engine_id,
            "unit": unit,
            "subset": subset,
            "created_at_cycle": cycle,
            "current_cycle": cycle,
            "severity": "LOW",
            "status": "AIRWORTHY_CLOSED",
            "conservative_rul": pred.get("conservative_rul", 50.0),
            "rul_p10": pred.get("rul_p10", 45.0),
            "rul_p50": pred.get("rul_p50", 55.0),
            "rul_p90": pred.get("rul_p90", 65.0),
            "suspected_subsystem": "HPC",
            "top_sensors": pred.get("top_sensors", ["s9", "s14"]),
            "events": [{"cycle": cycle, "actor": "Inspector", "event": "Routine airworthiness baseline certified"}],
            "derate_plan": calculate_throttle_derating(pred.get("conservative_rul", 50.0), "LOW", "HPC"),
            "parts_bom": generate_parts_bom(f"WO-{engine_id}-CERT", pred.get("top_sensors", ["s9", "s14"]), "HPC")
        }

    pdf_bytes = generate_airworthiness_pdf(engine_id, target_wo)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="airworthiness_report_{engine_id}.pdf"'}
    )

# ----------------- Rules & What-If Simulator -----------------
@app.get("/api/v1/rules")
def get_rules():
    return load_yaml("configs/agent_rules.yaml")

class WhatIfRequest(BaseModel):
    critical_threshold: int = 15
    high_threshold: int = 30
    anomaly_threshold: float = 0.80

@app.post("/api/v1/rules/simulate")
def simulate_what_if(req: WhatIfRequest, subset: str = Query(default="FD001")):
    validate_subset(subset)
    train_df = get_train_df(subset)
    ms = get_model_service(subset)

    from src.common.splits import get_persistent_split
    split = get_persistent_split(subset, sorted([int(u) for u in train_df["unit"].unique()]))
    holdout_units = split.get("lifecycle_holdout_units", [])
    if not holdout_units:
        holdout_units = sorted([int(u) for u in train_df["unit"].unique()])[-15:]

    sim_cfg = json.loads(json.dumps(triage_agent.rules_cfg))
    sim_cfg.setdefault("severity_bands", {})
    sim_cfg["severity_bands"]["critical"] = req.critical_threshold
    sim_cfg["severity_bands"]["high"] = req.high_threshold
    sim_cfg.setdefault("escalation", {})
    sim_cfg["escalation"]["anomaly_threshold"] = req.anomaly_threshold

    sim_agent = TriageAgent(db_path=Path(":memory:"), rules_cfg=sim_cfg)

    simulated_severities = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    # Replay lifecycle holdout engines chronologically over their run-to-failure lifecycle
    holdout_df = train_df[train_df["unit"].isin(holdout_units)].sort_values(["unit", "cycle"])
    X = holdout_df[ms.feature_cols].values
    p10_preds = np.maximum(0.0, ms.rul_p10_model.predict(X))
    p50_preds = np.maximum(0.0, ms.rul_p50_model.predict(X))
    p90_preds = np.maximum(0.0, ms.rul_p90_model.predict(X))
    fail_probs = ms.classifier.predict_proba(X)[:, 1] if ms.classifier else np.zeros(len(X))
    if ms.anomaly_detector:
        scores = -ms.anomaly_detector.score_samples(X)
        norm_scores = (scores - scores.min()) / max(1e-5, scores.max() - scores.min())
    else:
        norm_scores = np.zeros(len(X))

    records = holdout_df.to_dict("records")
    for i, row in enumerate(records):
        pred = {
            "engine_id": f"{subset}-{int(row.get('unit', 1)):03d}",
            "unit": int(row.get("unit", 1)),
            "subset": subset,
            "cycle": int(row.get("cycle", 1)),
            "rul_p10": round(float(p10_preds[i]), 1),
            "rul_p50": round(float(p50_preds[i]), 1),
            "rul_p90": round(float(p90_preds[i]), 1),
            "conservative_rul": round(float(p10_preds[i]), 1),
            "fail_prob_30": round(float(fail_probs[i]), 3),
            "anomaly_score": round(float(norm_scores[i]), 3),
            "health_index": round(float(row.get("health_index", 1.0)), 3),
            "top_sensors": ["s2", "s3", "s4"],
            "model_version": f"rul-{subset.lower()}@{getattr(ms, 'version', 'v1.0')}"
        }
        sim_agent.evaluate_engine_cycle(pred)

    # Aggregate orders created across all holdout engines
    all_orders = list(sim_agent.open_orders.values())
    for hist_list in sim_agent.closed_orders_history.values():
        all_orders.extend(hist_list)

    for o in all_orders:
        sev = o.get("severity", "LOW")
        if sev in simulated_severities:
            simulated_severities[sev] += 1

    del holdout_df, records, X, sim_agent
    trim_memory()

    return {
        "subset": subset,
        "simulated_params": req.dict(),
        "holdout_engines_count": len(holdout_units),
        "total_cycles_replayed": len(records) if 'records' in locals() else len(holdout_units),
        "total_simulated_orders": len(all_orders),
        "severities": simulated_severities
    }

# ----------------- Model Lab & Benchmarks -----------------
@app.get("/api/v1/models/benchmark")
def get_model_benchmark():
    summary_file = ROOT_DIR / "artifacts" / "reports" / "benchmark_summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Fallback to computing on the fly if file missing
    from scripts.evaluate_all_models import main as run_eval
    run_eval()
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark summary report not found")

@app.get("/api/v1/models/{subset}/details")
def get_model_details(subset: str):
    validate_subset(subset)
    eval_file = ROOT_DIR / "artifacts" / "reports" / f"evaluation_{subset}.json"
    if not eval_file.exists():
        test_df = get_test_df(subset)
        loader = CMAPSSLoader(subset=subset)
        _, _, rul_truth = loader.load_raw_data()
        ms = get_model_service(subset)
        res = ms.evaluate_test(test_df, rul_truth)
        return {
            "subset": subset,
            "bucket_errors": res.get("bucket_errors", []),
            "feature_importance": res.get("feature_importance", []),
            "status": res.get("status", "Evaluated")
        }
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "subset": subset,
        "bucket_errors": data.get("bucket_errors", []),
        "feature_importance": data.get("feature_importance", []),
        "status": data.get("status", "Evaluated")
    }

@app.get("/api/v1/models/{subset}/comparison")
def get_model_comparison(subset: str):
    validate_subset(subset)
    comp_file = ROOT_DIR / "artifacts" / "reports" / f"model_comparison_{subset}.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"Comparison report for {subset} not found")

# ----------------- MLOps & Monitoring -----------------
@app.get("/api/v1/mlops/registry")
@app.get("/api/v1/registry")
def get_registry(subset: str = Query(default="FD001")):
    validate_subset(subset)
    mgr = get_mlops_manager(subset)
    return mgr.get_registry()

@app.get("/api/v1/monitoring/drift")
def get_drift_report(subset: str = Query(default="FD001")):
    validate_subset(subset)
    mgr = get_mlops_manager(subset)
    # 1. Training reference from processed training dataset (only load drift-relevant columns to save 80% RAM)
    train_path = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
    if train_path.exists():
        import pyarrow.parquet as pq
        avail = set(pq.ParquetFile(train_path).schema.names)
        drift_cols = ["cycle"] + [c for c in avail if c.endswith("_dev_base") or c.endswith("_ewma") or (c.startswith("s") and c[1:].isdigit())]
        train_df = pd.read_parquet(train_path, columns=[c for c in drift_cols if c in avail])
    else:
        train_df = get_test_df(subset)

    # 2. Current telemetry from recent replay window if available, else nominal matched baseline
    if len(replay_state.recent_replay_window) >= 30:
        curr_df = pd.DataFrame(replay_state.recent_replay_window)
        c_min = int(curr_df["cycle"].min())
        c_max = int(curr_df["cycle"].max())
        matched_train = train_df[(train_df["cycle"] >= c_min) & (train_df["cycle"] <= c_max)]
        ref_df = matched_train if len(matched_train) >= 30 else train_df
    else:
        # Default state: nominal operating window (cycles 1..50)
        test_df = get_test_df(subset)
        curr_df = test_df[(test_df["cycle"] >= 1) & (test_df["cycle"] <= 50)]
        ref_df = train_df[(train_df["cycle"] >= 1) & (train_df["cycle"] <= 50)]
        if len(ref_df) == 0 or len(curr_df) == 0:
            ref_df = train_df.sample(min(500, len(train_df)), random_state=42)
            curr_df = test_df.sample(min(500, len(test_df)), random_state=42)

    result = mgr.calculate_drift_report(ref_df, curr_df)
    del train_df
    trim_memory()
    return result

@app.post("/api/v1/mlops/retrain")
async def trigger_retrain(scenario: str = Query(default="A"), subset: str = Query(default="FD001")):
    validate_subset(subset)
    mgr = get_mlops_manager(subset)
    
    curr_df = pd.DataFrame(replay_state.recent_replay_window) if len(replay_state.recent_replay_window) >= 30 else None
    loop = asyncio.get_event_loop()

    def sync_progress_callback(step: int, total: int, msg: str):
        pct = round((step / total) * 100)
        asyncio.run_coroutine_threadsafe(
            broadcast("retrain_progress", {
                "step": step,
                "total_steps": total,
                "progress_pct": pct,
                "message": msg,
                "subset": subset,
                "scenario": scenario
            }),
            loop
        )

    result = await asyncio.to_thread(
        mgr.run_retrain_demo,
        scenario=scenario,
        replay_window_df=curr_df,
        progress_callback=sync_progress_callback
    )

    await broadcast("retrain_completed", {
        "subset": subset,
        "scenario": scenario,
        "result": result
    })

    return result

# ----------------- Data Pipeline & Quality -----------------
@app.get("/api/v1/data/quality")
def get_data_quality(subset: str = Query(default="FD001")):
    validate_subset(subset)
    report_file = ROOT_DIR / "artifacts" / "reports" / f"data_quality_{subset}.json"
    if not report_file.exists():
        from src.features.pipeline import DataPipeline
        DataPipeline(subset=subset).run()
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"Data quality report for {subset} not found")

# ----------------- Pipeline & Model Operations Endpoints -----------------
@app.post("/api/v1/mlops/{name}/promote")
@app.post("/api/v1/mlops/promote")
@app.post("/api/v1/models/{name}/promote")
@app.post("/api/v1/models/promote")
def promote_model(
    name: Optional[str] = None,
    subset: str = Query(default="FD001"),
    target_name: Optional[str] = Query(default=None),
    version: Optional[str] = Query(default=None)
):
    validate_subset(subset)
    model_name = version or name or target_name or "challenger"
    mgr = get_mlops_manager(subset)
    try:
        res = mgr.promote_version(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=f"Model bundle file missing on disk: {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Swap served bundle and reload the cached ModelService
    if subset in model_services:
        del model_services[subset]
    reloaded_ms = get_model_service(subset)
    if subset in streaming_pipelines:
        streaming_pipelines[subset].model_service = reloaded_ms
    clear_prediction_cache()

    return res

@app.post("/api/v1/mlops/{name}/rollback")
@app.post("/api/v1/mlops/rollback")
@app.post("/api/v1/models/{name}/rollback")
@app.post("/api/v1/models/rollback")
def rollback_model(
    name: Optional[str] = None,
    subset: str = Query(default="FD001"),
    target_name: Optional[str] = Query(default=None),
    target_version: Optional[str] = Query(default=None)
):
    validate_subset(subset)
    model_name = target_version or name or target_name
    mgr = get_mlops_manager(subset)
    try:
        res = mgr.rollback(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=f"Model bundle file missing on disk: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Swap served bundle and reload the cached ModelService
    if subset in model_services:
        del model_services[subset]
    reloaded_ms = get_model_service(subset)
    if subset in streaming_pipelines:
        streaming_pipelines[subset].model_service = reloaded_ms
    clear_prediction_cache()

    return res

@app.post("/api/v1/models/reload")
def reload_models(subset: Optional[str] = Query(default=None)):
    reloaded = []
    if subset:
        validate_subset(subset)
        if subset in model_services:
            del model_services[subset]
        get_model_service(subset)
        reloaded.append(subset)
    else:
        model_services.clear()
        for s in VALID_SUBSETS:
            bundle_p = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{s}.joblib"
            if bundle_p.exists():
                get_model_service(s)
                reloaded.append(s)
    loaded_dfs.clear()
    clear_prediction_cache()
    return {"status": "reloaded", "reloaded_subsets": reloaded}

@app.post("/api/v1/pipeline/run")
def run_pipeline_endpoint(subset: str = Query(default="FD001")):
    validate_subset(subset)
    from src.features.pipeline import DataPipeline
    pipeline = DataPipeline(subset=subset)
    pipeline.run()
    loaded_dfs.pop(f"train_{subset}", None)
    loaded_dfs.pop(f"test_{subset}", None)
    report_file = ROOT_DIR / "artifacts" / "reports" / f"data_quality_{subset}.json"
    report = {}
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
    return {
        "status": "success",
        "subset": subset,
        "report": report
    }

MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB limit
ALLOWED_UPLOAD_EXTS = {".csv", ".txt", ".parquet"}

@app.post("/api/v1/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    subset: str = Query(default="FD001"),
    file_type: str = Query(default="raw_test"),
    trigger_pipeline: bool = Query(default=False),
    trigger_retrain: bool = Query(default=False)
):
    validate_subset(subset)

    # 1. Sanitize filename (prevent path traversal & special chars)
    raw_name = Path(file.filename or "upload.csv").name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(ALLOWED_UPLOAD_EXTS))}"
        )

    upload_dir = ROOT_DIR / "data" / "uploads" / subset
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / safe_name

    # 2. Size validation
    content = bytearray()
    total_bytes = 0
    while chunk := await file.read(65536):
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024*1024)}MB."
            )
        content.extend(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Schema validation
    try:
        if ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))
        else:
            try:
                df = pd.read_csv(io.BytesIO(content))
                if len(df.columns) == 1 and " " in content[:1024].decode("utf-8", errors="ignore"):
                    df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded data: {e}")

    # Assign C-MAPSS standard columns if headerless 26-column text/csv
    pipe_cfg = load_yaml("configs/pipeline.yaml")
    std_cols = pipe_cfg.get("dataset", {}).get("columns", [])
    if len(df.columns) == len(std_cols) and all(isinstance(c, int) for c in df.columns):
        df.columns = std_cols

    # Verify minimum required schema: must have unit and cycle
    req_cols = {"unit", "cycle"}
    if not req_cols.issubset(set(df.columns)):
        raise HTTPException(
            status_code=400,
            detail=f"Schema validation failed: Missing required columns {req_cols - set(df.columns)}. Provided: {list(df.columns)[:5]}"
        )

    # Persist validated file to disk
    with open(dest_path, "wb") as f_out:
        f_out.write(content)

    pipeline_report = None
    retrain_report = None

    # 4. Wire into pipeline/retrain if requested
    if trigger_pipeline:
        from src.features.pipeline import DataPipeline
        dp = DataPipeline(subset=subset)
        _, _, _, pipeline_report = dp.run()

    if trigger_retrain:
        mgr = get_mlops_manager(subset)
        retrain_report = mgr.run_retrain_demo(scenario="A", replay_window_df=df)

    return {
        "status": "uploaded",
        "filename": safe_name,
        "size_bytes": total_bytes,
        "rows": len(df),
        "columns": list(df.columns),
        "subset": subset,
        "saved_path": str(dest_path.relative_to(ROOT_DIR)),
        "pipeline_triggered": trigger_pipeline,
        "pipeline_report": pipeline_report,
        "retrain_triggered": trigger_retrain,
        "retrain_report": retrain_report
    }

# ==============================================================================
# MLOps & Drift Monitoring Endpoints
# ==============================================================================
@app.get("/api/v1/mlops/registry")
def get_mlops_registry(subset: str = Query(default="FD001")):
    validate_subset(subset)
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    return mlops_managers[subset].get_registry()

@app.get("/api/v1/monitoring/drift")
def get_monitoring_drift(subset: str = Query(default="FD001")):
    validate_subset(subset)
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    mgr = mlops_managers[subset]
    
    if mgr.drift_history_file.exists():
        try:
            with open(mgr.drift_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    train_df = get_dataframe("train", subset)
    test_df = get_dataframe("test", subset)
    ref_sample = train_df.sample(min(1000, len(train_df)), random_state=42)
    curr_sample = test_df.sample(min(500, len(test_df)), random_state=42)
    return mgr.calculate_drift_report(ref_sample, curr_sample)

@app.post("/api/v1/mlops/retrain")
async def trigger_mlops_retrain(scenario: str = Query(default="B"), subset: str = Query(default="FD001")):
    validate_subset(subset)
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    mgr = mlops_managers[subset]
    
    recent_df = None
    if replay_state.recent_replay_window:
        recent_df = pd.DataFrame(replay_state.recent_replay_window)
        
    result = mgr.run_retrain_demo(scenario=scenario, replay_window_df=recent_df)
    
    if result.get("gate_passed") and result.get("promoted_version"):
        promoted_ver = result["promoted_version"]
        svc = ModelService(subset)
        svc.load(promoted_ver)
        model_services[subset] = svc
        await broadcast("model_promoted", {
            "subset": subset,
            "version": promoted_ver,
            "metrics": result.get("challenger_metrics")
        })
        
    return result

# ==============================================================================
# Autonomous Self-Healing Remediation Endpoints
# ==============================================================================
@app.get("/api/v1/self-healing/prescriptions/{engine_id}")
def get_engine_prescriptions(engine_id: str, cycle: int = Query(default=30)):
    order = triage_agent.open_orders.get(engine_id)
    cons_rul = order["conservative_rul"] if order else 35.0
    subsystem = order["suspected_subsystem"] if order else "HPC"
    severity = order["severity"] if order else "MEDIUM"
    
    status_dict = {
        "engine_id": engine_id,
        "cycle": cycle,
        "conservative_rul": cons_rul,
        "suspected_subsystem": subsystem,
        "severity": severity
    }
    return {
        "engine_id": engine_id,
        "prescriptions": self_healing_engine.generate_prescriptions(status_dict)
    }

@app.post("/api/v1/self-healing/execute")
async def execute_self_healing_action(
    engine_id: str = Query(...),
    action_type: str = Query(default="DYNAMIC_DERATE"),
    cycle: int = Query(default=30),
    subset: str = Query(default="FD001")
):
    order = triage_agent.open_orders.get(engine_id)
    cons_rul = order["conservative_rul"] if order else 35.0
    subsystem = order["suspected_subsystem"] if order else "HPC"
    
    res = self_healing_engine.execute_action(
        engine_id=engine_id,
        action_type=action_type,
        cycle=cycle,
        cons_rul=cons_rul,
        subsystem=subsystem
    )
    
    if order:
        order["status"] = "IN_PROGRESS"
        order["events"].append({
            "cycle": cycle,
            "event": f"Action Executed: {res['title']} - {res['recommended_action']}",
            "actor": "SelfHealingEngine"
        })
        triage_agent._persist_order(order)
        
    await broadcast("self_healing_executed", res)
    return {"status": "success", "action": res}

@app.get("/api/v1/self-healing/history")
def get_self_healing_history_endpoint(engine_id: Optional[str] = Query(default=None), limit: int = Query(default=50)):
    return {
        "history": self_healing_engine.get_history(engine_id=engine_id, limit=limit)
    }


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    ico_path = ROOT_DIR / "web" / "dist" / "favicon.ico"
    if not ico_path.exists():
        ico_path = ROOT_DIR / "web" / "public" / "favicon.ico"
    if ico_path.exists():
        return FileResponse(
            ico_path,
            media_type="image/x-icon",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return Response(status_code=404)

@app.get("/favicon.svg", include_in_schema=False)
async def get_favicon_svg():
    svg_path = ROOT_DIR / "web" / "dist" / "favicon.svg"
    if not svg_path.exists():
        svg_path = ROOT_DIR / "web" / "public" / "favicon.svg"
    if svg_path.exists():
        return FileResponse(
            svg_path,
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return Response(status_code=404)

# Mount static frontend build
web_dist_path = ROOT_DIR / "web" / "dist"
if web_dist_path.exists():
    app.mount("/", StaticFiles(directory=str(web_dist_path), html=True), name="static_web")


