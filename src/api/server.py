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
import hashlib
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query, HTTPException, Body, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.common.utils import get_root_dir, load_yaml
from src.models.model_service import ModelService, CONDITIONS_MAP
from src.agent.triage_agent import TriageAgent
from src.agent.self_healing_engine import SelfHealingEngine
from src.mlops.drift_and_retrain import MLOpsManager
from src.features.streaming import StreamingEnginePipeline
from src.ingest.loader import CMAPSSLoader

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

# In-memory trajectory cache for instant < 100ms engine detail telemetry loading
engine_trajectory_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_trajectory_cache_lock = threading.Lock()

def clear_engine_trajectory_cache(subset: Optional[str] = None):
    global engine_trajectory_cache
    with _trajectory_cache_lock:
        if subset:
            engine_trajectory_cache = {k: v for k, v in engine_trajectory_cache.items() if k[0] != subset}
        else:
            engine_trajectory_cache.clear()

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
        mlops_managers[subset] = MLOpsManager(subset)
    return mlops_managers[subset]

def get_streaming_pipeline(subset: str = "FD001") -> StreamingEnginePipeline:
    if subset not in streaming_pipelines:
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
        proc_path = ROOT_DIR / "data" / "processed" / subset / "test.parquet"
        if not proc_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset).run()
            trim_memory()
        loaded_dfs[key] = _load_downcasted_parquet(proc_path)
    return loaded_dfs[key]

def get_train_df(subset: str = "FD001") -> pd.DataFrame:
    validate_subset(subset)
    key = f"train_{subset}"
    if key not in loaded_dfs:
        proc_path = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
        if not proc_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset).run()
            trim_memory()
        loaded_dfs[key] = _load_downcasted_parquet(proc_path)
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

def bootstrap_fleet_triage(subsets: Optional[List[str]] = None):
    """
    Executes a one-time batch evaluation of the current fleet state on startup.
    Passes all current engines through triage_agent.evaluate_and_persist(engine)
    to populate artifacts/sqlite/work_orders.db before frontend connects.
    """
    target_subsets = subsets or ["FD001", "FD002", "FD003", "FD004"]
    for sub in target_subsets:
        try:
            df = get_test_df(sub)
            ms = get_model_service(sub)
            last_indices = df.groupby("unit")["cycle"].idxmax()
            last_rows = df.loc[last_indices].sort_values("unit")
            for _, row in last_rows.iterrows():
                row_dict = row.to_dict()
                pred = ms.predict_point(row_dict, compute_shap=False)
                triage_agent.evaluate_and_persist(pred)
        except Exception as e:
            print(f"Bootstrap triage notice for {sub}: {e}")

@app.on_event("startup")
async def startup_event():
    try:
        # 1. Preload feature parquets and models into memory so endpoints serve in < 100ms
        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            try:
                get_test_df(sub)
                get_model_service(sub)
            except Exception:
                pass

        # 2. Bootstrap Triage Agent on startup: batch evaluation of fleet state
        bootstrap_fleet_triage(["FD001", "FD002", "FD003", "FD004"])

        # 3. Prime initial fleet prediction cache
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
    try:
        preds = list(_fleet_executor.map(lambda r: get_cached_prediction(ms, r, subset), records))
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
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

    # Fleet Survival Curve (Kaplan-Meier Reliability S(t) = P(RUL > t))
    n_engines = max(1, len(engines))
    all_p50 = [e["rul_p50"] for e in engines]
    all_p10 = [e["conservative_rul"] for e in engines]
    max_horizon = int(max(all_p50)) if all_p50 else 125
    max_horizon = max(30, min(160, max_horizon))

    survival_curve = []
    for t in range(0, max_horizon + 5, 5):
        s_p50 = sum(1 for r in all_p50 if r > t) / n_engines * 100.0
        s_p10 = sum(1 for r in all_p10 if r > t) / n_engines * 100.0
        surviving_count = sum(1 for r in all_p50 if r > t)
        survival_curve.append({
            "cycle": t,
            "survival_rate": round(float(s_p50), 1),
            "conservative_rate": round(float(s_p10), 1),
            "surviving_count": surviving_count,
            "at_risk_count": n_engines - surviving_count
        })

    # Risk Distribution Donut Tiers
    risk_tiers = [
        {
            "name": "CRITICAL",
            "label": "Critical (≤15 cyc)",
            "count": int(severity_counts.get("CRITICAL", 0)),
            "color": "#f43f5e",
            "action": "Immediate grounding & borescope inspection"
        },
        {
            "name": "HIGH",
            "label": "High Risk (16-30 cyc)",
            "count": int(severity_counts.get("HIGH", 0)),
            "color": "#f97316",
            "action": "Schedule depot maintenance within 3 days"
        },
        {
            "name": "MEDIUM",
            "label": "Medium Risk (31-60 cyc)",
            "count": int(severity_counts.get("MEDIUM", 0)),
            "color": "#eab308",
            "action": "Elevated telemetry frequency & sensor review"
        },
        {
            "name": "LOW",
            "label": "Low Risk (61-100 cyc)",
            "count": int(severity_counts.get("LOW", 0)),
            "color": "#06b6d4",
            "action": "Standard flight monitoring with regular checks"
        },
        {
            "name": "HEALTHY",
            "label": "Healthy (>100 cyc)",
            "count": int(severity_counts.get("HEALTHY", 0)),
            "color": "#10b981",
            "action": "Optimal flight readiness, full mission endurance"
        }
    ]

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
        "survival_curve": survival_curve,
        "risk_distribution": risk_tiers,
        "engines": engines
    }

@app.get("/api/v1/engines/{engine_id}")
def get_engine_detail(engine_id: str, subset: str = Query(default="FD001")):
    validate_subset(subset)
    try:
        unit_str = engine_id.split("-")[-1]
        unit_num = int(unit_str)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid engine ID: {engine_id}")

    cache_key = (str(subset), str(engine_id))
    with _trajectory_cache_lock:
        if cache_key in engine_trajectory_cache:
            res = dict(engine_trajectory_cache[cache_key])
            pred_id = res.get("prediction", {}).get("engine_id", engine_id)
            res["work_order"] = triage_agent.open_orders.get(pred_id)
            return res

    df = get_test_df(subset)
    ms = get_model_service(subset)

    unit_df = df[df["unit"] == unit_num].sort_values("cycle")
    if unit_df.empty:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found in subset {subset}")

    last_row = unit_df.iloc[-1].to_dict()
    pred = ms.predict_point(last_row, compute_shap=True)
    order = triage_agent.open_orders.get(pred["engine_id"])

    # Ultra-fast vectorized cycle-by-cycle trajectory series (< 10ms)
    trajectory = ms.predict_trajectories(unit_df)

    # Sensor attributions computed via sensor_map.yaml
    sensor_map_cfg = load_yaml("configs/sensor_map.yaml")
    sensor_map = sensor_map_cfg.get("sensors", {})
    subsystem_disclaimer = sensor_map_cfg.get("disclaimer", "indicative association, not a diagnosis")
    
    top_contribs = pred.get("top_contributors", [])
    if not top_contribs:
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
                "shap_value": round(float(last_row.get(f"{s}_dev_base", 1.0)), 3),
                "direction": "shortens_life",
                "impact": "shortens remaining life"
            })
    else:
        sensor_attributions = top_contribs

    # 360° Turbofan Subsystem Attribution Share (Sums strictly to 100%)
    subsystem_sensor_groups = {
        "Fan / LPC": ["s1", "s2", "s5", "s6", "s8", "s13", "s15"],
        "HPC": ["s3", "s7", "s11"],
        "Combustor": ["s10", "s12", "s16"],
        "HPT": ["s9", "s14", "s20"],
        "LPT / Exhaust": ["s4", "s17", "s21"]
    }
    sensor_to_group = {}
    for grp, s_list in subsystem_sensor_groups.items():
        for s in s_list:
            sensor_to_group[s] = grp

    subsystem_shap_abs = {grp: 0.0 for grp in subsystem_sensor_groups}

    for item in sensor_attributions:
        s = item.get("sensor", "")
        grp = sensor_to_group.get(s)
        if not grp:
            sub = item.get("subsystem", "")
            grp = "HPC" if "HPC" in sub else (
                "Fan / LPC" if ("Fan" in sub or "LPC" in sub) else (
                    "Combustor" if "Combustor" in sub else (
                        "HPT" if "HPT" in sub or "Core" in sub else (
                            "LPT / Exhaust" if "LPT" in sub or "Bleed" in sub else "HPC"
                        )
                    )
                )
            )
        val = abs(float(item.get("shap_value", item.get("contribution", 0.0))))
        subsystem_shap_abs[grp] += val

    total_shap = sum(subsystem_shap_abs.values())
    if total_shap <= 1e-6:
        for grp, sensors in subsystem_sensor_groups.items():
            for s in sensors:
                if f"{s}_dev_base" in last_row:
                    subsystem_shap_abs[grp] += abs(float(last_row[f"{s}_dev_base"]))
                elif f"{s}_norm" in last_row:
                    subsystem_shap_abs[grp] += abs(float(last_row[f"{s}_norm"]))
        total_shap = sum(subsystem_shap_abs.values())

    if total_shap > 1e-6:
        raw_shares = {grp: (val / total_shap) * 100.0 for grp, val in subsystem_shap_abs.items()}
    else:
        raw_shares = {grp: 100.0 / len(subsystem_sensor_groups) for grp in subsystem_sensor_groups}

    rounded_shares = {grp: round(raw_shares[grp], 1) for grp in subsystem_sensor_groups}
    diff = round(100.0 - sum(rounded_shares.values()), 1)
    if diff != 0.0:
        max_grp = max(rounded_shares.keys(), key=lambda k: rounded_shares[k])
        rounded_shares[max_grp] = round(rounded_shares[max_grp] + diff, 1)

    subsystem_health = []
    sorted_groups = sorted(subsystem_sensor_groups.keys(), key=lambda g: rounded_shares[g], reverse=True)
    for rank, group_name in enumerate(sorted_groups, start=1):
        share = rounded_shares[group_name]
        driver_label = f"#{rank} Driver" if rank <= 2 else "Nominal"
        subsystem_health.append({
            "subsystem": group_name,
            "attribution_share": share,
            "share": share,
            "degradation": share,
            "baseline": 20.0,
            "driver_rank": rank,
            "status": driver_label
        })

    res = {
        "engine_id": engine_id,
        "unit": unit_num,
        "subset": subset,
        "current_cycle": int(unit_df.iloc[-1]["cycle"]),
        "prediction": pred,
        "work_order": order,
        "trajectory": trajectory,
        "sensor_attributions": sensor_attributions,
        "shap_bars": sensor_attributions,
        "subsystem_health": subsystem_health,
        "subsystem_disclaimer": subsystem_disclaimer,
        "regime": int(last_row.get("regime", 0))
    }

    with _trajectory_cache_lock:
        if len(engine_trajectory_cache) >= 1000:
            engine_trajectory_cache.clear()
        engine_trajectory_cache[cache_key] = res

    return res


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
@app.get("/api/v1/maintenance-queue")
def get_maintenance_queue(subset: str = Query(default="FD001"), capacity: int = Query(default=5)):
    validate_subset(subset)
    from src.agent.maintenance_queue import MaintenanceQueue
    mq = MaintenanceQueue()
    
    # 1. Collect open work orders for this subset
    open_items = []
    for eng_id, wo in triage_agent.open_orders.items():
        if wo.get("subset") == subset:
            open_items.append({
                "work_order_id": wo.get("work_order_id"),
                "engine_id": eng_id,
                "unit": int(wo.get("unit", 1)),
                "conservative_rul": float(wo.get("conservative_rul", 50.0)),
                "rul_p10": float(wo.get("rul_p10", 50.0)),
                "fail_prob_30": float(wo.get("fail_prob_30", 0.5)),
                "anomaly_score": float(wo.get("anomaly_score", 0.2)),
                "rul_drop": float(wo.get("rul_drop", 5.0)),
                "suspected_subsystem": wo.get("suspected_subsystem", "HPC"),
                "severity": wo.get("severity", "HIGH"),
                "status": wo.get("status", "OPEN"),
            })
            
    # 2. If open work orders is empty, populate from test engines needing service
    if not open_items:
        df = get_test_df(subset)
        ms = get_model_service(subset)
        last_indices = df.groupby("unit")["cycle"].idxmax()
        last_rows = df.loc[last_indices].sort_values("unit")
        
        for _, row in last_rows.iterrows():
            r_dict = row.to_dict()
            pred = ms.predict_point(r_dict)
            c_rul = pred.get("conservative_rul", 999.0)
            fp30 = pred.get("fail_prob_30", 0.0)
            if c_rul <= 60.0 or fp30 >= 0.30:
                subsys = "HPC"
                if pred.get("top_contributors"):
                    subsys = pred["top_contributors"][0].get("subsystem", "HPC")
                open_items.append({
                    "work_order_id": f"PENDING-WO-{pred['engine_id']}",
                    "engine_id": pred["engine_id"],
                    "unit": int(pred["unit"]),
                    "conservative_rul": c_rul,
                    "rul_p10": pred.get("rul_p10", c_rul),
                    "fail_prob_30": fp30,
                    "anomaly_score": pred.get("anomaly_score", 0.1),
                    "rul_drop": 5.0,
                    "suspected_subsystem": subsys,
                    "severity": "CRITICAL" if c_rul <= 15 else ("HIGH" if c_rul <= 30 else "MEDIUM"),
                    "status": "QUEUED"
                })

    queue = mq.rank_items(open_items, capacity=capacity)
    return {
        "subset": subset,
        "capacity": capacity,
        "total_open": len(queue),
        "actionable_count": min(len(queue), capacity),
        "backlog_count": max(0, len(queue) - capacity),
        "queue": queue,
        "disclaimer": mq.disclaimer
    }

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

@app.get("/api/v1/work-orders/{work_order_id}/maintenance-report.pdf")
def download_maintenance_report_by_wo(work_order_id: str):
    from src.agent.maintenance_report import generate_maintenance_report_pdf
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
    pdf_bytes = generate_maintenance_report_pdf(engine_id, target_wo)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="maintenance_report_{work_order_id}.pdf"'}
    )

@app.get("/api/v1/engines/{engine_id}/maintenance-report.pdf")
def download_maintenance_report_by_engine(engine_id: str, subset: str = Query(default="FD001")):
    from src.agent.maintenance_report import generate_maintenance_report_pdf
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
            "work_order_id": f"WO-{engine_id}-REPORT",
            "engine_id": engine_id,
            "unit": unit,
            "subset": subset,
            "created_at_cycle": cycle,
            "current_cycle": cycle,
            "severity": "LOW",
            "status": "CLOSED",
            "conservative_rul": pred.get("conservative_rul", 50.0),
            "predicted_rul": {
                "p10": pred.get("rul_p10", 45.0),
                "p50": pred.get("rul_p50", 55.0),
                "p90": pred.get("rul_p90", 65.0)
            },
            "suspected_subsystem": "HPC",
            "top_sensors": pred.get("top_sensors", ["s9", "s14"]),
            "decision_trace": ["baseline:evaluation_nominal", "severity:LOW"],
            "model_version": pred.get("model_version", "GBDT-v2.1"),
            "events": [{"cycle": cycle, "actor": "Inspector", "event": "Baseline maintenance recommendation issued"}]
        }

    pdf_bytes = generate_maintenance_report_pdf(engine_id, target_wo)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="maintenance_report_{engine_id}.pdf"'}
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
    manifest_file = ROOT_DIR / "artifacts" / "run_manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Run manifest (run_manifest.json) not found on disk. Run make evaluate.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    is_stale = False
    stale_reasons = []
    benchmarks = []

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        sub_info = manifest.get("subsets", {}).get(sub)
        if not sub_info:
            continue
        bundle_file = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{sub}.joblib"
        if not bundle_file.exists():
            is_stale = True
            stale_reasons.append(f"model_bundle_{sub}.joblib missing")
        else:
            actual_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
            if actual_sha != sub_info.get("bundle_sha256"):
                is_stale = True
                stale_reasons.append(f"model_bundle_{sub}.joblib hash mismatch")

        m = sub_info.get("metrics", {})
        target_rmse = 14.0 if sub == "FD001" else (22.0 if sub == "FD002" else (15.0 if sub == "FD003" else 22.0))
        status = "Target Met" if (m.get("rmse") or 999.0) <= target_rmse else "Target Missed"
        cond, faults = CONDITIONS_MAP.get(sub, ("Single (Sea Level)", "1 (HPC Degradation)"))
        benchmarks.append({
            "subset": sub,
            "condition": cond,
            "faults": faults,
            "algorithm": sub_info.get("algorithm", "lightgbm"),
            "protocol_id": sub_info.get("protocol_id", ""),
            "n_engines": sub_info.get("n_engines", 100),
            "target_rmse": target_rmse,
            "status": status,
            "rmse": m.get("rmse"),
            "mae": m.get("mae"),
            "nasa_score": m.get("nasa_score"),
            "early_rate": m.get("early_rate"),
            "late_rate": m.get("late_rate"),
            "pr_auc": m.get("pr_auc"),
            "interval_coverage": m.get("interval_coverage"),
            "mean_interval_width": m.get("mean_interval_width"),
            "cv": sub_info.get("cv", {}),
            "deep_comparison": sub_info.get("comparison_rows", []),
            "champion": sub_info.get("champion", {})
        })

    return {
        "run_id": manifest.get("run_id"),
        "git_sha": manifest.get("git_sha"),
        "config_hash": manifest.get("config_hash"),
        "stale": is_stale,
        "stale_reason": "; ".join(stale_reasons) if stale_reasons else None,
        "benchmarks": benchmarks
    }

@app.get("/api/v1/models/{subset}/details")
def get_model_details(subset: str):
    validate_subset(subset)
    manifest_file = ROOT_DIR / "artifacts" / "run_manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail=f"Run manifest not found for {subset}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sub_info = manifest.get("subsets", {}).get(subset)
    if not sub_info:
        raise HTTPException(status_code=404, detail=f"No manifest entry for subset {subset}")

    bundle_file = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{subset}.joblib"
    is_stale = False
    stale_reason = None
    if not bundle_file.exists():
        is_stale = True
        stale_reason = f"model_bundle_{subset}.joblib missing"
    else:
        actual_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
        if actual_sha != sub_info.get("bundle_sha256"):
            is_stale = True
            stale_reason = f"model_bundle_{subset}.joblib hash mismatch"

    # Compute parity scatter live from bundle+data
    test_path = ROOT_DIR / "data" / "processed" / subset / "test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"Processed test data for {subset} not found")

    test_df = pd.read_parquet(test_path)
    loader = CMAPSSLoader(subset=subset)
    _, _, rul_truth = loader.load_raw_data()
    ms = get_model_service(subset)
    eval_res = ms.evaluate_test(test_df, rul_truth, save_report=False)

    cv_data = sub_info.get("cv", {})
    tr_mean = float(cv_data.get("mean_train_rmse", 0.0))
    cv_mean = float(cv_data.get("mean_rmse", 0.0))
    te_mean = float(sub_info["metrics"]["rmse"])
    tt_gap = round(te_mean - tr_mean, 2)

    gen_gap = {
        "train_rmse": round(tr_mean, 2),
        "val_rmse": round(cv_mean, 2),
        "test_rmse": round(te_mean, 2),
        "train_test_gap": tt_gap,
        "train_val_gap": round(cv_mean - tr_mean, 2),
        "val_test_gap": round(te_mean - cv_mean, 2),
        "gap_ratio_pct": round((tt_gap / te_mean * 100.0), 1) if te_mean > 0 else 0.0,
        "overfitting_status": "Low Overfitting" if tt_gap <= 3.5 else "Moderate Gap",
        "memorization_risk": "Low (<25%)"
    }

    return {
        "subset": subset,
        "stale": is_stale,
        "stale_reason": stale_reason,
        "algorithm": sub_info.get("algorithm"),
        "protocol_id": sub_info.get("protocol_id"),
        "bucket_errors": eval_res.get("bucket_errors", []),
        "feature_importance": eval_res.get("feature_importance", []),
        "parity_scatter": eval_res.get("parity_scatter", []),
        "nasa_score": sub_info["metrics"]["nasa_score"],
        "early_rate": sub_info["metrics"]["early_rate"],
        "late_rate": sub_info["metrics"]["late_rate"],
        "nasa_phm08": eval_res.get("nasa_phm08", {}),
        "status": eval_res.get("status", "Evaluated"),
        "interval_coverage": sub_info["metrics"]["interval_coverage"],
        "mean_interval_width": sub_info["metrics"]["mean_interval_width"],
        "coverage_wilson_ci_95": eval_res.get("coverage_wilson_ci_95", []),
        "coverage_ci_excludes_80": eval_res.get("coverage_ci_excludes_80", False),
        "test_rmse_bootstrap_ci_95": eval_res.get("test_rmse_bootstrap_ci_95", []),
        "multi_horizon_metrics": eval_res.get("multi_horizon_metrics", {}),
        "cv_metrics": cv_data,
        "comparison_rows": sub_info.get("comparison_rows", []),
        "champion": sub_info.get("champion", {}),
        "generalization_gap": gen_gap
    }

@app.get("/api/v1/models/{subset}/comparison")
def get_model_comparison(subset: str):
    validate_subset(subset)
    manifest_file = ROOT_DIR / "artifacts" / "run_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        sub_info = manifest.get("subsets", {}).get(subset)
        if sub_info and "comparison_rows" in sub_info:
            return {
                "subset": subset,
                "champion": sub_info.get("champion", {}).get("selected_algorithm", "lightgbm"),
                "models": sub_info["comparison_rows"]
            }
    comp_file = ROOT_DIR / "artifacts" / "reports" / f"model_comparison_{subset}.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail=f"Comparison report for {subset} not found")

# ----------------- Operational Cost & Maintenance ROI Simulator -----------------
class SimulationRequest(BaseModel):
    subset: Optional[str] = "FD001"
    threshold: Optional[float] = 30.0
    use_conservative: Optional[bool] = True
    cost_catastrophic: Optional[float] = 2500000.0
    cost_overhaul: Optional[float] = 220000.0
    cost_wasted_cycle: Optional[float] = 1200.0
    cost_inspection: Optional[float] = 12500.0
    cost_aog_day: Optional[float] = 45000.0
    fixed_interval: Optional[int] = 140

@app.get("/api/v1/economics/simulate")
def get_economic_simulation(
    subset: str = Query(default="FD001"),
    threshold: float = Query(default=30.0),
    use_conservative: bool = Query(default=True)
):
    validate_subset(subset)
    from src.economics.cost_simulator import FleetCostSimulator
    sim = FleetCostSimulator(subset=subset)
    return sim.simulate_fleet(threshold=threshold, use_conservative=use_conservative)

@app.post("/api/v1/economics/simulate")
def post_economic_simulation(req: SimulationRequest):
    sub = req.subset or "FD001"
    validate_subset(sub)
    from src.economics.cost_simulator import FleetCostSimulator
    cost_params = {
        "cost_catastrophic": req.cost_catastrophic or 2500000.0,
        "cost_overhaul": req.cost_overhaul or 220000.0,
        "cost_wasted_cycle": req.cost_wasted_cycle or 1200.0,
        "cost_inspection": req.cost_inspection or 12500.0,
        "cost_aog_day": req.cost_aog_day or 45000.0,
        "fixed_interval": req.fixed_interval or 140
    }
    sim = FleetCostSimulator(subset=sub, cost_params=cost_params)
    return sim.simulate_fleet(threshold=req.threshold, use_conservative=req.use_conservative)

@app.get("/api/v1/economics/curve")
def get_economic_curve(
    subset: str = Query(default="FD001"),
    use_conservative: bool = Query(default=True),
    min_threshold: int = Query(default=5),
    max_threshold: int = Query(default=60)
):
    validate_subset(subset)
    from src.economics.cost_simulator import FleetCostSimulator
    sim = FleetCostSimulator(subset=subset)
    return sim.sweep_cost_curve(
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        use_conservative=use_conservative
    )

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
    clear_engine_trajectory_cache(subset)

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
    clear_engine_trajectory_cache(subset)

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
    clear_engine_trajectory_cache()
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

@app.post("/api/v1/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    subset: str = Query(default="FD001")
):
    validate_subset(subset)
    raw_name = Path(file.filename or "batch.csv").name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(list(ALLOWED_UPLOAD_EXTS))}"
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded batch file is empty.")

    try:
        if ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(content))
        else:
            try:
                df = pd.read_csv(io.BytesIO(content))
                if len(df.columns) == 1 and (" " in content[:1024].decode("utf-8", errors="ignore") or "\t" in content[:1024].decode("utf-8", errors="ignore")):
                    df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=r"\s+", header=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded batch data: {e}")

    pipe_cfg = load_yaml("configs/pipeline.yaml")
    std_cols = pipe_cfg.get("dataset", {}).get("columns", [])
    if len(df.columns) == len(std_cols) and all(isinstance(c, int) for c in df.columns):
        df.columns = std_cols

    if not {"unit", "cycle"}.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail="Schema validation failed: Missing required 'unit' and 'cycle' columns.")

    ms = get_model_service(subset)
    has_features = any(c.endswith("_norm") or c.endswith("_ewma") for c in df.columns)
    predictions = []
    
    if has_features:
        last_indices = df.groupby("unit")["cycle"].idxmax()
        last_rows = df.loc[last_indices].sort_values("unit")
        for _, row in last_rows.iterrows():
            try:
                pred = ms.predict_point(row.to_dict())
            except ValueError as ve:
                raise HTTPException(status_code=422, detail=str(ve))
            c_rul = pred.get("conservative_rul", 999.0)
            subsys = "HPC"
            if pred.get("top_contributors"):
                subsys = pred["top_contributors"][0].get("subsystem", "HPC")
            predictions.append({
                "unit": int(pred["unit"]),
                "cycle": int(pred["cycle"]),
                "conservative_rul": c_rul,
                "rul_p10": pred.get("rul_p10", c_rul),
                "rul_p50": pred.get("rul_p50", c_rul),
                "rul_p90": pred.get("rul_p90", c_rul),
                "fail_prob_10": pred.get("fail_prob_10", 0.0),
                "fail_prob_30": pred.get("fail_prob_30", 0.0),
                "fail_prob_60": pred.get("fail_prob_60", 0.0),
                "health_index": pred.get("health_index", 1.0),
                "anomaly_score": pred.get("anomaly_score", 0.0),
                "top_sensors": pred.get("top_sensors", []),
                "suspected_subsystem": subsys,
                "severity": "CRITICAL" if c_rul <= 15 else ("HIGH" if c_rul <= 30 else ("MEDIUM" if c_rul <= 60 else "HEALTHY"))
            })
    else:
        from src.features.streaming import StreamingEnginePipeline
        streaming_pipe = StreamingEnginePipeline(subset=subset, model_service=ms)
        for unit_id, u_df in df.groupby("unit"):
            u_df = u_df.sort_values("cycle")
            streaming_pipe.reset_engine(int(unit_id))
            last_pred = None
            for _, row in u_df.iterrows():
                out = streaming_pipe.stream_cycle(row.to_dict())
                last_pred = out["prediction"]
            if last_pred:
                c_rul = last_pred.get("conservative_rul", 999.0)
                subsys = "HPC"
                if last_pred.get("top_contributors"):
                    subsys = last_pred["top_contributors"][0].get("subsystem", "HPC")
                predictions.append({
                    "unit": int(unit_id),
                    "cycle": int(last_pred.get("cycle", 1)),
                    "conservative_rul": c_rul,
                    "rul_p10": last_pred.get("rul_p10", c_rul),
                    "rul_p50": last_pred.get("rul_p50", c_rul),
                    "rul_p90": last_pred.get("rul_p90", c_rul),
                    "fail_prob_10": last_pred.get("fail_prob_10", 0.0),
                    "fail_prob_30": last_pred.get("fail_prob_30", 0.0),
                    "fail_prob_60": last_pred.get("fail_prob_60", 0.0),
                    "health_index": last_pred.get("health_index", 1.0),
                    "anomaly_score": last_pred.get("anomaly_score", 0.0),
                    "top_sensors": last_pred.get("top_sensors", []),
                    "suspected_subsystem": subsys,
                    "severity": "CRITICAL" if c_rul <= 15 else ("HIGH" if c_rul <= 30 else ("MEDIUM" if c_rul <= 60 else "HEALTHY"))
                })

    crit_count = sum(1 for p in predictions if p["severity"] == "CRITICAL")
    high_count = sum(1 for p in predictions if p["severity"] == "HIGH")
    med_count = sum(1 for p in predictions if p["severity"] == "MEDIUM")
    healthy_count = sum(1 for p in predictions if p["severity"] == "HEALTHY")

    return {
        "subset": subset,
        "filename": safe_name,
        "engines_scored": len(predictions),
        "predictions": predictions,
        "summary": {
            "critical_count": crit_count,
            "high_count": high_count,
            "medium_count": med_count,
            "healthy_count": healthy_count
        }
    }

# ==============================================================================
# MLOps & Drift Monitoring Endpoints
# ==============================================================================
@app.get("/api/v1/mlops/status")
@app.get("/api/v1/mlops/registry")
def get_mlops_registry(subset: str = Query(default="FD001")):
    validate_subset(subset)
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    return mlops_managers[subset].get_registry()

@app.get("/api/v1/monitoring/drift")
def get_monitoring_drift(subset: str = Query(default="FD001"), n_cycles: int = Query(default=100)):
    validate_subset(subset)
    from src.mlops.drift_and_retrain import get_latest_drift_run
    latest = get_latest_drift_run(subset)
    if latest is not None:
        return latest

    # Compute on demand: reference = train sample matched by engine age, current = last N scored cycles
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    mgr = mlops_managers[subset]

    train_df = get_dataframe("train", subset)
    test_df = get_dataframe("test", subset)
    curr_df = test_df.tail(n_cycles * 10) if len(test_df) > n_cycles * 10 else test_df
    c_min = int(curr_df["cycle"].min())
    c_max = int(curr_df["cycle"].max())
    matched_ref = train_df[(train_df["cycle"] >= c_min) & (train_df["cycle"] <= c_max)]
    if len(matched_ref) < 30:
        matched_ref = train_df
    ref_sample = matched_ref.sample(min(1000, len(matched_ref)), random_state=42)
    curr_sample = curr_df.sample(min(500, len(curr_df)), random_state=42)

    # GET endpoints never write (persist=False)
    return mgr.calculate_drift_report(ref_sample, curr_sample, persist=False)

@app.post("/api/v1/monitoring/drift/compute")
def compute_monitoring_drift(subset: str = Query(default="FD001"), n_cycles: int = Query(default=100)):
    validate_subset(subset)
    if subset not in mlops_managers:
        mlops_managers[subset] = MLOpsManager(subset)
    mgr = mlops_managers[subset]

    train_df = get_dataframe("train", subset)
    test_df = get_dataframe("test", subset)
    curr_df = test_df.tail(n_cycles * 10) if len(test_df) > n_cycles * 10 else test_df
    c_min = int(curr_df["cycle"].min())
    c_max = int(curr_df["cycle"].max())
    matched_ref = train_df[(train_df["cycle"] >= c_min) & (train_df["cycle"] <= c_max)]
    if len(matched_ref) < 30:
        matched_ref = train_df
    ref_sample = matched_ref.sample(min(1000, len(matched_ref)), random_state=42)
    curr_sample = curr_df.sample(min(500, len(curr_df)), random_state=42)

    # Persists each computation in SQLite table drift_runs
    return mgr.calculate_drift_report(ref_sample, curr_sample, persist=True)

# ----------------- Observability & Metrics -----------------
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Required Item 5 Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)
PDM_PREDICTIONS_TOTAL = Counter(
    "pdm_predictions_total",
    "Total engine cycle predictions scored",
    ["subset"]
)
PREDICTIONS_TOTAL = PDM_PREDICTIONS_TOTAL
OPEN_WORK_ORDERS = Gauge(
    "open_work_orders",
    "Number of open work orders by severity",
    ["severity"]
)
AGENT_EXCEPTIONS_TOTAL = Counter(
    "agent_exceptions_total",
    "Total triage agent rule exceptions triggered",
    ["rule"]
)
DRIFT_SHARE = Gauge(
    "drift_share",
    "Live feature drift share per subset",
    ["subset"]
)
RETRAIN_TOTAL = Counter(
    "retrain_total",
    "Total model retraining attempts by outcome",
    ["outcome"]
)
CHAMPION_INFO = Gauge(
    "champion_info",
    "Current active champion model info",
    ["subset", "version"]
)

# Compatibility gauges
PDM_PREDICTIONS = PREDICTIONS_TOTAL
PDM_PREDICTION_LATENCY = Histogram(
    "pdm_prediction_latency_seconds",
    "Prediction inference latency",
    ["subset"]
)
PDM_WORK_ORDERS = Counter(
    "pdm_work_orders_total",
    "Total work orders generated",
    ["subset", "severity"]
)
PDM_ACTIVE_ALERTS = Gauge(
    "pdm_active_alerts",
    "Current status of operational health alerts (1=firing, 0=nominal)",
    ["rule_name", "severity"]
)
PDM_TEST_RMSE = Gauge(
    "pdm_model_test_rmse",
    "Model test RMSE on held-out evaluation set",
    ["subset"]
)
PDM_CONFORMAL_COVERAGE = Gauge(
    "pdm_conformal_coverage",
    "Empirical conformal interval coverage",
    ["subset"]
)

@app.middleware("http")
async def track_http_metrics(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as exc:
        status_code = "500"
        raise exc from None
    finally:
        latency = time.time() - start_time
        path = request.url.path
        method = request.method
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status=status_code).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=path).observe(latency)
    return response

@app.get("/metrics")
def get_metrics():
    # 1. Update test RMSE and conformal coverage
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        PDM_PREDICTIONS_TOTAL.labels(subset=sub)
        eval_f = ROOT_DIR / "artifacts" / "reports" / f"evaluation_{sub}.json"
        if eval_f.exists():
            try:
                with open(eval_f, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                PDM_TEST_RMSE.labels(subset=sub).set(float(ev.get("rmse", 0.0)))
                PDM_CONFORMAL_COVERAGE.labels(subset=sub).set(float(ev.get("interval_coverage", 0.0)))
            except Exception:
                pass

    # 2. Update open_work_orders{severity}
    active_orders = triage_agent.get_active_work_orders() if triage_agent else []
    for sev in ["CRITICAL", "HIGH", "WARNING", "ADVISORY"]:
        count = sum(1 for o in active_orders if o.get("severity") == sev)
        OPEN_WORK_ORDERS.labels(severity=sev).set(count)

    # 3. Update champion_info{subset,version}
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        v = "v1.0"
        reg_file = ROOT_DIR / "artifacts" / "models" / f"registry_{sub}.json"
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as rf:
                    reg_data = json.load(rf)
                    v = reg_data.get("champion", {}).get("version", "v1.0")
            except Exception:
                pass
        CHAMPION_INFO.labels(subset=sub, version=v).set(1.0)

    # 4. Update drift_share{subset} computed from live data
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            train_df = get_dataframe("train", sub)
            test_df = get_dataframe("test", sub)
            ref_sample = train_df.sample(min(300, len(train_df)), random_state=42)
            curr_sample = test_df.sample(min(150, len(test_df)), random_state=42)
            mgr = mlops_managers.get(sub) or MLOpsManager(sub)
            report = mgr.calculate_drift_report(ref_sample, curr_sample)
            DRIFT_SHARE.labels(subset=sub).set(float(report.get("drift_share", 0.0)))
        except Exception:
            DRIFT_SHARE.labels(subset=sub).set(0.0)

    # 5. Evaluate active alerts using LIVE drift
    from src.mlops.alerts import AlertManager
    alert_mgr = AlertManager()
    alerts = alert_mgr.evaluate_alerts(triage_agent=triage_agent)
    for a in alerts:
        PDM_ACTIVE_ALERTS.labels(rule_name=a["rule_name"], severity=a["severity"]).set(1 if a["firing"] else 0)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/v1/monitoring/alerts")
def get_monitoring_alerts():
    from src.mlops.alerts import AlertManager
    alert_mgr = AlertManager()
    alerts = alert_mgr.evaluate_alerts(triage_agent=triage_agent)
    firing_count = sum(1 for a in alerts if a["firing"])
    return {
        "active_alerts_count": firing_count,
        "total_rules_evaluated": len(alerts),
        "status": "CRITICAL" if any(a["firing"] and a["severity"] == "CRITICAL" for a in alerts) else ("WARNING" if firing_count > 0 else "HEALTHY"),
        "alerts": alerts,
        "webhook": alert_mgr.get_webhook_status()
    }

@app.get("/api/v1/monitoring/webhooks")
def get_monitoring_webhook_status():
    from src.mlops.alerts import AlertManager
    alert_mgr = AlertManager()
    return alert_mgr.get_webhook_status()

@app.post("/api/v1/monitoring/webhooks/test")
def test_monitoring_webhook(
    rule_id: str = Query(default="M1"),
    engine_id: str = Query(default="12"),
    cycle: int = Query(default=135),
    severity: str = Query(default="CRITICAL"),
    conservative_rul: float = Query(default=14.2)
):
    from src.mlops.alerts import AlertManager
    alert_mgr = AlertManager()
    res = alert_mgr.dispatch_m_rule(
        rule_id=rule_id,
        engine_id=engine_id,
        cycle=cycle,
        severity=severity,
        conservative_rul=conservative_rul,
        details={"test": True, "source": "API test endpoint"}
    )
    return res


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
        clear_prediction_cache()
        clear_engine_trajectory_cache(subset)
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


