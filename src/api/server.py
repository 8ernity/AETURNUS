import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.common.utils import get_root_dir, load_yaml
from src.models.model_service import ModelService
from src.agent.triage_agent import TriageAgent
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
mlops_managers: Dict[str, MLOpsManager] = {}
streaming_pipelines: Dict[str, StreamingEnginePipeline] = {}

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

# Cached processed dataframes
loaded_dfs: Dict[str, pd.DataFrame] = {}

def get_model_service(subset: str = "FD001") -> ModelService:
    if subset not in model_services:
        ms = ModelService(subset)
        try:
            ms.load()
        except Exception:
            proc_train = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
            if proc_train.exists():
                train_df = pd.read_parquet(proc_train)
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
        loaded_dfs[key] = pd.read_parquet(proc_path)
    return loaded_dfs[key]

def get_train_df(subset: str = "FD001") -> pd.DataFrame:
    validate_subset(subset)
    key = f"train_{subset}"
    if key not in loaded_dfs:
        proc_path = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
        if not proc_path.exists():
            from src.features.pipeline import DataPipeline
            DataPipeline(subset).run()
        loaded_dfs[key] = pd.read_parquet(proc_path)
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
                replay_state.current_cycle += 1
                if replay_state.current_cycle > replay_state.max_cycle:
                    replay_state.current_cycle = 15

                cycle = replay_state.current_cycle

                # In lifecycle demo mode, evaluate held-out run-to-failure train engine over full lifespan
                if replay_state.mode == "lifecycle_demo":
                    train_df = get_train_df(current_subset)
                    holdout_units = sorted(train_df["unit"].unique())[int(len(train_df["unit"].unique()) * 0.8):]
                    unit = replay_state.selected_lifecycle_unit
                    if unit not in train_df["unit"].values:
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
                    fleet_predictions = []
                    for u in all_units:
                        row = sub_rows[sub_rows["unit"] == u].iloc[0].to_dict()
                        replay_state.recent_replay_window.append(row)
                        pred = ms.predict_point(row)
                        order, trace = triage_agent.evaluate_engine_cycle(pred)
                        fleet_predictions.append(pred)
                        if order and order.get("created_at_cycle") == cycle:
                            await broadcast("work_order_created", order)

                    if len(replay_state.recent_replay_window) > 1000:
                        replay_state.recent_replay_window = replay_state.recent_replay_window[-1000:]

                    await broadcast("tick", {
                        "cycle": cycle,
                        "engine_count": len(all_units),
                        "predictions": fleet_predictions
                    })

                delay = max(0.05, 1.0 / replay_state.speed)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Replay ticker loop error: {e}")
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    get_model_service(current_subset)
    asyncio.create_task(replay_ticker_loop())

# ----------------- REST Endpoints -----------------

@app.get("/api/v1/health")
def health():
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

    for row_dict in last_rows_df.to_dict("records"):
        u = int(row_dict["unit"])
        max_c = int(row_dict["cycle"])
        pred = ms.predict_point(row_dict)

        order = triage_agent.open_orders.get(pred["engine_id"])
        c_rul = float(pred["conservative_rul"])
        severity = order["severity"] if order else ("CRITICAL" if c_rul <= 15 else "HIGH" if c_rul <= 30 else "MEDIUM" if c_rul <= 60 else "LOW" if c_rul <= 100 else "HEALTHY")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        total_rul += c_rul

        sparkline = history_by_unit.get(u, [float(pred["health_index"])])

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
            "suspected_subsystem": str(pred["top_sensors"][0] if pred.get("top_sensors") else "HPC")
        })

    avg_rul = round(float(total_rul) / max(1, len(engines)), 1)
    critical_count = int(severity_counts["CRITICAL"])
    open_orders_count = int(len(triage_agent.open_orders))
    needs_review_count = int(sum(1 for o in triage_agent.open_orders.values() if o.get("status") == "NEEDS_REVIEW"))

    return {
        "subset": str(subset),
        "kpis": {
            "engines_monitored": int(len(all_units)),
            "critical_count": critical_count,
            "open_work_orders": open_orders_count,
            "exception_queue_size": needs_review_count,
            "mean_conservative_rul": avg_rul,
            "live_model_version": f"rul-{subset.lower()}@champion"
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

    # Top contributing sensors (SHAP approximation)
    sensor_map = load_yaml("configs/sensor_map.yaml")["sensors"]
    shap_bars = []
    for s in pred["top_sensors"]:
        subsystem = sensor_map.get(s, {}).get("subsystem", "HPC")
        dev = abs(float(last_row.get(f"{s}_dev_base", 1.0)))
        shap_bars.append({
            "sensor": s,
            "name": sensor_map.get(s, {}).get("name", s),
            "subsystem": subsystem,
            "contribution": round(min(1.0, dev / 10.0), 3)
        })

    return {
        "engine_id": engine_id,
        "unit": unit_num,
        "subset": subset,
        "current_cycle": int(unit_df.iloc[-1]["cycle"]),
        "prediction": pred,
        "work_order": order,
        "trajectory": trajectory,
        "shap_bars": shap_bars,
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
    return {"status": "success", "action": req.action, "work_order_id": work_order_id}

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
    df = get_test_df(subset)
    ms = get_model_service(subset)
    last_cycles = df.groupby("unit")["cycle"].max()

    created_count = 0
    simulated_severities = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for u in sorted(df["unit"].unique())[:30]:
        max_c = last_cycles[u]
        row = df[(df["unit"] == u) & (df["cycle"] == max_c)].iloc[0].to_dict()
        pred = ms.predict_point(row)
        
        c_rul = pred["conservative_rul"]
        anom = pred["anomaly_score"]

        if c_rul <= req.critical_threshold or (anom >= req.anomaly_threshold and c_rul <= 25):
            simulated_severities["CRITICAL"] += 1
            created_count += 1
        elif c_rul <= req.high_threshold:
            simulated_severities["HIGH"] += 1
            created_count += 1
        elif c_rul <= 60:
            simulated_severities["MEDIUM"] += 1
        else:
            simulated_severities["LOW"] += 1

    return {
        "subset": subset,
        "simulated_params": req.dict(),
        "total_simulated_orders": created_count,
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

# ----------------- MLOps & Monitoring -----------------
@app.get("/api/v1/mlops/registry")
def get_registry(subset: str = Query(default="FD001")):
    validate_subset(subset)
    mgr = get_mlops_manager(subset)
    return mgr.get_registry()

@app.get("/api/v1/monitoring/drift")
def get_drift_report(subset: str = Query(default="FD001")):
    validate_subset(subset)
    mgr = get_mlops_manager(subset)
    # 1. Training reference from processed training dataset
    train_path = ROOT_DIR / "data" / "processed" / subset / "train.parquet"
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        ref_df = train_df.sample(min(1000, len(train_df)), random_state=42)
    else:
        test_df = get_test_df(subset)
        ref_df = test_df.sample(min(500, len(test_df)), random_state=42)

    # 2. Current telemetry from recent replay window if available, else recent streaming test cycles
    if len(replay_state.recent_replay_window) >= 30:
        curr_df = pd.DataFrame(replay_state.recent_replay_window)
    else:
        test_df = get_test_df(subset)
        max_c = test_df["cycle"].max()
        curr_df = test_df[test_df["cycle"] >= max(1, max_c - 25)]

    return mgr.calculate_drift_report(ref_df, curr_df)

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
@app.post("/api/v1/models/{name}/promote")
@app.post("/api/v1/models/promote")
def promote_model(
    name: Optional[str] = None,
    subset: str = Query(default="FD001"),
    target_name: Optional[str] = Query(default=None)
):
    validate_subset(subset)
    model_name = name or target_name or "challenger"
    mgr = get_mlops_manager(subset)
    reg = mgr.get_registry()
    models = reg.get("models", [])
    
    target_idx = None
    for idx, m in enumerate(models):
        if m.get("version") == model_name or m.get("alias") == model_name:
            target_idx = idx
            break
            
    if target_idx is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in registry for subset {subset}")
        
    for m in models:
        if m.get("alias") == "champion":
            m["alias"] = "previous"
            
    promoted = models[target_idx]
    promoted["alias"] = "champion"
    reg["active_champion"] = promoted.get("version")
    
    with open(mgr.registry_file, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
        
    if "model_path" in promoted and (ROOT_DIR / promoted["model_path"]).exists():
        ms = get_model_service(subset)
        bundle_file = ROOT_DIR / promoted["model_path"]
        active_dest = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{subset}.joblib"
        shutil.copyfile(bundle_file, active_dest)
        ms.load()
        
    return {
        "status": "promoted",
        "version": promoted.get("version"),
        "alias": "champion",
        "subset": subset
    }

@app.post("/api/v1/models/{name}/rollback")
@app.post("/api/v1/models/rollback")
def rollback_model(
    name: Optional[str] = None,
    subset: str = Query(default="FD001"),
    target_name: Optional[str] = Query(default=None)
):
    validate_subset(subset)
    model_name = name or target_name
    mgr = get_mlops_manager(subset)
    reg = mgr.get_registry()
    models = reg.get("models", [])
    
    target_idx = None
    if model_name and model_name not in ["rollback", "default"]:
        for idx, m in enumerate(models):
            if m.get("version") == model_name or m.get("alias") == model_name:
                target_idx = idx
                break
    else:
        for idx, m in enumerate(models):
            if m.get("alias") == "previous":
                target_idx = idx
                break
                
    if target_idx is None:
        if len(models) > 1:
            target_idx = 0
        else:
            raise HTTPException(status_code=400, detail="No previous model available to rollback to.")
            
    for m in models:
        if m.get("alias") == "champion":
            m["alias"] = "rolled_back"
            
    target_model = models[target_idx]
    target_model["alias"] = "champion"
    reg["active_champion"] = target_model.get("version")
    
    with open(mgr.registry_file, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
        
    if "model_path" in target_model and (ROOT_DIR / target_model["model_path"]).exists():
        ms = get_model_service(subset)
        bundle_file = ROOT_DIR / target_model["model_path"]
        active_dest = ROOT_DIR / "artifacts" / "models" / f"model_bundle_{subset}.joblib"
        shutil.copyfile(bundle_file, active_dest)
        ms.load()
        
    return {
        "status": "rolled_back",
        "version": target_model.get("version"),
        "alias": "champion",
        "subset": subset
    }

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

@app.post("/api/v1/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    subset: str = Query(default="FD001"),
    file_type: str = Query(default="raw_test")
):
    validate_subset(subset)
    upload_dir = ROOT_DIR / "data" / "uploads" / subset
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename
    
    total_bytes = 0
    with open(dest_path, "wb") as f_out:
        while chunk := await file.read(65536):
            f_out.write(chunk)
            total_bytes += len(chunk)
            
    return {
        "status": "uploaded",
        "filename": file.filename,
        "size_bytes": total_bytes,
        "subset": subset,
        "saved_path": str(dest_path.relative_to(ROOT_DIR))
    }

# Mount static frontend build
web_dist_path = ROOT_DIR / "web" / "dist"
if web_dist_path.exists():
    app.mount("/", StaticFiles(directory=str(web_dist_path), html=True), name="static_web")


