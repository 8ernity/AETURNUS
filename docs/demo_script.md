# AETURNUS Platform Demo Walkthrough Script

This script walks through the core capabilities of the AETURNUS Turbofan Predictive Maintenance platform.

---

## 1. Fleet Overview & Live Telemetry Replay

1. **Navigate to the "Fleet Status" Tab**:
   - Observe the 100 turbofan units operating in the sea-level single-regime dataset (FD001).
   - Point out the conservative RUL (p10 floor), health index (0–1), and active triage badges (`CRITICAL`, `HIGH`, `HEALTHY`).
2. **Start Telemetry Replay**:
   - Click **Start Replay** at 1x or 2x speed.
   - Note the real-time WebSocket telemetry updates, degradation tracking across cycles, and dynamic sparklines.
   - Switch subset to **FD002** or **FD004** to demonstrate handling 6 complex operating regimes with multiple fault modes.

---

## 2. Engine Deep-Dive & Explainability

1. **Select a Critical Engine** (e.g., `FD001-042` or `FD001-034`):
   - Review the Remaining Useful Life degradation curve with the **shaded 80% conformal uncertainty interval [p10–p90]**.
   - Note that ground truth tracks closely with the median p50 estimate.
2. **Signed SHAP Sensor Attribution**:
   - Inspect the **Top Contributing Sensors** panel.
   - Show the signed contribution bars:
     - **Red bars (`shortens_life`)**: Sensors accelerating component wear relative to the healthy baseline (e.g., elevated static pressure Ps30 or high turbine temperature T30).
     - **Blue bars (`prolongs_life`)**: Sensors indicating stable or below-average wear.
   - Note the subsystem tags (HPC, LPT, Fan, LPC) and the disclaimer: *"Subsystem mapping is an indicative association, not a diagnosis. For decision support only."*
3. **Multi-Horizon Failure Risk**:
   - Inspect the $P(\text{fail} \le 10)$, $P(\text{fail} \le 30)$, and $P(\text{fail} \le 60)$ horizon cards with validation-tuned threshold indicators.
4. **Download Maintenance Recommendation Report**:
   - Click **Maintenance Report PDF** to generate an on-the-fly decision support summary.

---

## 3. Triage Agent & Ranked Maintenance Queue

1. **Navigate to the "Work Orders" Tab**:
   - Show the **Ranked Maintenance Priority Queue** table.
   - Demonstrate the **Daily Service Capacity constraint**:
     - Toggle between 3, 5, and 10 engines per shift.
     - Observe how engines are sorted strictly by composite priority score $S_i$:
       $$S_i = 0.40 \cdot \text{RUL Urgency} + 0.30 \cdot P(\text{fail}\le 30) + 0.15 \cdot \text{Anomaly} + 0.15 \cdot \text{Trend}$$
     - Top engines receive the **`ACTIONABLE TODAY`** badge; the remainder fall into the **`BACKLOG`** queue.
2. **Human-in-the-Loop (HITL) Workflow**:
   - Select an order in the Kanban pipeline.
   - Review the deterministic decision trace (e.g., `base:MEDIUM -> M3:fired_30 -> HIGH`).
   - Execute an operator action: **APPROVE**, **OVERRIDE** (change severity), or **ADVANCE** stage through the audit log.

---

## 4. Model Lab & Statistical Rigor

1. **Navigate to the "Model Lab" Tab**:
   - Point out the official **C-MAPSS Benchmark Leaderboard**.
   - Highlight the **Statistical Rigor Cards**:
     - **5-Fold Group CV on Train Units**: Demonstrates out-of-fold generalization without test leakage (e.g. 13.06 ± 0.95 RMSE on FD001).
     - **1,000-Resample Bootstrap 95% CI**: Highlights parameter variance on the test set (`[11.23, 15.39]`).
     - **Split-Conformal Interval Coverage**: Wilson 95% score interval and badge indicator if coverage dips below nominal targets.
     - **Optuna Tuning Study**: Compares baseline CV RMSE with tuned LightGBM parameters.
2. **Architecture Comparison Table**:
   - Compare Ridge regression (linear), HistGradientBoosting, LightGBM (Default), LightGBM (Tuned Champion), 1D-CNN, and Sequence GRU.
   - Highlight the serving champion selection rule: *Selected strictly on train engines via GroupKFold-5, never on the held-out test set.*

---

## 5. MLOps Drift & Autonomous Retraining Gate

1. **Navigate to the "MLOps & Monitoring" Tab**:
   - Review active model registry entries (`v1.0` active champion with immutable data hash `sha256:...`).
   - Inspect sensor distribution drift (Population Stability Index / PSI tracking).
2. **Trigger Challenger Retraining Demo**:
   - Select **Scenario B** (simulated operational drift).
   - Observe real-time progress steps:
     1. Partitioning into 60% champion train, 20% incoming batch, and 20% fixed lifecycle holdout.
     2. Training challenger model on old + new engines.
     3. Strict 3-tier promotion gate evaluation on the fixed holdout set.
     4. Challenger evaluated for promotion or rejected with clear causal reasons.
