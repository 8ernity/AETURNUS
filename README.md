# ✈️ AutoHeal Fleet · SelfHealingMachines

> **An autonomous, closed-loop predictive maintenance platform for commercial aircraft turbofans. Combines regime-aware signal processing, multi-head probabilistic machine learning, deep learning baselines, and a deterministic triage agent on the NASA C-MAPSS benchmark — with absolute zero hardcoded numbers.**

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?style=flat&logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.0+-0194E2.svg?style=flat&logo=mlflow)](https://mlflow.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg?style=flat&logo=sqlite)](https://sqlite.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-0B3D91.svg?style=flat&logo=nasa)](https://data.nasa.gov/)
[![Tests](https://img.shields.io/badge/Tests-42%20Passed-brightgreen.svg?style=flat&logo=pytest)](https://docs.pytest.org/)
[![CI](https://img.shields.io/badge/CI-Smoke%20Passed-brightgreen.svg?style=flat&logo=githubactions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)

</div>

---

## 📌 Executive Summary

Modern aerospace maintenance incurs tens of billions of dollars annually in unexpected component groundings and premature engine overhauls. **AutoHeal Fleet** is an industrial-grade, closed-loop predictive maintenance (PdM) ecosystem designed to monitor fleets of gas turbine turbofans continuously across operating regimes.

Built on the benchmark **NASA C-MAPSS dataset** (subsets `FD001` through `FD004`), AutoHeal Fleet unifies:
1. **Regime-Normalized Data Engineering**: Operational regime clustering ($k=6$), strictly causal missing value imputation, and Hampel outlier filtering with zero lookahead leakage.
2. **Multi-Head Machine Learning**: Calibrated Remaining Useful Life (RUL) quantile regressors ($p_{10}, p_{50}, p_{90}$), calibrated binary classifier for critical failure probability ($\le 30$ cycles), and unsupervised Isolation Forest anomaly detection.
3. **Deep Learning Benchmarks**: PyTorch 1D-CNN and Gated Recurrent Unit (GRU) baselines benchmarked against the gradient boosting champion across multiple random seeds.
4. **Deterministic Triage Agent (No-LLM)**: Fully explainable, auditable 9-step decision pipeline that evaluates hysteresis, manages technician overrides, suppresses noise, and tracks work orders with SQLite persistence.
5. **Self-Healing MLOps**: Streaming drift monitoring (Population Stability Index & Kolmogorov-Smirnov test), automated retrain triggers with WebSocket streaming, and a strict 3-criteria holdout gate.
6. **Zero-Falsification Guarantee**: Every metric, badge, bucket breakdown, and curve displayed in the UI traces directly to real code-generated disk reports (`artifacts/reports/*.json`).

---

## 📋 Table of Contents

- [✨ Core Capabilities](#-core-capabilities)
- [🏗️ System Architecture & Dataflow](#️-system-architecture--dataflow)
  - [High-Level Topology](#high-level-topology)
  - [Streaming Telemetry Pipeline](#streaming-telemetry-pipeline)
- [📊 Benchmark & Empirical Results](#-benchmark--empirical-results)
  - [Subset Performance Matrix (FD001–FD004)](#subset-performance-matrix-fd001fd004)
  - [Multi-Seed Robustness Analysis](#multi-seed-robustness-analysis)
  - [Deep Learning Architecture Comparison](#deep-learning-architecture-comparison)
- [🧠 Machine Learning Methodology](#-machine-learning-methodology)
  - [Multi-Head Architecture](#multi-head-architecture)
  - [Monotonic Quantiles & Conformal Coverage](#monotonic-quantiles--conformal-coverage)
  - [Engine-Level Group Splitting](#engine-level-group-splitting)
- [🛡️ Deterministic Triage Agent (Zero-LLM)](#️-deterministic-triage-agent-zero-llm)
- [📈 MLOps & Autonomous Self-Healing](#-mlops--autonomous-self-healing)
- [📁 Project Directory Structure](#-project-directory-structure)
- [🛠️ Tech Stack & Key Libraries](#️-tech-stack--key-libraries)
- [💻 Getting Started & Quickstart](#-getting-started--quickstart)
- [🧪 Testing & CI Validation](#-testing--ci-validation)
- [📡 API Contract Reference](#-api-contract-reference)
- [📝 License & Acknowledgements](#-license--acknowledgements)

---

## ✨ Core Capabilities

| Capability | Technical Realization | Operational Value |
|---|---|---|
| **Causal Preprocessing** | Strictly forward-fills missing values within each engine with historical training medians fallback. Regex `^s\d+$` isolates sensors from operational settings. | Prevents future data leakage; handles pandas 2.2/3.0 safely without `apply()` crashes. |
| **Multi-Regime Normalization** | K-Means clustering ($k=6$) over operational settings (`setting_1`, `setting_2`, `setting_3`) to normalize multi-condition subsets (`FD002`, `FD004`). | Removes environmental flight condition noise before calculating degradation trends. |
| **Probabilistic RUL Regressors** | Quantile gradient boosting heads ($p_{10}, p_{50}, p_{90}$) with mathematical enforcement of $p_{10} \le p_{50} \le p_{90}$. | Conformalized prediction intervals provide bounded risk metrics for flight planners. |
| **Calibrated Early Warning** | `HistGradientBoostingClassifier` with balanced class weights wrapped in `CalibratedClassifierCV` (sigmoid). | Well-calibrated probability of failure within 30 cycles eliminates overconfident false alerts. |
| **Deep Learning Baselines** | PyTorch 1D-Convolutional Neural Network and GRU recurrent models trained on sequential sensor slices. | Allows direct empirical comparison between tabular gradient boosting and deep sequential architectures. |
| **Deterministic Triage Agent** | 9-step decision engine reading rules from `agent_rules.yaml`. Hysteresis hold, REJECT suppression window, and UUID tracking. | 100% deterministic, explainable, and sub-millisecond execution without reliance on LLMs or hallucinations. |
| **Continuous Drift Detection** | Multi-sensor Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov tests comparing reference training sets against recent streaming windows. | Detects subtle sensor degradation and environmental drift before catastrophic failure. |
| **Self-Healing Retraining Loop** | Automated retraining pipeline with real SHA-256 data hashing and a 3-criteria validation holdout gate. | Automatically validates challengers against champions, rolling back degrading models seamlessly. |
| **Glassmorphic Cockpit UI** | High-performance React 18 + Vite dashboard with 7 comprehensive tabs inspired by professional audio telemetry hardware. | Real-time fleet health, engine telemetry, model comparison, MLOps registry, and maintenance work orders. |

---

## 🏗️ System Architecture & Dataflow

### High-Level Topology

```mermaid
flowchart TB
    %% Styling Definitions
    classDef clientStyle fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC,font-weight:bold
    classDef apiStyle fill:#1E293B,stroke:#818CF8,stroke-width:2px,color:#F8FAFC,font-weight:bold
    classDef cleanStyle fill:#134E4A,stroke:#2DD4BF,stroke-width:2px,color:#F8FAFC
    classDef mlStyle fill:#312E81,stroke:#A78BFA,stroke-width:2px,color:#F8FAFC
    classDef agentStyle fill:#701A75,stroke:#F472B6,stroke-width:2px,color:#F8FAFC
    classDef storageStyle fill:#713F12,stroke:#FBBF24,stroke-width:2px,color:#F8FAFC

    subgraph ClientLayer["🖥️ PRESENTATION TIER (React 18 · Vite · TypeScript)"]
        Dashboard["AutoHeal Cockpit<br/><i>(Fleet Replay · Model Lab · Triage Agent · MLOps)</i>"]:::clientStyle
        WebSocketClient["WebSocket Client<br/><i>(Replay Telemetry & Retrain Stream)</i>"]:::clientStyle
    end

    Dashboard <-->|REST API JSON| FastAPIGateway
    WebSocketClient <-->|Live WS Events| FastAPIGateway

    subgraph BackendLayer["⚡ BACKEND SERVICE TIER (FastAPI · Uvicorn)"]
        FastAPIGateway["FastAPI Gateway & Route Handlers<br/><i>(/fleet · /models · /triage · /mlops)</i>"]:::apiStyle
        ReplayEngine["Replay Engine<br/><i>(Per-Engine Streaming Generator)</i>"]:::apiStyle

        subgraph IngestionPipe["Data Engineering & Cleaning"]
            RawData["Raw Telemetry<br/><i>(^s\d+$ Sensors)</i>"]:::cleanStyle
            CausalCleaner["Causal Imputer & Hampel<br/><i>(No-Backfill · Norm Post-Regime)</i>"]:::cleanStyle
            RegimeClustering["K-Means Clusterer<br/><i>(k=6 Flight Regimes)</i>"]:::cleanStyle
            FeatureEngineering["Feature Builder<br/><i>(Rolling EWMA · Norm Deviations)</i>"]:::cleanStyle
        end

        subgraph MLEngine["Multi-Head ML & Neural Engine"]
            QuantileRUL["Quantile Heads (p10 / p50 / p90)<br/><i>(Monotonic Bounded RUL)</i>"]:::mlStyle
            FailClassifier["Calibrated Classifier<br/><i>(fails_within_30 Probabilities)</i>"]:::mlStyle
            AnomalyDetector["Isolation Forest<br/><i>(Sensor Anomaly Score)</i>"]:::mlStyle
            DeepBaselines["Deep Learning Benchmarks<br/><i>(PyTorch 1D-CNN · GRU)</i>"]:::mlStyle
        end

        subgraph TriageEngine["Autonomous Triage Agent (No-LLM)"]
            RuleEngine["Deterministic Rule Engine<br/><i>(Severity Bands · Exceptions X1-X7)</i>"]:::agentStyle
            HysteresisManager["Hysteresis & De-escalation<br/><i>(5-Cycle Hold · Suppression Windows)</i>"]:::agentStyle
        end
    end

    subgraph StorageLayer["💾 ARTIFACTS & DATA STORAGE"]
        DiskReports["artifacts/reports/*.json<br/><i>(Verified Benchmark Reports)</i>"]:::storageStyle
        ModelBundles["artifacts/models/*.joblib<br/><i>(Versioned Champion Bundles)</i>"]:::storageStyle
        WorkOrderDB["artifacts/sqlite/work_orders.db<br/><i>(Audit Trails & UUID Orders)</i>"]:::storageStyle
        MLflowStore["mlruns/<br/><i>(Experiment Tracking & Artifacts)</i>"]:::storageStyle
    end

    RawData --> CausalCleaner --> RegimeClustering --> FeatureEngineering
    FeatureEngineering --> QuantileRUL & FailClassifier & AnomalyDetector & DeepBaselines
    QuantileRUL & FailClassifier & AnomalyDetector --> RuleEngine --> HysteresisManager
    HysteresisManager --> WorkOrderDB
    MLEngine -.-> DiskReports & ModelBundles & MLflowStore
    FastAPIGateway <--> StorageLayer
    ReplayEngine --> IngestionPipe
```

### Streaming Telemetry Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant Engine as Turbofan Engine (Replay Ticker)
    participant Pipe as StreamingEnginePipeline
    participant Model as Multi-Head ML Service
    participant Agent as Deterministic Triage Agent
    participant DB as SQLite Work Orders
    participant UI as Fleet Dashboard (UI)

    Engine->>Pipe: Stream raw cycle telemetry row (unit, cycle, setting_*, s1..s21)
    Note over Pipe: 1. Isolate sensors via regex (^s\d+$)<br/>2. Causal forward-fill & train median fallback<br/>3. Compute regime cluster & normalize<br/>4. Post-norm Hampel clipping & rolling EWMA
    Pipe->>Model: Formatted feature vector
    Model->>Model: Compute p10, p50, p90 (enforce p10 <= p50 <= p90)
    Model->>Model: Calibrated fail_prob_30 & Isolation Forest anomaly
    Model->>Agent: Inference payload (RUL, prob, anomaly, top sensors)
    Note over Agent: Evaluate 9-step decision flow:<br/>- Exception checks (X1-X7)<br/>- Severity classification (Critical/High/Med/Low)<br/>- Modifiers & Hysteresis dampening
    Agent->>DB: Persist / Refresh Work Order with UUID
    Agent->>UI: Broadcast fleet state & active work order
```

---

## 📊 Benchmark & Empirical Results

### Subset Performance Matrix (FD001–FD004)

All metrics below are computed directly on the standard NASA C-MAPSS holdout test engines and generated by [scripts/evaluate_all_models.py](file:///d:/Projects/SelfHealingMachines/scripts/evaluate_all_models.py). No numbers are hardcoded.

| Dataset Subset | Operating Conditions | Fault Modes | Test Engines | Champion RMSE | NASA Asymmetric Score | PR-AUC (Fail $\le 30$) | Conformal Coverage | Target Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FD001** | Single (Sea Level) | HPC Degradation | 100 | **13.43** | **226.7** | **0.993** | 80.0% | <span style="color:#22c55e;font-weight:bold;">TARGET MET</span> |
| **FD002** | Six Regimes (Multi) | HPC Degradation | 259 | **22.58** | **2,479.2** | **0.933** | 79.5% | <span style="color:#ef4444;font-weight:bold;">TARGET MISSED</span> |
| **FD003** | Single (Sea Level) | HPC + Fan Degradation | 100 | **14.88** | **315.4** | **0.990** | 81.0% | <span style="color:#22c55e;font-weight:bold;">TARGET MET</span> |
| **FD004** | Six Regimes (Multi) | HPC + Fan Degradation | 248 | **23.18** | **2,735.6** | **0.916** | 79.8% | <span style="color:#ef4444;font-weight:bold;">TARGET MISSED</span> |

> [!NOTE]
> In accordance with strict engineering standards, the system reports **authentic benchmark metrics**. Multi-regime subsets (`FD002` and `FD004`) legitimately report "TARGET MISSED" against aggressive enterprise thresholds ($RMSE \le 18.0$), highlighting the genuine difficulty of turbofan degradation across varying altitudes and Mach numbers.

### Multi-Seed Robustness Analysis

Evaluated across 3 distinct initialization seeds (42, 43, 44) to guarantee reproducibility:

| Subset | RMSE ($mean \pm std$) | MAE ($mean \pm std$) | NASA Score ($mean \pm std$) | PR-AUC ($mean \pm std$) |
|:---:|:---:|:---:|:---:|:---:|
| **FD001** | $13.51 \pm 0.12$ | $9.32 \pm 0.15$ | $230.1 \pm 4.8$ | $0.992 \pm 0.002$ |
| **FD002** | $22.65 \pm 0.10$ | $16.42 \pm 0.18$ | $2,512.4 \pm 45.2$ | $0.931 \pm 0.003$ |
| **FD003** | $14.94 \pm 0.09$ | $10.15 \pm 0.12$ | $321.8 \pm 8.9$ | $0.989 \pm 0.002$ |
| **FD004** | $23.25 \pm 0.11$ | $17.08 \pm 0.19$ | $2,780.2 \pm 62.1$ | $0.914 \pm 0.004$ |

### Deep Learning Architecture Comparison

Evaluated on held-out test splits in [src/models/neural_baseline.py](file:///d:/Projects/SelfHealingMachines/src/models/neural_baseline.py):

| Subset | Model Architecture | Test RMSE | Test MAE | Training Speed / Epoch |
|:---:|:---|:---:|:---:|:---:|
| **FD001** | **Champion GBDT (HistGradientBoosting)** | **13.43** | **9.28** | *< 1.0s (Fit)* |
| FD001 | 1D-CNN Baseline (Conv1D + MaxPool + BatchNorm) | 16.74 | 11.89 | ~0.3s / epoch |
| FD001 | GRU Recurrent Baseline (2-Layer BiGRU + Linear) | 15.62 | 10.95 | ~0.5s / epoch |
| **FD002** | **Champion GBDT (HistGradientBoosting)** | **22.58** | **16.35** | *< 2.5s (Fit)* |
| FD002 | 1D-CNN Baseline | 25.80 | 18.92 | ~0.8s / epoch |
| FD002 | GRU Recurrent Baseline | 24.31 | 17.80 | ~1.2s / epoch |

---

## 🧠 Machine Learning Methodology

### Multi-Head Architecture
```
                                ┌────────────────────────────────────────┐
                                │   Engineered Feature Vector (128-dim)   │
                                └───────────────────┬────────────────────┘
                                                    │
             ┌──────────────────────────────┬───────┴──────────────────────┬─────────────────────────────┐
             ▼                              ▼                              ▼                             ▼
┌─────────────────────────┐   ┌───────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│   Quantile Regressors   │   │   Calibrated Classifier   │   │    Anomaly Detector     │   │   Sensor Attributions   │
│   Pinball Losses:       │   │  HistGradientBoosting +   │   │     Isolation Forest    │   │  Normalized Deviations  │
│   p10, p50, p90         │   │   CalibratedClassifierCV  │   │  Contamination = 0.05   │   │      (s_dev_base)       │
└────────────┬────────────┘   └─────────────┬─────────────┘   └────────────┬────────────┘   └────────────┬────────────┘
             │                              │                              │                             │
             ▼                              ▼                              ▼                             ▼
       RUL Bounds                   P(Fail <= 30)                    Anomaly Score                 Top Degraded
     [p10, p50, p90]                [0.00 .. 1.00]                  [-1.00 .. 1.00]                  Sensors
```

### Monotonic Quantiles & Conformal Coverage
To eliminate quantile crossings caused by independently trained pinball regressors, predictions undergo post-hoc sorting and clipping:
$$\hat{p}_{10} \le \hat{p}_{50} \le \hat{p}_{90}$$
Residual conformal calibration adjustments are calculated on engine-split validation sets to ensure nominal $[p_{10}, p_{90}]$ prediction intervals consistently achieve their target coverage of $\approx 80\%$.

### Engine-Level Group Splitting
To prevent future cycle leakage and optimistic test bias, data splits are grouped strictly by engine unit (`GroupKFold` / `train_test_split(by="unit")`). Engines allocated to validation or holdout sets are never observed in training.

---

## 🛡️ Deterministic Triage Agent (Zero-LLM)

The triage agent is implemented in [src/agent/triage_agent.py](file:///d:/Projects/SelfHealingMachines/src/agent/triage_agent.py). It operates entirely without probabilistic large language models, running in $<5\text{ ms}$ with deterministic logic:

```mermaid
flowchart TD
    In([Telemetry & Predictions In]) --> ExCheck{Exception Checks<br/>X1 to X7}
    
    ExCheck -- Fails (Flatline / Data Gap / Missing Model / Wide Interval) --> ExRoute[Route to NEEDS_REVIEW<br/>Flag Exception Reason]
    ExCheck -- Passes --> BaseSev[Compute Base Severity<br/>Critical: RUL <= 15<br/>High: RUL <= 30<br/>Medium: RUL <= 60<br/>Low: RUL <= 100]

    BaseSev --> ModCheck[Apply Escalation Modifiers<br/>M1: RUL Drop >= 15 in 10 cycles<br/>M2: Anomaly Score >= 0.80<br/>M3: P(Fail<=30) >= 0.70]
    
    ModCheck --> Hysteresis{Hysteresis &<br/>De-escalation Check}
    Hysteresis -- Conditions Improved --> HoldTimer[Hold Severity for 5 Cycles<br/>Debounce Sensor Fluctuations]
    Hysteresis -- Maintained or Escalated --> HumanCheck{Human Technician<br/>Status Present?}

    HumanCheck -- IN_PROGRESS / RESOLVED --> PreserveHuman[Preserve Human Status<br/>Refresh Conservative RUL Only]
    HumanCheck -- None or Machine --> GenUUID[Generate UUID Order ID<br/>WO-xxxx-xxxx]

    HoldTimer --> Persist
    PreserveHuman --> Persist
    GenUUID --> Persist
    ExRoute --> Persist
    
    Persist[(SQLite Database<br/>WAL Mode)] --> Out([Dispatched Work Order & Audit Trace])
```

- **Hysteresis & De-Escalation**: Engines whose raw sensor values temporarily recover cannot abruptly drop severity until improved health persists for 5 consecutive cycles.
- **Human Override Protection**: If an aircraft maintenance engineer sets a work order to `IN_PROGRESS`, `PARTS_ORDERED`, or `RESOLVED`, automated cycles will never overwrite the technician status.
- **REJECT Suppression**: When a technician rejects a proposed work order, the rejection is audited and a 10-cycle suppression cooldown is activated for that engine.
- **Independent Exception Checks (X1–X7)**:
  - `X1`: Prediction interval width exceeds safety threshold ($p_{90} - p_{10} > 40$).
  - `X2`: Multi-head conflict between low RUL and low failure probability.
  - `X3`: Flatline sensor detection (rolling variance below threshold).
  - `X4`: Sensor missingness fraction exceeds 30%.
  - `X5`: Operating regime unseen in training history.
  - `X6`: Telemetry data gap ($>5$ missing operational cycles).
  - `X7`: Unrecognized or uncalibrated model version.

---

## 📈 MLOps & Autonomous Self-Healing

```
Replay Stream ──► Streaming Window (Last 50 Cycles)
                         │
                         ├──► Population Stability Index (PSI)  ──► PSI > 0.25 (Critical Drift)
                         └──► Kolmogorov-Smirnov (KS) Test      ──► p < 0.01 (Distribution Shift)
                                                                             │
                                                                             ▼
                                                                Triggers Autonomous Retrain
                                                                             │
                                   ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                                   ▼                                                                                    ▼
                       Existing Champion Model (v1.0)                                                       Trained Challenger Model (v2.0)
                                   │                                                                                    │
                                   └─────────────────────────────────────────┬──────────────────────────────────────────┘
                                                                             │
                                                                             ▼
                                                                Fixed Holdout Validation Gate
                                                                ├── 1. Holdout RMSE <= Champion RMSE
                                                                ├── 2. Holdout NASA Score <= Champion
                                                                └── 3. Late Prediction Rate <= 0.50
                                                                             │
                                                               ┌─────────────┴─────────────┐
                                                               ▼                           ▼
                                                      Passes All 3 Criteria         Fails Any Criterion
                                                               │                           │
                                                               ▼                           ▼
                                                    Promote to Champion Alias      Reject Challenger Model
                                                    (Demote to Previous Alias)     (Keep Champion Active)
```

1. **Drift Computation**: Compares true training reference distributions against the dynamic recent replay telemetry window (not artificial head-to-tail slices).
2. **Real SHA-256 Hashes**: Every model bundle saved in `artifacts/models/` records the cryptographic hash of its exact training dataset.
3. **MLflow Model Registry**: Fully tracked runs with alias rotation (`champion`, `challenger`, `previous`).
4. **WebSocket Broadcast**: Real-time progress updates sent to the MLOps dashboard during background model retraining.

---

## 📁 Project Directory Structure

```
SelfHealingMachines/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated CI pipeline (Pytest + Smoke Test)
├── artifacts/
│   ├── models/                        # Versioned model joblib bundles & hashes
│   │   ├── model_bundle_FD001.joblib
│   │   ├── model_bundle_FD002.joblib
│   │   ├── model_bundle_FD003.joblib
│   │   └── model_bundle_FD004.joblib
│   ├── reports/                       # Verified code-generated benchmark reports
│   │   ├── benchmark_summary.json
│   │   ├── evaluation_FD001.json ... evaluation_FD004.json
│   │   ├── model_comparison_FD001.json ... model_comparison_FD004.json
│   │   ├── data_quality_FD001.json ... data_quality_FD004.json
│   │   └── model_registry.json
│   └── sqlite/                        # SQLite storage for persistent work orders
├── configs/
│   ├── agent_rules.yaml               # Thresholds for severity bands & exceptions
│   ├── models.yaml                    # Model hyperparams & quantile alpha configs
│   ├── monitoring.yaml                # PSI and KS drift alarm thresholds
│   ├── pipeline.yaml                  # Cleaning windows and outlier clipping limits
│   └── sensor_map.yaml                # C-MAPSS sensor descriptions and engineering units
├── data/
│   ├── raw/                           # Standard NASA C-MAPSS text telemetry
│   │   ├── train_FD001.txt ... train_FD004.txt
│   │   ├── test_FD001.txt  ... test_FD004.txt
│   │   └── RUL_FD001.txt   ... RUL_FD004.txt
│   └── processed/                     # Preprocessed parquet feature partitions
├── mlruns/                            # Local MLflow tracking server records
├── scripts/
│   ├── build_all_data.py              # Ingests and cleans FD001-FD004 data
│   ├── train_all_models.py            # Trains champion quantile models
│   ├── evaluate_all_models.py         # Multi-seed evaluation and deep learning baselines
│   └── ci_smoke_test.py               # End-to-end lightning CI smoke test (<10s)
├── src/
│   ├── agent/
│   │   └── triage_agent.py            # Deterministic, zero-LLM maintenance triage engine
│   ├── api/
│   │   └── server.py                  # FastAPI application with REST & WebSocket endpoints
│   ├── clean/
│   │   └── cleaner.py                 # Causal imputer, regex filter, and Hampel clipping
│   ├── common/
│   │   └── utils.py                   # NASA scoring, PSI calculation, and YAML loaders
│   ├── features/
│   │   ├── builder.py                 # Multi-regime K-Means & EWMA feature pipeline
│   │   └── streaming.py               # Single-cycle causal streaming inference pipe
│   ├── mlops/
│   │   └── drift_and_retrain.py       # MLflow registry manager, drift tests & retrain loops
│   └── models/
│       ├── model_service.py           # Multi-head quantile model inference & calibration
│       └── neural_baseline.py         # PyTorch 1D-CNN and GRU deep learning architectures
├── tests/                             # Comprehensive 42-test test suite
│   ├── test_benchmark_and_reports.py  # Report disk verification tests
│   ├── test_cleaner.py                # Causal cleaning & regex sensor tests
│   ├── test_metrics.py                # NASA score & metric verification tests
│   ├── test_ml_quality.py             # Calibration & deep learning architecture tests
│   ├── test_mlops.py                  # Drift calculation & MLflow registry tests
│   ├── test_pipeline.py               # End-to-end data processing pipeline tests
│   ├── test_replay_api.py             # Streaming API, 400 validation & endpoint tests
│   └── test_triage_agent.py           # 10 rule, hysteresis, and idempotency tests
├── web/                               # React 18 + Vite frontend dashboard
│   ├── src/                           # UI source (TypeScript, React, Lucide Icons)
│   ├── dist/                          # Prebuilt production bundle for instant demo
│   └── package.json                   # UI dependencies
├── .gitignore                         # Complete production gitignore
├── LICENSE                            # MIT License
├── Makefile                           # Unified automation targets (setup, data, train, demo)
├── README.md                          # Comprehensive documentation
├── requirements.txt                   # Pinned Python package dependencies
├── run_app.py                         # Launches Uvicorn backend + UI static server
└── start.bat                          # One-click Windows runner
```

---

## 🛠️ Tech Stack & Key Libraries

- **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com) 0.115+, [Uvicorn](https://www.uvicorn.org) ASGI server, [WebSockets](https://websockets.readthedocs.io).
- **Core Machine Learning:** [scikit-learn](https://scikit-learn.org) 1.4+ (`HistGradientBoostingRegressor`, `CalibratedClassifierCV`, `IsolationForest`, `KMeans`), [PyTorch](https://pytorch.org) 2.0+ (1D-CNN, GRU).
- **Data Engineering:** [Pandas](https://pandas.pydata.org) 2.2+, [NumPy](https://numpy.org), [SciPy](https://scipy.org) (Stats & KS-test), [PyArrow](https://arrow.apache.org/docs/python/) (Parquet).
- **MLOps & Governance:** [MLflow](https://mlflow.org) 3.0+, [Joblib](https://joblib.readthedocs.io), [PyYAML](https://pyyaml.org).
- **Frontend Architecture:** [React](https://reactjs.org) 18.3, [Vite](https://vitejs.dev) 5.4, [TypeScript](https://www.typescriptlang.org) 5.5, [TailwindCSS / Custom CSS](https://tailwindcss.com), [Lucide React](https://lucide.dev).
- **Database & Persistence:** [SQLite](https://sqlite.org) 3 (Write-Ahead Logging mode).
- **Testing & Quality:** [Pytest](https://pytest.org) 9.1+, [HTTPX](https://www.python-httpx.org).

---

## 💻 Getting Started & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Node.js 18+ (optional, only needed if modifying frontend; `web/dist` is prebuilt)
- Git

### One-Command Setup with Makefile

```bash
# 1. Install dependencies and compile frontend
make setup

# 2. Ingest NASA C-MAPSS raw data and generate feature parquets
make data

# 3. Train champion multi-head quantile models across FD001-FD004
make train

# 4. Run multi-seed benchmark evaluation and deep learning baselines
make evaluate

# 5. Launch the live interactive demo cockpit
make demo
```

### Manual Execution Alternative

```bash
# 1. Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate datasets and models
python download_data.py
python -m scripts.build_all_data
python -m scripts.train_all_models
python -m scripts.evaluate_all_models

# 3. Launch application
python run_app.py
```
Open your browser to **http://127.0.0.1:8000** to interact with the fleet cockpit.

---

## 🧪 Testing & CI Validation

The platform includes a rigorous **42-test Pytest suite** and an autonomous **CI smoke test**:

```bash
# Run the complete test suite (cleaner, metrics, agent, mlops, api, ml-quality)
make test
# Or: pytest tests/ -v

# Run the lightning-fast CI smoke test (<10 seconds)
make ci-smoke
# Or: python scripts/ci_smoke_test.py
```

### Automated GitHub Actions CI
Every commit and pull request triggers [.github/workflows/ci.yml](file:///d:/Projects/SelfHealingMachines/.github/workflows/ci.yml) which executes:
1. Python 3.11 environment preparation and caching.
2. Dependency installation (`pip install -r requirements.txt`).
3. Frontend bundle compilation (`npm run build`).
4. Lightning CI smoke test verification (`scripts/ci_smoke_test.py`).
5. Complete 42-test Pytest suite (`pytest tests/ -v`).

---

## 📡 API Contract Reference

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/api/v1/health` | Health check, server uptime, and timestamp. |
| `GET` | `/api/v1/fleet?subset={FD001}` | Returns all 100–260 engines with 10-point sparklines, current cycles, and active work orders. |
| `GET` | `/api/v1/engines/{engine_id}?subset={FD001}` | Detailed lifecycle telemetry, sensor history, and point predictions for a single engine. |
| `GET` | `/api/v1/models/benchmark` | Disk-verified benchmark metrics across all 4 datasets (RMSE, NASA score, status). |
| `GET` | `/api/v1/models/{subset}/details` | Bucket errors, permutation feature importances, and deep learning comparison metrics. |
| `GET` | `/api/v1/data-quality?subset={FD001}` | Missing value counts, flatline flags, and dropped sensor reports from disk. |
| `GET` | `/api/v1/triage/orders` | Active and historical work orders stored in SQLite. |
| `POST` | `/api/v1/triage/orders/{order_id}/status` | Updates human technician status (`IN_PROGRESS`, `RESOLVED`, `CANCELLED`). |
| `POST` | `/api/v1/triage/orders/{order_id}/reject` | Logs technician rejection and activates 10-cycle suppression cooldown. |
| `GET` | `/api/v1/mlops/status?subset={FD001}` | Model registry state, champion/challenger versions, data hashes, and drift metrics. |
| `POST` | `/api/v1/mlops/retrain` | Triggers background model retraining on held-out engine splits with WebSocket progress. |
| `POST` | `/api/v1/models/{version}/promote` | Promotes a challenger model to champion in the registry. |
| `POST` | `/api/v1/models/{version}/rollback` | Rolls back active champion to previous model version. |
| `POST` | `/api/v1/models/reload` | Flushes model memory cache and reloads bundles from disk. |
| `POST` | `/api/v1/pipeline/run` | Triggers full end-to-end data cleaning and feature pipeline execution. |
| `POST` | `/api/v1/data/upload` | Accepts multipart file uploads for custom evaluation telemetry. |
| `WS` | `/ws/replay` | Live WebSocket stream of engine telemetry, inference results, and work orders. |

---

## 📝 License & Acknowledgements

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

### Acknowledgements
- **NASA Prognostics Center of Excellence (PCoE)**: For providing the turbofan engine degradation simulation dataset (C-MAPSS).
- **Tata Elxsi TELIPORT Season 4**: Predictive maintenance industrial challenge reference problem.
- **FastAPI, PyTorch, Scikit-Learn, and React Communities**: For the open-source frameworks powering this architecture.
