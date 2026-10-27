# Deployment & Production Runbook

This runbook documents single-container, low-footprint deployment options for the AETURNUS / SelfHealingMachines predictive maintenance platform.

---

## 1. Local Development Execution

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend development)

### Backend Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Verify tests and build reports
pytest
python scripts/generate_readme_tables.py

# 4. Start FastAPI server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd web
npm install
npm run dev      # Local Vite dev server on http://localhost:5173
npm run build    # Generates production bundle into web/dist
```

---

## 2. Single-Container Docker Deployment

The application includes a production multi-stage `Dockerfile` that builds the React frontend with Vite and serves both API and static assets from FastAPI on port 8000.

### Build and Run
```bash
# Build production Docker image
docker build -t aeturnus-pdm:latest .

# Run container with 512MB RAM constraint (matches Render/free tier limits)
docker run -p 8000:8000 \
  --memory=512m \
  --cpus=1.0 \
  -e OMP_NUM_THREADS=1 \
  -e MALLOC_ARENA_MAX=2 \
  aeturnus-pdm:latest
```

---

## 3. Free-Tier Cloud Deployment (Render / Railway / GCP Cloud Run)

To run reliably within free-tier resource boundaries (e.g., 512 MB RAM limit):

1. **Thread Pool Constraints**:
   FastAPI automatically sets:
   ```python
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["OPENBLAS_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   os.environ["MALLOC_ARENA_MAX"] = "2"
   ```
   This prevents glibc and OpenMP from allocating unbound per-thread arenas that lead to OOM on 512MB instances.

2. **Lazy Loading of Models & Datasets**:
   - Telemetry datasets and LightGBM model bundles are cached per subset upon first access.
   - Parquet projection loading is used (`pyarrow` schema inspection) so only drift-relevant columns are loaded during monitoring runs, reducing RAM usage by 80%.

3. **Render Setup**:
   - Environment: Docker
   - Instance Type: Free (512 MB RAM, 0.1 CPU)
   - Start Command: `python run_app.py`
   - Health Check Path: `/api/v1/health`
   - Port: Render-injected `$PORT` (local default `8000`)

4. **Fast Port Binding**:
   - FastAPI does not load datasets or models during lifespan startup. It
     returns from startup immediately and warms all configured subsets
     sequentially in a background thread, so Render's `$PORT` opens without
     waiting for model/data work.
   - Each subset's telemetry frames are evicted after its fleet prediction cache
     is primed, keeping peak memory bounded on 512 MB instances.

---

## 4. Observability & Health Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `GET /api/v1/health` | HTTP JSON | Uptime, active subset, and memory health check |
| `GET /metrics` | Prometheus | Standard Prometheus scrape target with `pdm_` metrics |
| `GET /api/v1/monitoring/alerts` | HTTP JSON | Operational health alerts (drift, backlog, coverage, latency) |
