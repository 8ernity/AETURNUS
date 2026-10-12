# ==============================================================================
# AETURNUS - Autonomous Equipment Tracking and Uptime Restoration
# Multi-stage Production Container
# ==============================================================================

# Stage 1: Build Frontend Assets
FROM node:22-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci || npm install
COPY web/ ./
RUN npm run build

# Stage 2: Python 3.11 Production Runtime
FROM python:3.11-slim AS runtime

# System environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install lightweight OS packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend, configs, data loaders, reports, scripts, and initial data
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY download_data.py run_app.py Procfile LICENSE README.md /app/

# Build all four subsets at image build so it starts immediately without runtime delay
RUN python -m scripts.build_all_data

# Copy compiled frontend from Stage 1 into web/dist
COPY --from=frontend-builder /app/web/dist /app/web/dist

# Create runtime directories for SQLite and uploads
RUN mkdir -p /app/artifacts/sqlite /app/data/uploads

# Expose target application port
EXPOSE 8000

# Container healthcheck (readiness-based /api/v1/health with 180s start period)
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/api/v1/health || exit 1

# Launch application
CMD ["python", "run_app.py"]
