# Production Deployment Guide — Aero Piston Engine Digital Twin

## 1. Overview
This document outlines the production packaging, containerization, environment configuration, database migrations, CI pipeline, and deployment procedures for the Aero Piston Engine Digital Twin backend (Modules 0–24).

The system is delivered as a reproducible, offline-capable Python package and containerized microservice running FastAPI over Uvicorn.

---

## 2. Prerequisites
- **Python**: `^3.11`
- **Package Manager**: `uv` or `pip`
- **Container Engine**: Docker 24.0+
- **Orchestration**: Docker Compose 2.20+

---

## 3. Environment Configuration (`.env`)

Configure operational settings using environment variables (see `.env.example`):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Environment mode (`development`, `staging`, `production`). |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `CONFIG_PATH` | `config/default.yaml` | Path to core engine configuration YAML. |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/digital_twin.db` | Async database connection URL. |
| `API_HOST` | `0.0.0.0` | Bind host address. |
| `API_PORT` | `8000` | Bind HTTP port. |
| `DEPLOYMENT_ROLE` | `GROUND_STATION` | Deployment partition role (`EDGE` vs `GROUND_STATION`). |
| `TELEMETRY_SOURCE` | `simulator` | Telemetry input source (`simulator`, `csv_replay`, `live`). |
| `MODEL_DIR` | `models/` | Directory path for ML model `.pkl` / `.onnx` artifacts. |

> **Security Note**: Never commit actual passwords, API keys, or private HMAC secret keys to version control. Set `HMAC_SECRET_KEY` in environment variables or container secret mounts.

---

## 4. Container Deployment (Docker & Compose)

### 4.1 Building Production Image
```bash
docker build -t piston-engine-digital-twin:latest .
```

### 4.2 Running via Docker Compose
```bash
docker-compose up -d
```

### 4.3 Container Health Verification
```bash
docker exec piston-engine-twin python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/system/health'); print(r.json())"
```

---

## 5. Database Migrations (Alembic)
Apply database schema migrations safely before application launch:

```bash
# Run online migrations
uv run alembic upgrade head
```

---

## 6. ML Model Artifact Handling
- The backend checks `models/` (or `MODEL_DIR`) for trained ML model artifacts.
- If model artifacts are missing or unreadable, the system defaults cleanly to **`MODEL_UNAVAILABLE`** status and executes deterministic heuristic physics rules without process failure.

---

## 7. Security Hardening Measures
1. **Non-Root Execution**: Container process executes as user `appuser` (UID 1000).
2. **Minimal Base Image**: Built on `python:3.11-slim` with multi-stage build eliminating build-time dependencies.
3. **Passive Advisory Guard**: API and Advisory engine contain zero control or actuator manipulation capabilities.
4. **Ground Truth Isolation**: Simulator ground truth (`SimulationGroundTruth`) is strictly inaccessible to L2/L3 pipeline inference.

---

## 8. Continuous Integration (CI/CD)
Automated GitHub Actions workflow (`.github/workflows/ci.yml`) executes on every push/PR:
1. Static analysis (`ruff check src/ tests/`)
2. Full pytest suite (Modules 0–24 unit, integration, scientific acceptance)
3. Python package wheel build
4. Production Docker build & container health smoke check
