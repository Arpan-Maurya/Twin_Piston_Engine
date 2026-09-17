"""
API v1 Master Router Package (Module 20).

Aggregates all versioned v1 REST and WebSocket sub-routers:
- System (/system)
- Telemetry (/telemetry)
- Engine Health (/engine)
- Diagnostics (/diagnostics)
- Faults & Anomalies (/diagnostics)
- Remaining Useful Life (/engine)
- Mission Risk (/mission)
- Advisories & Q&A (/advisories, /explanations, /diagnostic)
- History (/history)
- Scenario Replay (/replay)
- What-If Scenarios (/scenarios)
- Real-Time WebSocket (/ws)
"""

from fastapi import APIRouter

from src.api.v1.advisories import router as advisories_router
from src.api.v1.diagnostics import router as diagnostics_router
from src.api.v1.engine_health import router as engine_health_router
from src.api.v1.fault_anomaly import router as fault_anomaly_router
from src.api.v1.history import router as history_router
from src.api.v1.mission_risk import router as mission_risk_router
from src.api.v1.replay import router as replay_router
from src.api.v1.rul import router as rul_router
from src.api.v1.system import router as system_router
from src.api.v1.telemetry import router as telemetry_router
from src.api.v1.websocket import router as websocket_router
from src.api.v1.what_if import router as what_if_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(system_router)
api_v1_router.include_router(telemetry_router)
api_v1_router.include_router(engine_health_router)
api_v1_router.include_router(diagnostics_router)
api_v1_router.include_router(fault_anomaly_router)
api_v1_router.include_router(rul_router)
api_v1_router.include_router(mission_risk_router)
api_v1_router.include_router(advisories_router)
api_v1_router.include_router(history_router)
api_v1_router.include_router(replay_router)
api_v1_router.include_router(what_if_router)
api_v1_router.include_router(websocket_router)
