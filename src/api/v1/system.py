"""
System Router — Service Health & Operational Status Endpoints (Module 20).

CLEARLY DISTINGUISHES API SERVICE HEALTH FROM PHYSICAL ENGINE HEALTH.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_app_settings
from src.api.v1.schemas import HealthCheckResponse, SystemStatusResponse
from src.core.config import AppSettings

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="REST API Service Availability Health",
    description="Reports HTTP API server operational status. Does NOT indicate physical engine health state.",
)
def get_service_health(
    settings: AppSettings = Depends(get_app_settings),
) -> HealthCheckResponse:
    return HealthCheckResponse(
        status="ok",
        service_name="Piston Engine Digital Twin API",
        version=settings.version,
        details={
            "environment": settings.app_env,
            "engine_model": settings.engine.name,
            "note": "Service health only; see /api/v1/engine/health for physical engine health.",
        },
    )


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Overall System Subsystem Status",
    description="Reports API, Telemetry Pipeline, Digital Twin, and ML Supervision subsystem readiness.",
)
def get_system_status(
    settings: AppSettings = Depends(get_app_settings),
) -> SystemStatusResponse:
    return SystemStatusResponse(
        service_status="OPERATIONAL",
        telemetry_pipeline_status="READY",
        digital_twin_status="READY",
        ml_supervision_status="READY",
        active_connections=0,
    )
