"""
Unit tests for Original Module 20: REST API & Real-Time Backend Interfaces DTOs and App Configuration.
"""

import pytest
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.v1.schemas import (
    APIErrorResponse,
    HealthCheckResponse,
    SystemStatusResponse,
    TelemetryIngestRequest,
    EngineHealthResponse,
    DiagnosticSummaryResponse,
    AnomalyResponse,
    FaultClassificationResponse,
    RULResponse,
    MissionRiskResponse,
    AdvisoryResponse,
    ExplanationResponse,
    DiagnosticQueryRequest,
    DiagnosticQueryResponse,
    ReplayActionRequest,
    ReplayStateResponse,
    WhatIfRequest,
    WhatIfResponse,
    ScenarioCompareRequest,
    ScenarioCompareResponse,
    EventEnvelope
)
from src.core.config import AppSettings
from src.core.exceptions import (
    PistonEngineError,
    ResourceNotFoundException,
    ValidationException,
    DigitalTwinError,
)

class TestModule20Schemas:
    """Test suite for Module 20 Schema models and serialization."""

    def test_api_error_response_schema(self):
        err = APIErrorResponse(
            error_code="RESOURCE_NOT_FOUND",
            message="Item not found",
            timestamp=datetime.now(timezone.utc),
            details={"path": "/api/v1/test"}
        )
        data = err.model_dump()
        assert data["error_code"] == "RESOURCE_NOT_FOUND"

    def test_health_check_response_schema(self):
        res = HealthCheckResponse(
            status="ok",
            service_name="Piston Engine Digital Twin API",
            version="1.0.0"
        )
        assert res.status == "ok"
        assert res.version == "1.0.0"

    def test_telemetry_ingest_request_schema(self):
        req = TelemetryIngestRequest(
            sequence_number=1,
            egt_cyl1_hot_uv=12000.0,
            egt_cyl2_hot_uv=12100.0,
            egt_cyl3_hot_uv=11950.0,
            egt_cyl4_hot_uv=12050.0,
            egt_cold_c=25.0,
            cht_hot_uv=8000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=110.0,
            oil_p_counts=2048,
            map_counts=2048,
            adc_vref_counts=4095,
            crank_period_us=12000.0,
            fuel_pulse_hz=50.0,
            accel_counts_xyz=(2048, 2048, 2048),
            ambient_temp_c=25.0,
            ambient_press_pa=101325.0
        )
        assert req.sequence_number == 1
        assert req.egt_cyl1_hot_uv == 12000.0

    def test_event_envelope(self):
        envelope = EventEnvelope(
            event_type="telemetry.updated",
            payload={"rpm": 5200.0}
        )
        assert envelope.event_type == "telemetry.updated"
        assert envelope.payload["rpm"] == 5200.0


class TestModule20AppFactory:
    """Test suite for FastAPI app instantiation, CORS, and exception handlers."""

    def test_create_app_returns_fastapi_instance(self):
        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "Aero Piston Engine Digital Twin REST & Real-Time API"
        assert app.version == "1.0.0"

    def test_openapi_schema_generation(self):
        app = create_app()
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/system/health" in schema["paths"]
        assert "/api/v1/telemetry" in schema["paths"]
        assert "/api/v1/engine/health" in schema["paths"]
        assert "/api/v1/scenarios/what-if" in schema["paths"]

    def test_cors_middleware_configured(self):
        custom_settings = AppSettings(api_cors_origins=["http://localhost:3000", "https://ground-station.org"])
        app = create_app(settings=custom_settings)
        client = TestClient(app)
        response = client.options(
            "/api/v1/system/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_exception_handler_custom_not_found(self):
        app = create_app()

        @app.get("/test-not-found")
        def raise_not_found():
            raise ResourceNotFoundException("Resource XYZ missing", resource_id="XYZ")

        client = TestClient(app)
        response = client.get("/test-not-found")
        assert response.status_code == 404
        json_data = response.json()
        assert json_data["error_code"] == "ResourceNotFoundException"
        assert json_data["message"] == "Resource XYZ missing"

    def test_exception_handler_custom_validation(self):
        app = create_app()

        @app.get("/test-validation-error")
        def raise_val_err():
            raise ValidationException("RPM parameter out of range")

        client = TestClient(app)
        response = client.get("/test-validation-error")
        assert response.status_code == 422
        json_data = response.json()
        assert json_data["error_code"] == "ValidationException"
        assert json_data["message"] == "RPM parameter out of range"

    def test_exception_handler_custom_digital_twin_error(self):
        app = create_app()

        @app.get("/test-twin-error")
        def raise_twin_err():
            raise DigitalTwinError("Internal state divergence")

        client = TestClient(app)
        response = client.get("/test-twin-error")
        assert response.status_code == 500
        json_data = response.json()
        assert json_data["error_code"] == "DigitalTwinError"
