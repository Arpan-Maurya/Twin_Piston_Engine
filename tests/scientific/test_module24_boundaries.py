"""
Scientific Acceptance Suite — Modules 19–21: Advisory, API Contracts & Edge/Ground Boundary.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.core.config import get_settings
from src.edge_ground import (
    EdgeStoreAndForwardBuffer,
    EdgeTelemetryEnvelope,
    GroundTelemetryGateway,
    InMemoryTelemetryTransport,
)
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l3_ml import AdvisoryEngine


def test_advisory_engine_zero_control_actuator_commands():
    settings = get_settings()
    engine = AdvisoryEngine(settings=settings)

    # Inspect all attributes and methods of AdvisoryEngine to ensure NO actuation methods exist
    advisory_methods = [m for m in dir(engine) if not m.startswith("_")]
    forbidden_terms = ["actuate", "control", "set_throttle", "set_rpm", "command_engine", "override_ecu"]

    for method in advisory_methods:
        for term in forbidden_terms:
            assert term not in method.lower()


def test_api_v1_scientific_schemas_and_no_actuation_endpoints():
    app = create_app()
    client = TestClient(app)

    # Verify standard API endpoint responds with valid health schema
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

    # Search routes for forbidden control/actuation endpoints
    for route in app.routes:
        path = getattr(route, "path", "")
        assert "actuate" not in path
        assert "control" not in path
        assert "set_throttle" not in path


def test_edge_ground_partition_and_store_forward():
    buf = EdgeStoreAndForwardBuffer(max_capacity=10)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    for rec in records:
        env = EdgeTelemetryEnvelope.from_record(rec)
        buf.enqueue(env)

    assert len(buf) == 5

    gw = GroundTelemetryGateway()
    flushed = buf.dequeue_batch(10)
    assert len(flushed) == 5
    assert len(buf) == 0
