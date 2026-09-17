"""
Boundary and Security Verification Tests for Original Module 20.

Enforces Module 20 mandatory design and security gates:
1. No simulator ground-truth leakage in production endpoints.
2. No actuator or UAV engine control endpoints present in API router.
3. No arbitrary code execution or filesystem access routes.
4. No physics or ML formulas calculated directly inside API route functions.
"""

import pytest
import inspect
import re
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.v1 import (
    advisories,
    diagnostics,
    engine_health,
    fault_anomaly,
    history,
    mission_risk,
    replay,
    rul,
    system,
    telemetry,
    websocket,
    what_if,
)
from src.core.exceptions import ActuationAttemptError


class TestModule20BoundarySafety:
    """Boundary and security verification suite."""

    def test_no_actuator_control_endpoints_in_routes(self):
        app = create_app()
        prohibited_terms = ["actuate", "throttle_control", "set_rpm", "set_fuel_rate", "override_ignition", "kill_engine", "servo"]
        
        for route in app.routes:
            path = getattr(route, "path", "").lower()
            name = getattr(route, "name", "").lower()
            for term in prohibited_terms:
                assert term not in path, f"Prohibited actuation/control term '{term}' found in path: {path}"
                assert term not in name, f"Prohibited actuation/control term '{term}' found in route name: {name}"

    def test_no_physics_or_ml_formulas_in_route_modules(self):
        """Verify route functions delegate calculations to lower-level domain modules."""
        modules_to_check = [
            advisories,
            diagnostics,
            engine_health,
            fault_anomaly,
            history,
            mission_risk,
            replay,
            rul,
            system,
            telemetry,
            websocket,
            what_if,
        ]

        forbidden_math_patterns = [
            r"math\.sqrt\(",
            r"math\.exp\(",
            r"math\.log\(",
            r"\.predict\(",
            r"\.fit\(",
            r"wiebe",
            r"stefan_boltzmann",
        ]

        for mod in modules_to_check:
            source = inspect.getsource(mod)
            for pattern in forbidden_math_patterns:
                match = re.search(pattern, source, re.IGNORECASE)
                assert match is None, f"Forbidden calculation or ML model invocation '{pattern}' found in API module {mod.__name__}"

    def test_no_ground_truth_leakage_in_api_schemas(self):
        """Verify API DTOs do not contain ground-truth state or internal simulator parameters."""
        app = create_app()
        client = TestClient(app)

        res_health = client.get("/api/v1/engine/health").json()
        assert "ground_truth" not in res_health
        assert "wiebe" not in str(res_health).lower()

        res_diag = client.get("/api/v1/diagnostics").json()
        assert "ground_truth" not in res_diag
        assert "wiebe" not in str(res_diag).lower()

    def test_no_arbitrary_code_or_filesystem_routes(self):
        app = create_app()
        for route in app.routes:
            path = getattr(route, "path", "").lower()
            assert "eval" not in path
            assert "exec" not in path
            assert "shell" not in path
            assert "file_system" not in path
            assert "read_file" not in path
            assert "write_file" not in path
