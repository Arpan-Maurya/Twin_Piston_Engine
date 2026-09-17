"""
Integration tests for Module 23 — Resilience & Failure Isolation.
"""

import pytest
from src.performance_resilience.resilience import safe_subsystem_call, retry_with_backoff


def test_subsystem_failure_isolation_ml():
    def failing_ml_model_load():
        raise FileNotFoundError("Model file missing: model_v1.onnx")

    fallback_state = {"status": "UNAVAILABLE", "error": "Model offline"}
    result = safe_subsystem_call(
        subsystem_name="ml_model_loader",
        func=failing_ml_model_load,
        fallback=fallback_state,
    )
    assert result == fallback_state
    assert result["status"] == "UNAVAILABLE"


def test_subsystem_failure_isolation_digital_twin():
    def failing_twin_calc():
        raise ZeroDivisionError("Engine RPM is zero in divisor")

    fallback_state = {"status": "DEGRADED", "engine_rpm": 0.0, "health_index": 0.0}
    result = safe_subsystem_call(
        subsystem_name="digital_twin_physics",
        func=failing_twin_calc,
        fallback=fallback_state,
    )
    assert result == fallback_state
    assert result["status"] == "DEGRADED"


def test_retry_transient_network_failure():
    attempts = 0

    @retry_with_backoff(max_retries=3, initial_delay_s=0.01)
    def fetch_ground_station_telemetry():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionResetError("Ground station link reset")
        return {"telemetry_packet": "VALID_RECORD"}

    data = fetch_ground_station_telemetry()
    assert data["telemetry_packet"] == "VALID_RECORD"
    assert attempts == 3
