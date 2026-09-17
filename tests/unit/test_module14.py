"""
Unit tests for Original Module 14: Health Index and Degradation Supervision.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import get_settings
from src.core.provenance import (
    DegradationState,
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    CombustionStabilityState,
    DiagnosticState,
    FaultClassificationResult,
    HealthState,
    LubricationState,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.health_supervision import HealthSupervisionEngine


@pytest.fixture
def engine() -> HealthSupervisionEngine:
    eng = HealthSupervisionEngine()
    eng.reset_state()
    return eng


@pytest.fixture
def healthy_residual_state() -> ResidualState:
    op = OperatingPoint(
        rpm=4000.0,
        map_pressure_pa=120000.0,
        altitude_m=1000.0,
        ambient_temp_k=288.15,
        ambient_pressure_pa=101325.0,
        throttle_pct=50.0,
    )
    res_dict = {
        "egt_cyl1": make_tagged(0.0, Provenance.DERIVED),
        "egt_cyl2": make_tagged(0.0, Provenance.DERIVED),
        "egt_cyl3": make_tagged(0.0, Provenance.DERIVED),
        "egt_cyl4": make_tagged(0.0, Provenance.DERIVED),
        "oil_pressure": make_tagged(0.0, Provenance.DERIVED),
        "oil_temp": make_tagged(0.0, Provenance.DERIVED),
        "vibration_rms": make_tagged(0.0, Provenance.DERIVED),
        "brake_power_kw": make_tagged(0.0, Provenance.DERIVED),
        "map_pressure": make_tagged(0.0, Provenance.DERIVED),
    }
    return ResidualState(operating_point=op, residuals=res_dict)


class TestHealthIndexCalculation:
    """Unit tests for normalized Health Index aggregation and component scores."""

    def test_healthy_inputs_produce_nominal_health(
        self,
        engine: HealthSupervisionEngine,
        healthy_residual_state: ResidualState,
    ) -> None:
        health = engine.evaluate_health(residual_state=healthy_residual_state)

        assert health.health_index.valid is True
        assert math.isclose(health.health_index.value, 1.0, abs_tol=1e-3)
        assert health.degradation_state == DegradationState.HEALTHY
        assert health.health_status == DiagnosticStatus.NORMAL
        assert health.trend == "STABLE"
        assert health.health_rate == 0.0

    def test_degraded_inputs_reduce_health_index(
        self,
        engine: HealthSupervisionEngine,
    ) -> None:
        op = OperatingPoint(
            rpm=4000.0,
            map_pressure_pa=120000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=50.0,
        )
        res_dict = {
            "egt_cyl1": make_tagged(-10.0, Provenance.DERIVED),
            "oil_pressure": make_tagged(-80000.0, Provenance.DERIVED),  # Severe oil pressure drop
            "oil_temp": make_tagged(25.0, Provenance.DERIVED),         # High oil temp residual
            "vibration_rms": make_tagged(5.0, Provenance.DERIVED),      # High vibration residual
            "brake_power_kw": make_tagged(-5.0, Provenance.DERIVED),
            "map_pressure": make_tagged(-20000.0, Provenance.DERIVED),
        }
        res_state = ResidualState(operating_point=op, residuals=res_dict)

        health = engine.evaluate_health(residual_state=res_state)

        assert health.health_index.valid is True
        assert 0.0 <= health.health_index.value < 0.85
        assert health.degradation_state in (DegradationState.WATCH, DegradationState.CAUTION, DegradationState.WARNING, DegradationState.CRITICAL)

    def test_severe_degradation_boundary_clamping(
        self,
        engine: HealthSupervisionEngine,
    ) -> None:
        # Extreme bad inputs
        comb_state = CombustionStabilityState(
            misfire_detected=[True, True, True, True],
            overall_combustion_status=DiagnosticStatus.CRITICAL,
        )
        anomaly_res = AnomalyResult(
            is_anomaly=True,
            anomaly_score=5.0,
            threshold=0.5,
            status=InferenceStatus.SUCCESS,
        )
        fault_res = FaultClassificationResult(
            predicted_class=FaultClass.BEARING_WEAR,
            confidence=1.0,
            status=InferenceStatus.SUCCESS,
        )
        health = engine.evaluate_health(
            comb_state=comb_state,
            anomaly_result=anomaly_res,
            fault_result=fault_res,
        )

        assert 0.0 <= health.health_index.value <= 1.0
        assert health.degradation_state in (DegradationState.WARNING, DegradationState.CRITICAL)

    def test_raw_telemetry_rejection(self, engine: HealthSupervisionEngine) -> None:
        raw_record = RawSignalRecord(
            sequence_number=1,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0,
            egt_cyl2_hot_uv=30500.0,
            egt_cyl3_hot_uv=29800.0,
            egt_cyl4_hot_uv=30100.0,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0,
            oil_p_counts=2048,
            map_counts=2048,
            adc_vref_counts=4095,
            crank_period_us=15000.0,
            fuel_pulse_hz=100.0,
            accel_counts_xyz=(100, 100, 100),
            ambient_temp_c=25.0,
            ambient_press_pa=101325.0,
        )
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            engine.evaluate_health(residual_state=raw_record)  # type: ignore[arg-type]


class TestHysteresisAndTrendSupervision:
    """Unit tests for degradation state transitions, hysteresis, and trend tracking."""

    def test_trend_rate_of_change(
        self,
        engine: HealthSupervisionEngine,
        healthy_residual_state: ResidualState,
    ) -> None:
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        h0 = engine.evaluate_health(residual_state=healthy_residual_state, timestamp=t0)
        assert h0.trend == "STABLE"

        # 10 seconds later, degraded input
        t1 = t0 + timedelta(seconds=10)
        op = healthy_residual_state.operating_point
        bad_res = ResidualState(
            operating_point=op,
            residuals={
                "oil_pressure": make_tagged(-100000.0, Provenance.DERIVED),
                "vibration_rms": make_tagged(10.0, Provenance.DERIVED),
            },
        )
        h1 = engine.evaluate_health(residual_state=bad_res, timestamp=t1)
        assert h1.health_rate < 0.0
        assert h1.trend in ("DEGRADATION", "RAPID_DEGRADATION")

    def test_hysteresis_prevents_recovery_flickering(
        self,
        engine: HealthSupervisionEngine,
    ) -> None:
        # Start in healthy
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        engine.evaluate_health(timestamp=t0)

        # Force state to WATCH/CAUTION via bad combustion
        t1 = t0 + timedelta(seconds=5)
        comb_bad = CombustionStabilityState(
            misfire_detected=[True, False, False, False],
            overall_combustion_status=DiagnosticStatus.WARNING,
        )
        h1 = engine.evaluate_health(comb_state=comb_bad, timestamp=t1)
        assert h1.degradation_state != DegradationState.HEALTHY

        # Slight improvement near threshold (e.g. 0.86), but below hysteresis requirement (0.85 + 0.03 = 0.88)
        # Should stay in previous state due to hysteresis
        t2 = t1 + timedelta(seconds=5)
        comb_slightly_better = CombustionStabilityState(
            misfire_detected=[False, False, False, False],
            overall_combustion_status=DiagnosticStatus.NORMAL,
        )
        # Combine with slight anomaly to keep HI around 0.86 (below 0.88 required for HEALTHY recovery)
        anom = AnomalyResult(is_anomaly=True, anomaly_score=0.56, status=InferenceStatus.SUCCESS)
        h2 = engine.evaluate_health(comb_state=comb_slightly_better, anomaly_result=anom, timestamp=t2)
        assert 0.85 <= h2.health_index.value < 0.88
        assert h2.degradation_state == h1.degradation_state


class TestModule13Integration:
    """Verify Module 13 AnomalyDetector & FaultClassifier integration into Module 14."""

    def test_anomaly_result_reduces_health(
        self,
        engine: HealthSupervisionEngine,
        healthy_residual_state: ResidualState,
    ) -> None:
        anom = AnomalyResult(
            is_anomaly=True,
            anomaly_score=0.9,
            status=InferenceStatus.SUCCESS,
        )
        health = engine.evaluate_health(residual_state=healthy_residual_state, anomaly_result=anom)

        assert health.health_index.value < 1.0
        assert "anomaly_fault" in health.evidence

    def test_missing_model_handled_gracefully(
        self,
        engine: HealthSupervisionEngine,
        healthy_residual_state: ResidualState,
    ) -> None:
        anom_missing = AnomalyResult(status=InferenceStatus.MODEL_UNAVAILABLE)
        fault_missing = FaultClassificationResult(status=InferenceStatus.MODEL_UNAVAILABLE)

        health = engine.evaluate_health(
            residual_state=healthy_residual_state,
            anomaly_result=anom_missing,
            fault_result=fault_missing,
        )

        assert health.health_index.valid is True
        assert health.quality < 1.0  # Quality is degraded due to missing ML evidence
