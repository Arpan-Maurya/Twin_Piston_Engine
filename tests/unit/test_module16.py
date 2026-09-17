"""
Unit tests for Original Module 16: Mission Phase and Mission Risk.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.provenance import (
    DegradationState,
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    FaultClassificationResult,
    HealthState,
    MissionState,
    OperatingPoint,
    ProvenanceTaggedValue,
    RULState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.mission_risk import (
    MissionPhaseClassifier,
    MissionRiskEngine,
)


@pytest.fixture
def classifier() -> MissionPhaseClassifier:
    clf = MissionPhaseClassifier()
    clf.reset_state()
    return clf


@pytest.fixture
def risk_engine() -> MissionRiskEngine:
    eng = MissionRiskEngine()
    eng.reset_state()
    return eng


class TestMissionPhaseClassifier:
    """Unit tests for MissionPhaseClassifier."""

    def test_all_configured_phases(self, classifier: MissionPhaseClassifier) -> None:
        # 1. GROUND
        p1, q1, _ = classifier.classify_phase(rpm=800.0, throttle_pct=5.0)
        assert p1 == FlightPhase.GROUND

        # 2. TAKEOFF
        classifier.reset_state()
        p2, q2, _ = classifier.classify_phase(rpm=5200.0, map_pa=115000.0, throttle_pct=90.0, altitude_m=100.0)
        p2_b, _, _ = classifier.classify_phase(rpm=5200.0, map_pa=115000.0, throttle_pct=90.0, altitude_m=100.0)
        assert p2_b == FlightPhase.TAKEOFF

        # 3. CLIMB
        classifier.reset_state()
        classifier.classify_phase(rpm=4600.0, map_pa=100000.0, vertical_rate_m_s=3.0)
        p3, _, _ = classifier.classify_phase(rpm=4600.0, map_pa=100000.0, vertical_rate_m_s=3.0)
        assert p3 == FlightPhase.CLIMB

        # 4. CRUISE
        classifier.reset_state()
        classifier.classify_phase(rpm=4300.0, map_pa=85000.0, vertical_rate_m_s=0.0)
        p4, _, _ = classifier.classify_phase(rpm=4300.0, map_pa=85000.0, vertical_rate_m_s=0.0)
        assert p4 == FlightPhase.CRUISE

        # 5. DESCENT
        classifier.reset_state()
        classifier.classify_phase(rpm=3200.0, map_pa=60000.0, vertical_rate_m_s=-2.5)
        p5, _, _ = classifier.classify_phase(rpm=3200.0, map_pa=60000.0, vertical_rate_m_s=-2.5)
        assert p5 == FlightPhase.DESCENT

        # 6. LANDING
        classifier.reset_state()
        classifier.classify_phase(rpm=2800.0, map_pa=50000.0, altitude_m=100.0, vertical_rate_m_s=-1.5)
        p6, _, _ = classifier.classify_phase(rpm=2800.0, map_pa=50000.0, altitude_m=100.0, vertical_rate_m_s=-1.5)
        assert p6 == FlightPhase.LANDING

    def test_unknown_phase_on_invalid_or_missing_rpm(self, classifier: MissionPhaseClassifier) -> None:
        p_missing, q_missing, _ = classifier.classify_phase(rpm=None)
        assert p_missing == "UNKNOWN"
        assert q_missing == 0.0

        p_nan, q_nan, _ = classifier.classify_phase(rpm=float("nan"))
        assert p_nan == "UNKNOWN"
        assert q_nan == 0.0

    def test_raw_telemetry_rejection(self, classifier: MissionPhaseClassifier) -> None:
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
            classifier.classify_phase(derived_state=raw_record)  # type: ignore[arg-type]


class TestMissionRiskEngine:
    """Unit tests for MissionRiskEngine."""

    def test_nominal_inputs_produce_low_risk(self, risk_engine: MissionRiskEngine) -> None:
        hs = HealthState(
            health_index=make_tagged(1.0, Provenance.DERIVED),
            degradation_state=DegradationState.HEALTHY,
        )
        state = risk_engine.evaluate_mission_risk(
            flight_phase=FlightPhase.GROUND,
            health_state=hs,
        )

        assert 0.0 <= state.risk_score <= 1.0
        assert state.risk_level == "LOW"
        assert state.go_no_go_recommendation == "GO"
        assert state.provenance == Provenance.DERIVED

    def test_phase_specific_risk_multiplier(self, risk_engine: MissionRiskEngine) -> None:
        hs = HealthState(
            health_index=make_tagged(0.60, Provenance.DERIVED),
            degradation_state=DegradationState.CAUTION,
        )

        risk_engine.reset_state()
        state_ground = risk_engine.evaluate_mission_risk(flight_phase=FlightPhase.GROUND, health_state=hs)

        risk_engine.reset_state()
        state_takeoff = risk_engine.evaluate_mission_risk(flight_phase=FlightPhase.TAKEOFF, health_state=hs)

        # TAKEOFF multiplier (1.5) must yield higher risk than GROUND (0.5)
        assert state_takeoff.risk_score > state_ground.risk_score

    def test_anomaly_and_fault_integration(self, risk_engine: MissionRiskEngine) -> None:
        hs = HealthState(
            health_index=make_tagged(0.75, Provenance.DERIVED),
            degradation_state=DegradationState.WATCH,
        )
        anom = AnomalyResult(is_anomaly=True, anomaly_score=0.8, status=InferenceStatus.SUCCESS)
        fault = FaultClassificationResult(
            predicted_class=FaultClass.EXHAUST_VALVE_LEAK,
            confidence=0.9,
            status=InferenceStatus.SUCCESS,
        )

        state = risk_engine.evaluate_mission_risk(
            flight_phase=FlightPhase.CRUISE,
            health_state=hs,
            anomaly_result=anom,
            fault_result=fault,
        )

        assert state.risk_score > 0.30
        assert state.contributing_fault == FaultClass.EXHAUST_VALVE_LEAK
        assert "predicted_class" in state.evidence

    def test_missing_rul_handled_safely(self, risk_engine: MissionRiskEngine) -> None:
        hs = HealthState(health_index=make_tagged(0.95, Provenance.DERIVED))
        rul_missing = RULState(status=InferenceStatus.MODEL_UNAVAILABLE)

        state = risk_engine.evaluate_mission_risk(
            flight_phase=FlightPhase.CRUISE,
            health_state=hs,
            rul_state=rul_missing,
        )

        assert 0.0 <= state.risk_score <= 1.0
        assert state.risk_level in ("LOW", "MODERATE")
        assert "rul_status" in state.evidence

    def test_raw_telemetry_rejection(self, risk_engine: MissionRiskEngine) -> None:
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
            risk_engine.evaluate_mission_risk(health_state=raw_record)  # type: ignore[arg-type]
