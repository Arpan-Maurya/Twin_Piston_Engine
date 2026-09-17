"""
Integration and Boundary audit tests for Original Module 16: Mission Phase and Mission Risk.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.provenance import DegradationState, FaultClass, FlightPhase, InferenceStatus, Provenance
from src.core.schemas import (
    OperatingPoint,
    ResidualState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.anomaly_fault import AnomalyDetector, NineClassFaultClassifier
from src.l3_ml.health_supervision import HealthSupervisionEngine
from src.l3_ml.mission_risk import MissionPhaseClassifier, MissionRiskEngine
from src.l3_ml.ml_infrastructure import MLFeatureVectorBuilder, MLInferenceService
from src.l3_ml.rul_estimation import RULEstimator


class TestModule16BoundaryAndArchitecture:
    """AST boundary audit and end-to-end pipeline integration for Module 16."""

    def test_ast_boundary_no_forbidden_imports(self) -> None:
        """Verify src/l3_ml/mission_risk.py has zero forbidden imports."""
        module_path = Path("src/l3_ml/mission_risk.py")
        assert module_path.exists(), "src/l3_ml/mission_risk.py must exist"

        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        forbidden = {
            "src.l1_data.simulator",
            "src.l4_advisory",
            "src.api",
            "src.l3_ml.forward_model",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        assert not alias.name.startswith(f), f"Forbidden import found in Module 16: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for f in forbidden:
                        assert not node.module.startswith(f), f"Forbidden importFrom found in Module 16: {node.module}"

    def test_raw_telemetry_rejection_at_entry_points(self) -> None:
        """Verify RawSignalRecord is strictly rejected at Module 16 entry points."""
        phase_clf = MissionPhaseClassifier()
        risk_eng = MissionRiskEngine()
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
            phase_clf.classify_phase(derived_state=raw_record)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            risk_eng.evaluate_mission_risk(health_state=raw_record)  # type: ignore[arg-type]

    def test_end_to_end_module11_to_module16_pipeline(self) -> None:
        """Integration test: Module 11 -> Module 12 -> Module 13 -> Module 14 -> Module 15 -> Module 16."""
        op = OperatingPoint(
            rpm=4800.0,
            map_pressure_pa=100000.0,
            altitude_m=1200.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=75.0,
        )

        builder = MLFeatureVectorBuilder()
        service = MLInferenceService()
        detector = AnomalyDetector(inference_service=service)
        classifier = NineClassFaultClassifier(inference_service=service)
        health_engine = HealthSupervisionEngine()
        rul_estimator = RULEstimator(inference_service=service)
        phase_classifier = MissionPhaseClassifier()
        risk_engine = MissionRiskEngine()

        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Mission Phase Classification
        phase, phase_q, transition = phase_classifier.classify_phase(
            rpm=op.rpm,
            map_pa=op.map_pressure_pa,
            throttle_pct=op.throttle_pct,
            altitude_m=op.altitude_m,
            vertical_rate_m_s=0.0,
            timestamp=t0,
        )

        # 2. Residuals & L3 Pipeline steps
        res_dict = {
            "egt_cyl1": make_tagged(-15.0, Provenance.DERIVED),
            "oil_pressure": make_tagged(-12000.0, Provenance.DERIVED),
            "oil_temp": make_tagged(5.0, Provenance.DERIVED),
            "vibration_rms": make_tagged(0.8, Provenance.DERIVED),
            "brake_power_kw": make_tagged(-3.0, Provenance.DERIVED),
            "map_pressure": make_tagged(-15000.0, Provenance.DERIVED),
        }
        residual_state = ResidualState(timestamp=t0, operating_point=op, residuals=res_dict)
        fv = builder.build_feature_vector(residual_state=residual_state)

        anom_res = detector.detect_anomaly_rule_fallback(residual_state, threshold=10.0)
        fault_res = classifier.classify_fault_rule_fallback(residual_state=residual_state)

        health = health_engine.evaluate_health(
            residual_state=residual_state,
            anomaly_result=anom_res,
            fault_result=fault_res,
            timestamp=t0,
        )

        rul_result = rul_estimator.estimate_rul(health_state=health, feature_vector=fv)

        # 3. Mission Risk Evaluation
        mission_state = risk_engine.evaluate_mission_risk(
            flight_phase=phase,
            health_state=health,
            rul_state=rul_result,
            anomaly_result=anom_res,
            fault_result=fault_res,
            mission_id="MISSION-ALPHA-01",
            mission_elapsed_hours=1.5,
            phase_quality=phase_q,
            phase_transition=transition,
            timestamp=t0,
        )

        assert 0.0 <= mission_state.risk_score <= 1.0
        assert mission_state.flight_phase == FlightPhase.CRUISE
        assert mission_state.mission_id == "MISSION-ALPHA-01"
        assert mission_state.risk_level in ("LOW", "MODERATE", "HIGH", "CRITICAL")
        assert mission_state.provenance == Provenance.DERIVED
        assert mission_state.quality > 0.0
