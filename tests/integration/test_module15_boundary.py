"""
Integration and Boundary audit tests for Original Module 15: RUL + Uncertainty.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.core.provenance import DegradationState, InferenceStatus, Provenance
from src.core.schemas import (
    OperatingPoint,
    ResidualState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.anomaly_fault import AnomalyDetector, NineClassFaultClassifier
from src.l3_ml.health_supervision import HealthSupervisionEngine
from src.l3_ml.ml_infrastructure import MLFeatureVectorBuilder, MLInferenceService
from src.l3_ml.rul_estimation import RULEstimator


class TestModule15BoundaryAndArchitecture:
    """AST boundary audit and end-to-end pipeline integration for Module 15."""

    def test_ast_boundary_no_forbidden_imports(self) -> None:
        """Verify src/l3_ml/rul_estimation.py has zero forbidden imports."""
        module_path = Path("src/l3_ml/rul_estimation.py")
        assert module_path.exists(), "src/l3_ml/rul_estimation.py must exist"

        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        forbidden = {
            "src.l1_data.simulator",
            "src.l4_advisory",
            "src.api",
            "src.l3_ml.mission_risk",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        assert not alias.name.startswith(f), f"Forbidden import found in Module 15: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for f in forbidden:
                        assert not node.module.startswith(f), f"Forbidden importFrom found in Module 15: {node.module}"

    def test_raw_telemetry_rejection_at_entry_point(self) -> None:
        """Verify RawSignalRecord is strictly rejected at Module 15 entry point."""
        estimator = RULEstimator()
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
            estimator.estimate_rul(raw_record)  # type: ignore[arg-type]

    def test_end_to_end_module11_to_module15_pipeline(self) -> None:
        """Integration test: Module 11 -> Module 12 -> Module 13 -> Module 14 -> Module 15."""
        op = OperatingPoint(
            rpm=4000.0,
            map_pressure_pa=120000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=50.0,
        )

        builder = MLFeatureVectorBuilder()
        service = MLInferenceService()
        detector = AnomalyDetector(inference_service=service)
        classifier = NineClassFaultClassifier(inference_service=service)
        health_engine = HealthSupervisionEngine()
        rul_estimator = RULEstimator(inference_service=service)

        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # Feed 4 consecutive pipeline steps with slight degradation
        for i in range(4):
            ts = t0 + timedelta(seconds=i * 60)
            res_dict = {
                "egt_cyl1": make_tagged(-5.0 * i, Provenance.DERIVED),
                "oil_pressure": make_tagged(-1000.0 * i, Provenance.DERIVED),
                "oil_temp": make_tagged(1.0 * i, Provenance.DERIVED),
                "vibration_rms": make_tagged(0.1 * i, Provenance.DERIVED),
                "brake_power_kw": make_tagged(-0.5 * i, Provenance.DERIVED),
                "map_pressure": make_tagged(-2000.0 * i, Provenance.DERIVED),
            }
            residual_state = ResidualState(timestamp=ts, operating_point=op, residuals=res_dict)
            fv = builder.build_feature_vector(residual_state=residual_state)

            anom_res = detector.detect_anomaly_rule_fallback(residual_state, threshold=10.0)
            fault_res = classifier.classify_fault_rule_fallback(residual_state=residual_state)

            health = health_engine.evaluate_health(
                residual_state=residual_state,
                anomaly_result=anom_res,
                fault_result=fault_res,
                timestamp=ts,
            )

            rul_result = rul_estimator.estimate_rul(health_state=health, feature_vector=fv)

        # After 4 samples, RUL estimator should have sufficient history
        assert rul_result.status == InferenceStatus.SUCCESS
        assert rul_result.hours_remaining >= 0.0
        assert rul_result.unit == "hours"
        assert rul_result.operating_assumption == "constant_cruise_operating_profile"
        assert rul_result.is_ml is False
        assert rul_result.provenance == Provenance.DERIVED
