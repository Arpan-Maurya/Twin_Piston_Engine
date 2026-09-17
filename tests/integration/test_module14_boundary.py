"""
Integration and Boundary audit tests for Original Module 14: Health Index and Degradation Supervision.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.core.provenance import DegradationState, DiagnosticStatus, FaultClass, InferenceStatus, Provenance
from src.core.schemas import (
    OperatingPoint,
    ResidualState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.anomaly_fault import AnomalyDetector, NineClassFaultClassifier
from src.l3_ml.health_supervision import HealthSupervisionEngine
from src.l3_ml.ml_infrastructure import MLFeatureVectorBuilder, MLInferenceService


class TestModule14BoundaryAndArchitecture:
    """AST boundary audit and end-to-end pipeline integration for Module 14."""

    def test_ast_boundary_no_simulator_advisory_or_rul_imports(self) -> None:
        """Verify src/l3_ml/health_supervision.py has zero forbidden imports."""
        module_path = Path("src/l3_ml/health_supervision.py")
        assert module_path.exists(), "src/l3_ml/health_supervision.py must exist"

        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        forbidden = {
            "src.l1_data.simulator",
            "src.l4_advisory",
            "src.api",
            "src.l3_ml.rul",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        assert not alias.name.startswith(f), f"Forbidden import found in Module 14: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for f in forbidden:
                        assert not node.module.startswith(f), f"Forbidden importFrom found in Module 14: {node.module}"

    def test_raw_telemetry_rejection_at_entry_point(self) -> None:
        """Verify RawSignalRecord is strictly rejected at Module 14 entry point."""
        engine = HealthSupervisionEngine()
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

    def test_end_to_end_module11_to_module14_pipeline(self) -> None:
        """Integration test: Module 11 -> Module 12 -> Module 13 -> Module 14."""
        op = OperatingPoint(
            rpm=4000.0,
            map_pressure_pa=120000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=50.0,
        )
        res_dict = {
            "egt_cyl1": make_tagged(-15.0, Provenance.DERIVED),
            "egt_cyl2": make_tagged(2.0, Provenance.DERIVED),
            "egt_cyl3": make_tagged(1.0, Provenance.DERIVED),
            "egt_cyl4": make_tagged(0.0, Provenance.DERIVED),
            "oil_pressure": make_tagged(-12000.0, Provenance.DERIVED),
            "oil_temp": make_tagged(5.0, Provenance.DERIVED),
            "vibration_rms": make_tagged(0.8, Provenance.DERIVED),
            "brake_power_kw": make_tagged(-3.0, Provenance.DERIVED),
            "map_pressure": make_tagged(-15000.0, Provenance.DERIVED),
        }
        residual_state = ResidualState(operating_point=op, residuals=res_dict)

        # 1. Module 12 Feature Vector Generation
        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector(residual_state=residual_state)

        # 2. Module 12 & 13 Services
        service = MLInferenceService()
        detector = AnomalyDetector(inference_service=service)
        classifier = NineClassFaultClassifier(inference_service=service)

        anom_res = detector.detect_anomaly_rule_fallback(residual_state, threshold=10.0)
        fault_res = classifier.classify_fault_rule_fallback(residual_state=residual_state)

        # 3. Module 14 Health Index & Degradation Supervision
        health_engine = HealthSupervisionEngine()
        health = health_engine.evaluate_health(
            residual_state=residual_state,
            anomaly_result=anom_res,
            fault_result=fault_res,
        )

        assert 0.0 <= health.health_index.value <= 1.0
        assert health.health_index.valid is True
        assert health.degradation_state in (DegradationState.HEALTHY, DegradationState.WATCH, DegradationState.CAUTION, DegradationState.WARNING)
        assert health.contributing_fault == FaultClass.INTAKE_BOOST_LEAK
        assert health.trend == "STABLE"
