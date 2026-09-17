"""
Integration and Boundary audit tests for Original Module 13: Anomaly Detection and Nine-Class Fault Classifier.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.core.provenance import FaultClass, InferenceStatus, Provenance
from src.core.schemas import (
    OperatingPoint,
    ResidualState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.anomaly_fault import AnomalyDetector, NineClassFaultClassifier
from src.l3_ml.ml_infrastructure import MLFeatureVectorBuilder, MLInferenceService


class TestModule13BoundaryAndArchitecture:
    """AST boundary audit and end-to-end integration verification for Module 13."""

    def test_ast_boundary_no_simulator_or_advisory_imports(self) -> None:
        """Verify src/l3_ml/anomaly_fault.py does not import simulator, advisory, or REST API."""
        module_path = Path("src/l3_ml/anomaly_fault.py")
        assert module_path.exists(), "src/l3_ml/anomaly_fault.py must exist"

        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        forbidden = {
            "src.l1_data.simulator",
            "src.l4_advisory",
            "src.api",
            "src.l3_ml.health_index",
            "src.l3_ml.rul",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        assert not alias.name.startswith(f), f"Forbidden import found in Module 13: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for f in forbidden:
                        assert not node.module.startswith(f), f"Forbidden importFrom found in Module 13: {node.module}"

    def test_raw_telemetry_rejection_at_entry_points(self) -> None:
        """Verify RawSignalRecord is strictly rejected at all Module 13 entry points."""
        detector = AnomalyDetector()
        classifier = NineClassFaultClassifier()
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
            detector.detect_anomaly(raw_record)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            detector.detect_anomaly_rule_fallback(raw_record)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            classifier.classify_fault(raw_record)  # type: ignore[arg-type]

    def test_end_to_end_module11_to_module13_pipeline(self) -> None:
        """Integration test: Module 11 Residuals -> Module 12 Feature Vector -> Module 13 Anomaly & Classifier."""
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
        assert fv.valid is True
        assert len(fv.values) == 12

        # 2. Module 12 Inference Service
        service = MLInferenceService()

        # 3. Module 13 Anomaly Detector Execution
        detector = AnomalyDetector(inference_service=service)
        anom_res = detector.detect_anomaly(fv)
        # Missing model should produce MODEL_UNAVAILABLE without crashing
        assert anom_res.status == InferenceStatus.MODEL_UNAVAILABLE

        # Rule fallback execution
        anom_rule = detector.detect_anomaly_rule_fallback(residual_state, threshold=10.0)
        assert anom_rule.is_anomaly is True
        assert anom_rule.is_ml is False

        # 4. Module 13 Fault Classifier Execution
        classifier = NineClassFaultClassifier(inference_service=service)
        class_res = classifier.classify_fault(fv)
        assert class_res.status == InferenceStatus.MODEL_UNAVAILABLE
        assert class_res.predicted_class == FaultClass.NOMINAL

        # Rule fallback classification
        class_rule = classifier.classify_fault_rule_fallback(residual_state=residual_state)
        assert class_rule.predicted_class == FaultClass.INTAKE_BOOST_LEAK
        assert class_rule.is_ml is False
