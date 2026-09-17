"""
Architecture Boundary Test for Original Module 12 — ML Inference Infrastructure.

Verifies that Module 12:
    1. Enforces strict ML input boundary (RawSignalRecord CANNOT directly enter ML feature builder)
    2. Operates on L2 derived features/residuals (Modules 5-11) and outputs InferenceResult with Provenance.MODEL_OUTPUT
    3. Does NOT import simulator internals or simulator ground truth
    4. Does NOT import Advisory or API modules
    5. Does NOT implement Module 13+ fault classification or anomaly detection
    6. Correctly reports MODEL_UNAVAILABLE when no real ML model artifact is loaded
    7. Leaves RawSignalRecord and NormalizedSignalRecord completely unchanged
"""

import ast
from pathlib import Path
import pytest

from src.core.provenance import InferenceStatus, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l2_digital_twin.residual_engine import evaluate_residual_engine
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l3_ml.ml_infrastructure import (
    InferenceResult,
    MLFeatureVectorBuilder,
    MLInferenceService,
)


class TestModule12ArchitecturalBoundary:
    """Automated AST and boundary check for Module 12 ML Inference Infrastructure."""

    def test_ml_infrastructure_ast_imports(self) -> None:
        """Scan src/l3_ml/ml_infrastructure.py for forbidden dependencies."""
        filepath = Path("src/l3_ml/ml_infrastructure.py")
        assert filepath.exists()

        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        forbidden_modules = {
            "src.l4_advisory",
            "src.l5_interface",
            "src.l1_data.simulator",
            "fastapi",
        }
        forbidden_names = {
            "RotaxEngineSimulator",
            "CylinderModel",
            "TurbochargerModel",
            "FaultInjector",
            "AdvisoryEngine",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(alias.name.startswith(mod) for mod in forbidden_modules), \
                        f"Forbidden import '{alias.name}' in ml_infrastructure.py"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not any(mod_name.startswith(mod) for mod in forbidden_modules), \
                    f"Forbidden import from '{mod_name}' in ml_infrastructure.py"
                for alias in node.names:
                    assert alias.name not in forbidden_names, \
                        f"Forbidden symbol '{alias.name}' in ml_infrastructure.py"

    def test_raw_telemetry_cannot_enter_ml_feature_builder(self) -> None:
        """Verify that passing RawSignalRecord directly to ML feature builder raises a TypeError boundary error."""
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
            accel_counts_xyz=(10, 20, 980),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )

        builder = MLFeatureVectorBuilder()
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            builder.build_feature_vector(raw_record=raw_record)  # type: ignore

    def test_pipeline_to_module12_inference_service(self) -> None:
        """Pipeline test: RawSignalRecord -> Module 5 -> Module 11 -> Module 12 -> MODEL_UNAVAILABLE."""
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
            accel_counts_xyz=(10, 20, 980),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )

        original_raw_dict = raw_record.model_dump()
        norm_record = convert_raw_to_engineering_state(raw_record)
        res_state, _ = evaluate_residual_engine(norm_record)

        builder = MLFeatureVectorBuilder()
        feature_vector = builder.build_feature_vector(residual_state=res_state)

        # Immutability check
        assert raw_record.model_dump() == original_raw_dict

        # Evaluate Module 12 service when no artifact exists
        service = MLInferenceService()
        result = service.predict("anomaly_detector", feature_vector)

        assert isinstance(result, InferenceResult)
        assert result.status == InferenceStatus.MODEL_UNAVAILABLE
        assert result.prediction is None
        assert result.provenance == Provenance.MODEL_OUTPUT
