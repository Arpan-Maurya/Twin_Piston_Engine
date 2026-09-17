"""
Unit Tests for Original Module 12 — ML Inference Infrastructure.

Tests feature vector building, raw telemetry boundary enforcement, model lifecycle,
schema validation, inference execution statuses, latency tracking, reload safety, and robustness.
"""

import math
from datetime import datetime, timezone
from typing import Any

import pytest

from src.core.provenance import (
    ChannelValidity,
    InferenceStatus,
    ModelLifecycleStatus,
    Provenance,
)
from src.core.schemas import (
    DerivedEngineState,
    ModelMetadata,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    SignalQuality,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l3_ml.ml_infrastructure import (
    BaseMLModel,
    InferenceResult,
    MLFeatureVector,
    MLFeatureVectorBuilder,
    MLInferenceService,
    MLModelLoader,
    STANDARD_FEATURE_ORDER,
)


class DummyTestModel(BaseMLModel):
    """Test model implementation for unit tests."""

    def predict(self, feature_vector: MLFeatureVector) -> list[float]:
        if "raise_error" in feature_vector.features:
            raise RuntimeError("Model execution error test")
        return [sum(feature_vector.values)]


@pytest.fixture
def dummy_metadata() -> ModelMetadata:
    return ModelMetadata(
        name="test_model",
        version="1.0.0",
        trained_at=datetime.now(timezone.utc),
        input_shape=[12],
        output_shape=[1],
        description="Unit test dummy model",
    )


@pytest.fixture
def valid_residual_state() -> ResidualState:
    op = OperatingPoint(
        rpm=4000.0,
        map_pressure_pa=120000.0,
        altitude_m=1000.0,
        ambient_temp_k=288.15,
        ambient_pressure_pa=101325.0,
        throttle_pct=50.0,
    )
    res_dict = {
        "egt_cyl1": make_tagged(10.0, Provenance.DERIVED),
        "egt_cyl2": make_tagged(-5.0, Provenance.DERIVED),
        "egt_cyl3": make_tagged(0.0, Provenance.DERIVED),
        "egt_cyl4": make_tagged(5.0, Provenance.DERIVED),
        "oil_pressure": make_tagged(-10000.0, Provenance.DERIVED),
        "oil_temp": make_tagged(2.0, Provenance.DERIVED),
        "vibration_rms": make_tagged(0.5, Provenance.DERIVED),
        "brake_power_kw": make_tagged(-2.0, Provenance.DERIVED),
    }
    return ResidualState(operating_point=op, residuals=res_dict)


class TestMLFeatureVectorBuilder:
    """Unit tests for ML feature vector builder and raw boundary enforcement."""

    def test_raw_telemetry_rejection(self) -> None:
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

        # Direct raw argument MUST raise TypeError
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            builder.build_feature_vector(raw_record=raw_record)  # type: ignore

    def test_deterministic_feature_vector_building(self, valid_residual_state: ResidualState) -> None:
        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector(residual_state=valid_residual_state)

        assert fv.valid
        assert fv.feature_names == STANDARD_FEATURE_ORDER
        assert len(fv.values) == len(STANDARD_FEATURE_ORDER)
        assert fv.provenance == Provenance.DERIVED

    def test_nan_inf_rejection(self) -> None:
        builder = MLFeatureVectorBuilder()
        op = OperatingPoint(
            rpm=4000.0,
            map_pressure_pa=120000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=50.0,
        )
        bad_res = ResidualState(operating_point=op, residuals={"egt_cyl1": make_tagged(float("nan"), Provenance.DERIVED)})

        fv = builder.build_feature_vector(residual_state=bad_res)
        assert not fv.valid


class TestMLInferenceService:
    """Unit tests for ML Model lifecycle and inference execution."""

    def test_missing_model_returns_model_unavailable(self) -> None:
        service = MLInferenceService()
        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()

        res = service.predict("nonexistent_model", fv)

        assert res.status == InferenceStatus.MODEL_UNAVAILABLE
        assert res.prediction is None
        assert res.provenance == Provenance.MODEL_OUTPUT

    def test_valid_model_inference_success(self, dummy_metadata: ModelMetadata) -> None:
        service = MLInferenceService()
        model = DummyTestModel(dummy_metadata, feature_schema_version="1.0.0")
        service.register_model_instance(model)

        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()

        res = service.predict("test_model", fv)

        assert res.status == InferenceStatus.SUCCESS
        assert res.prediction is not None
        assert res.model_name == "test_model"
        assert res.model_version == "1.0.0"
        assert res.latency_ms >= 0.0

    def test_schema_mismatch_rejection(self, dummy_metadata: ModelMetadata) -> None:
        service = MLInferenceService()
        # Model expects 5 features, but standard vector has 12
        bad_meta = dummy_metadata.model_copy(update={"input_shape": [5]})
        model = DummyTestModel(bad_meta, feature_schema_version="1.0.0")
        service.register_model_instance(model)

        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()

        res = service.predict("test_model", fv)

        assert res.status == InferenceStatus.SCHEMA_MISMATCH
        assert res.prediction is None

    def test_invalid_input_vector_rejection(self, dummy_metadata: ModelMetadata) -> None:
        service = MLInferenceService()
        model = DummyTestModel(dummy_metadata, feature_schema_version="1.0.0")
        service.register_model_instance(model)

        bad_fv = MLFeatureVector(
            feature_names=STANDARD_FEATURE_ORDER,
            values=[float("nan")] * len(STANDARD_FEATURE_ORDER),
            features={},
            valid=False,
        )

        res = service.predict("test_model", bad_fv)

        assert res.status == InferenceStatus.INVALID_INPUT
        assert res.prediction is None

    def test_model_execution_error_isolation(self, dummy_metadata: ModelMetadata) -> None:
        service = MLInferenceService()
        model = DummyTestModel(dummy_metadata, feature_schema_version="1.0.0")
        service.register_model_instance(model)

        # Vector with special trigger key
        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()
        fv_dict = fv.model_dump()
        fv_dict["features"]["raise_error"] = {"name": "raise_error", "value": 1.0, "provenance": "DERIVED", "quality": 1.0, "valid": True}
        error_fv = MLFeatureVector(**fv_dict)

        res = service.predict("test_model", error_fv)

        assert res.status == InferenceStatus.INFERENCE_ERROR
        assert res.error_message is not None

    def test_reload_model_failure_preserves_previous_valid(self, dummy_metadata: ModelMetadata) -> None:
        service = MLInferenceService()
        model = DummyTestModel(dummy_metadata, feature_schema_version="1.0.0")
        service.register_model_instance(model)

        # Attempt reload with invalid non-existent path
        status = service.reload_model("test_model", "invalid_path/model.pkl", dummy_metadata)

        # Previous model must remain registered and ready
        assert service.is_model_loaded("test_model")
