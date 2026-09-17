"""
Unit tests for Original Module 13: Anomaly Detection and Nine-Class Fault Classifier.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pytest

from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    CombustionStabilityState,
    DerivedEngineState,
    DiagnosticState,
    FaultClassificationResult,
    LubricationState,
    ModelMetadata,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
    make_tagged,
)
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.anomaly_fault import (
    AnomalyDetector,
    NineClassFaultClassifier,
)
from src.l3_ml.ml_infrastructure import (
    BaseMLModel,
    InferenceResult,
    MLFeatureVector,
    MLFeatureVectorBuilder,
    MLInferenceService,
)


class DummyAnomalyModel(BaseMLModel):
    """Dummy model returning predictable anomaly scores for testing."""

    def __init__(self, metadata: ModelMetadata, is_anomaly: bool = True, score: float = 0.85) -> None:
        super().__init__(metadata)
        self._is_anomaly = is_anomaly
        self._score = score

    def load(self, path: str) -> bool:
        return True

    def predict(self, feature_vector: MLFeatureVector) -> InferenceResult:
        if "raise_error" in feature_vector.features:
            raise RuntimeError("Simulated anomaly model crash")

        return InferenceResult(
            model_name=self.metadata.name,
            model_version=self.metadata.version,
            prediction={"score": self._score, "is_anomaly": self._is_anomaly, "threshold": 0.5},
            status=InferenceStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
            feature_schema_version="1.0.0",
        )


class DummyClassifierModel(BaseMLModel):
    """Dummy classifier returning 9-class probability distributions for testing."""

    def __init__(self, metadata: ModelMetadata, predicted_class_id: int = 1) -> None:
        super().__init__(metadata)
        self._predicted_class_id = predicted_class_id

    def load(self, path: str) -> bool:
        return True

    def predict(self, feature_vector: MLFeatureVector) -> InferenceResult:
        if "raise_error" in feature_vector.features:
            raise RuntimeError("Simulated classifier model crash")

        # Build 9-class probabilities where target class gets 0.8
        probs = [0.025] * 9
        probs[self._predicted_class_id] = 0.8

        return InferenceResult(
            model_name=self.metadata.name,
            model_version=self.metadata.version,
            prediction=probs,
            status=InferenceStatus.SUCCESS,
            timestamp=datetime.now(timezone.utc),
            feature_schema_version="1.0.0",
        )


@pytest.fixture
def dummy_metadata() -> ModelMetadata:
    return ModelMetadata(
        name="test_model",
        version="1.0.0",
        trained_at=datetime.now(timezone.utc),
        input_shape=[12],
        output_shape=[9],
        accuracy_metrics={"f1": 0.95},
        description="Unit test dummy model",
    )


@pytest.fixture
def sample_feature_vector() -> MLFeatureVector:
    builder = MLFeatureVectorBuilder()
    op = OperatingPoint(
        rpm=4000.0,
        map_pressure_pa=120000.0,
        altitude_m=1000.0,
        ambient_temp_k=288.15,
        ambient_pressure_pa=101325.0,
        throttle_pct=50.0,
    )
    res_dict = {
        "egt_cyl1": make_tagged(-1.0, Provenance.DERIVED),
        "egt_cyl2": make_tagged(0.0, Provenance.DERIVED),
        "egt_cyl3": make_tagged(1.0, Provenance.DERIVED),
        "egt_cyl4": make_tagged(-0.5, Provenance.DERIVED),
        "oil_pressure": make_tagged(100.0, Provenance.DERIVED),
        "oil_temp": make_tagged(0.2, Provenance.DERIVED),
        "vibration_rms": make_tagged(0.05, Provenance.DERIVED),
        "brake_power_kw": make_tagged(0.0, Provenance.DERIVED),
    }
    residual_state = ResidualState(operating_point=op, residuals=res_dict)
    return builder.build_feature_vector(residual_state=residual_state)


class TestNineClassTaxonomy:
    """Verify exact nine-class fault taxonomy contract."""

    def test_taxonomy_preservation(self) -> None:
        assert len(FaultClass) == 9
        expected_names = [
            "NOMINAL",
            "MISFIRE",
            "DETONATION_KNOCK",
            "EXHAUST_VALVE_LEAK",
            "INTAKE_BOOST_LEAK",
            "OIL_DEGRADATION",
            "COOLING_FAULT",
            "BEARING_WEAR",
            "SENSOR_FAULT",
        ]
        for idx, name in enumerate(expected_names):
            fc = FaultClass(idx)
            assert fc.name == name
            assert fc.value == idx


class TestAnomalyDetector:
    """Unit tests for AnomalyDetector service."""

    def test_anomaly_detection_with_valid_model(
        self,
        dummy_metadata: ModelMetadata,
        sample_feature_vector: MLFeatureVector,
    ) -> None:
        service = MLInferenceService()
        model_meta = dummy_metadata.model_copy(update={"name": "anomaly_detector"})
        model = DummyAnomalyModel(model_meta, is_anomaly=True, score=0.88)
        service.register_model_instance(model)

        detector = AnomalyDetector(inference_service=service)
        res = detector.detect_anomaly(sample_feature_vector)

        assert res.status == InferenceStatus.SUCCESS
        assert res.is_anomaly is True
        assert math.isclose(res.anomaly_score, 0.88)
        assert res.is_ml is True
        assert res.model_metadata is not None
        assert res.model_metadata.name == "anomaly_detector"

    def test_anomaly_detection_missing_model_returns_model_unavailable(
        self,
        sample_feature_vector: MLFeatureVector,
    ) -> None:
        service = MLInferenceService()
        detector = AnomalyDetector(inference_service=service)
        res = detector.detect_anomaly(sample_feature_vector)

        assert res.status == InferenceStatus.MODEL_UNAVAILABLE
        assert res.is_anomaly is False
        assert res.is_ml is True

    def test_raw_telemetry_rejection(self) -> None:
        detector = AnomalyDetector()
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

    def test_rule_based_fallback_detection(self) -> None:
        detector = AnomalyDetector()
        op = OperatingPoint(
            rpm=4000.0,
            map_pressure_pa=120000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=50.0,
        )
        # Normal residuals
        norm_res = ResidualState(
            operating_point=op,
            residuals={"egt_cyl1": make_tagged(0.5, Provenance.DERIVED)},
        )
        res_norm = detector.detect_anomaly_rule_fallback(norm_res, threshold=3.0)
        assert res_norm.is_anomaly is False
        assert res_norm.is_ml is False
        assert res_norm.provenance == Provenance.DERIVED

        # Anomalous residuals
        anom_res = ResidualState(
            operating_point=op,
            residuals={"egt_cyl1": make_tagged(15.0, Provenance.DERIVED)},
        )
        res_anom = detector.detect_anomaly_rule_fallback(anom_res, threshold=3.0)
        assert res_anom.is_anomaly is True
        assert res_anom.is_ml is False


class TestNineClassFaultClassifier:
    """Unit tests for NineClassFaultClassifier service."""

    def test_classifier_with_valid_model(
        self,
        dummy_metadata: ModelMetadata,
        sample_feature_vector: MLFeatureVector,
    ) -> None:
        service = MLInferenceService()
        model_meta = dummy_metadata.model_copy(update={"name": "fault_classifier"})
        # Predict class 3: EXHAUST_VALVE_LEAK
        model = DummyClassifierModel(model_meta, predicted_class_id=3)
        service.register_model_instance(model)

        classifier = NineClassFaultClassifier(inference_service=service)
        res = classifier.classify_fault(sample_feature_vector)

        assert res.status == InferenceStatus.SUCCESS
        assert res.predicted_class == FaultClass.EXHAUST_VALVE_LEAK
        assert res.class_id == 3
        assert res.class_name == "EXHAUST_VALVE_LEAK"
        assert res.confidence is not None
        assert math.isclose(res.confidence, 0.8)
        assert res.probabilities is not None
        assert len(res.probabilities) == 9
        assert res.is_ml is True

    def test_classifier_missing_model_returns_model_unavailable(
        self,
        sample_feature_vector: MLFeatureVector,
    ) -> None:
        service = MLInferenceService()
        classifier = NineClassFaultClassifier(inference_service=service)
        res = classifier.classify_fault(sample_feature_vector)

        assert res.status == InferenceStatus.MODEL_UNAVAILABLE
        assert res.predicted_class == FaultClass.NOMINAL
        assert res.class_id == 0
        assert res.confidence is None
        assert res.probabilities is None

    def test_classifier_schema_mismatch_invalid_probs(
        self,
        dummy_metadata: ModelMetadata,
        sample_feature_vector: MLFeatureVector,
    ) -> None:
        service = MLInferenceService()
        model_meta = dummy_metadata.model_copy(update={"name": "fault_classifier"})

        # Bad model returning 5 probabilities instead of 9
        class BadProbModel(BaseMLModel):
            def load(self, path: str) -> bool:
                return True

            def predict(self, feature_vector: MLFeatureVector) -> InferenceResult:
                return InferenceResult(
                    model_name=self.metadata.name,
                    model_version=self.metadata.version,
                    prediction=[0.2] * 5,  # Invalid length
                    status=InferenceStatus.SUCCESS,
                    timestamp=datetime.now(timezone.utc),
                )

        service.register_model_instance(BadProbModel(model_meta))
        classifier = NineClassFaultClassifier(inference_service=service)
        res = classifier.classify_fault(sample_feature_vector)

        assert res.status == InferenceStatus.SCHEMA_MISMATCH

    def test_rule_fallback_fault_classifier(self) -> None:
        classifier = NineClassFaultClassifier()

        # Combustion misfire rule fallback test
        comb_state = CombustionStabilityState(
            misfire_detected=[True, False, False, False],
            overall_combustion_status=DiagnosticStatus.WARNING,
        )
        res_misfire = classifier.classify_fault_rule_fallback(comb_state=comb_state)
        assert res_misfire.predicted_class == FaultClass.MISFIRE
        assert res_misfire.class_id == 1
        assert res_misfire.is_ml is False
        assert res_misfire.provenance == Provenance.DERIVED

        # Vibration bearing wear rule fallback test
        vib_state = VibrationState(
            rms_x_m_s2=make_tagged(12.0, Provenance.DERIVED),
            rms_y_m_s2=make_tagged(12.0, Provenance.DERIVED),
            rms_z_m_s2=make_tagged(12.0, Provenance.DERIVED),
            overall_rms_m_s2=make_tagged(20.8, Provenance.DERIVED),
            peak_m_s2=make_tagged(30.0, Provenance.DERIVED),
            crest_factor=make_tagged(1.4, Provenance.DERIVED),
            dominant_freq_hz=make_tagged(120.0, Provenance.DERIVED),
            dominant_amplitude_m_s2=make_tagged(15.0, Provenance.DERIVED),
            dominant_order=make_tagged(2.0, Provenance.DERIVED),
        )
        res_vib = classifier.classify_fault_rule_fallback(vib_state=vib_state)
        assert res_vib.predicted_class == FaultClass.BEARING_WEAR
        assert res_vib.class_id == 7
        assert res_vib.is_ml is False
