"""
Unit tests for Original Module 15: Remaining Useful Life (RUL) with Uncertainty.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.core.provenance import (
    DegradationState,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    HealthState,
    ModelMetadata,
    OperatingPoint,
    ProvenanceTaggedValue,
    RULState,
    ResidualState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.ml_infrastructure import (
    BaseMLModel,
    InferenceResult,
    MLFeatureVector,
    MLFeatureVectorBuilder,
    MLInferenceService,
)
from src.l3_ml.rul_estimation import (
    HealthHistoryTracker,
    RULEstimator,
)


class DummyRULModel(BaseMLModel):
    """Dummy RUL model returning predictable remaining hours and confidence bounds."""

    def __init__(self, metadata: ModelMetadata, hours: float = 450.0) -> None:
        super().__init__(metadata)
        self._hours = hours

    def load(self, path: str) -> bool:
        return True

    def predict(self, feature_vector: MLFeatureVector) -> InferenceResult:
        if "raise_error" in feature_vector.features:
            raise RuntimeError("Simulated RUL model crash")

        return InferenceResult(
            model_name=self.metadata.name,
            model_version=self.metadata.version,
            prediction={
                "hours_remaining": self._hours,
                "lower_bound": self._hours * 0.8,
                "upper_bound": self._hours * 1.2,
                "confidence": 0.90,
                "uncertainty": {"std": 45.0},
            },
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
        output_shape=[1],
        accuracy_metrics={"rmse": 15.0},
        description="Unit test RUL model",
    )


@pytest.fixture
def sample_health_state() -> HealthState:
    hi = make_tagged(0.90, Provenance.DERIVED, valid=True, quality=1.0)
    return HealthState(
        timestamp=datetime.now(timezone.utc),
        provenance=Provenance.DERIVED,
        health_index=hi,
        degradation_state=DegradationState.HEALTHY,
        trend="STABLE",
        quality=1.0,
    )


class TestHealthHistoryTracker:
    """Unit tests for HealthHistoryTracker."""

    def test_tracker_adds_valid_samples(self, sample_health_state: HealthState) -> None:
        tracker = HealthHistoryTracker()
        assert tracker.count() == 0

        added = tracker.add_sample(sample_health_state)
        assert added is True
        assert tracker.count() == 1

    def test_tracker_skips_duplicate_timestamps(self, sample_health_state: HealthState) -> None:
        tracker = HealthHistoryTracker()
        tracker.add_sample(sample_health_state)

        # Same sample with identical timestamp
        added_dupe = tracker.add_sample(sample_health_state)
        assert added_dupe is False
        assert tracker.count() == 1

    def test_tracker_resets_on_large_time_gap(self) -> None:
        tracker = HealthHistoryTracker()
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        h0 = HealthState(health_index=make_tagged(0.9, Provenance.DERIVED), timestamp=t0)
        tracker.add_sample(h0)

        # 10 minutes later (> 300s threshold)
        t1 = t0 + timedelta(seconds=600)
        h1 = HealthState(health_index=make_tagged(0.85, Provenance.DERIVED), timestamp=t1)
        tracker.add_sample(h1)

        # Should have reset and kept only h1
        assert tracker.count() == 1

    def test_raw_telemetry_rejection(self) -> None:
        tracker = HealthHistoryTracker()
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
            tracker.add_sample(raw_record)  # type: ignore[arg-type]


class TestRULEstimator:
    """Unit tests for RULEstimator predictive service."""

    def test_insufficient_history_status_on_first_sample(
        self,
        sample_health_state: HealthState,
    ) -> None:
        estimator = RULEstimator()
        rul = estimator.estimate_rul(sample_health_state)

        assert rul.status == InferenceStatus.INSUFFICIENT_HISTORY
        assert rul.hours_remaining == 0.0
        assert rul.lower_bound_hours is None
        assert rul.upper_bound_hours is None
        assert rul.confidence is None
        assert rul.unit == "hours"
        assert rul.is_ml is False

    def test_sufficient_history_degrading_trend(self) -> None:
        estimator = RULEstimator()
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # 4 historical samples with linear degradation
        his = [0.95, 0.90, 0.85, 0.80]
        for i, val in enumerate(his):
            ts = t0 + timedelta(seconds=i * 60)
            hs = HealthState(
                health_index=make_tagged(val, Provenance.DERIVED),
                degradation_state=DegradationState.HEALTHY if val > 0.85 else DegradationState.WATCH,
                timestamp=ts,
            )
            res = estimator.estimate_rul(hs)

        assert res.status == InferenceStatus.SUCCESS
        assert res.hours_remaining > 0.0
        assert res.unit == "hours"
        assert res.is_ml is False
        assert res.provenance == Provenance.DERIVED
        assert res.operating_assumption == "constant_cruise_operating_profile"

        if res.lower_bound_hours is not None and res.upper_bound_hours is not None:
            assert res.lower_bound_hours <= res.hours_remaining <= res.upper_bound_hours

    def test_stable_health_returns_max_horizon(self) -> None:
        estimator = RULEstimator()
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # 4 historical samples with stable health 0.95
        for i in range(4):
            ts = t0 + timedelta(seconds=i * 60)
            hs = HealthState(
                health_index=make_tagged(0.95, Provenance.DERIVED),
                timestamp=ts,
            )
            res = estimator.estimate_rul(hs)

        assert res.status == InferenceStatus.SUCCESS
        assert math.isclose(res.hours_remaining, 2000.0)
        assert res.uncertainty_available is False

    def test_ml_rul_model_inference_success(
        self,
        dummy_metadata: ModelMetadata,
        sample_health_state: HealthState,
    ) -> None:
        service = MLInferenceService()
        model_meta = dummy_metadata.model_copy(update={"name": "rul_estimator"})
        model = DummyRULModel(model_meta, hours=350.0)
        service.register_model_instance(model)

        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()

        estimator = RULEstimator(inference_service=service)
        res = estimator.estimate_rul(sample_health_state, feature_vector=fv)

        assert res.status == InferenceStatus.SUCCESS
        assert math.isclose(res.hours_remaining, 350.0)
        assert math.isclose(res.lower_bound_hours, 280.0)
        assert math.isclose(res.upper_bound_hours, 420.0)
        assert res.confidence == 0.90
        assert res.is_ml is True
        assert res.provenance == Provenance.MODEL_OUTPUT

    def test_ml_model_unavailable_falls_back_to_baseline(
        self,
        sample_health_state: HealthState,
    ) -> None:
        service = MLInferenceService()  # empty service, no model loaded
        builder = MLFeatureVectorBuilder()
        fv = builder.build_feature_vector()

        estimator = RULEstimator(inference_service=service)
        res = estimator.estimate_rul(sample_health_state, feature_vector=fv)

        # First sample has insufficient history -> INSUFFICIENT_HISTORY
        assert res.status == InferenceStatus.INSUFFICIENT_HISTORY
        assert res.is_ml is False

    def test_raw_telemetry_rejection(self) -> None:
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
