"""
Unit tests for Original Module 22: Dataset/Source Allocation & Validation Harness.
"""

import pytest
from datetime import datetime, timezone

from src.core.provenance import FaultClass, Provenance
from src.dataset_validation import (
    DataAllocator,
    DatasetManifest,
    DatasetQualityValidator,
    DatasetRecordContainer,
    IntendedUse,
    MetricsCalculator,
    SourceType,
    ValidationStatus,
)
from src.l1_data.raw_signal_record import RawSignalRecord


@pytest.fixture
def sample_raw_record():
    return RawSignalRecord(
        timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        source_type=Provenance.SIMULATED,
        egt_cyl1_hot_uv=12000.0,
        egt_cyl2_hot_uv=12100.0,
        egt_cyl3_hot_uv=11950.0,
        egt_cyl4_hot_uv=12050.0,
        egt_cold_c=25.0,
        cht_hot_uv=8000.0,
        cht_cold_c=25.0,
        oil_rtd_ohms=110.0,
        oil_p_counts=2048,
        map_counts=2048,
        adc_vref_counts=4095,
        crank_period_us=12000.0,
        fuel_pulse_hz=50.0,
        accel_counts_xyz=(2048, 2048, 2048),
        ambient_temp_c=25.0,
        ambient_press_pa=101325.0,
    )


class TestDatasetManifestAndContainer:

    def test_manifest_creation(self):
        manifest = DatasetManifest(
            dataset_id="ds_sim_01",
            source_type=SourceType.SIMULATED,
            intended_use=IntendedUse.TRAINING,
        )
        assert manifest.dataset_id == "ds_sim_01"
        assert manifest.source_type == SourceType.SIMULATED
        assert manifest.intended_use == IntendedUse.TRAINING

    def test_container_auto_updates_manifest(self, sample_raw_record):
        manifest = DatasetManifest(
            dataset_id="ds_sim_02",
            source_type=SourceType.SIMULATED,
            intended_use=IntendedUse.TEST,
        )
        container = DatasetRecordContainer(manifest=manifest, records=[sample_raw_record])
        assert container.manifest.records_count == 1
        assert container.manifest.time_start is not None


class TestMetricsCalculator:

    def test_regression_metrics(self):
        y_true = [10.0, 20.0, 30.0]
        y_pred = [11.0, 19.0, 30.0]
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        assert metrics.sample_count == 3
        assert pytest.approx(metrics.mae, 0.01) == 0.666
        assert metrics.mbe == 0.0

    def test_nine_class_classification_metrics(self):
        y_true = [FaultClass.NOMINAL, FaultClass.MISFIRE, FaultClass.COOLING_FAULT]
        y_pred = [FaultClass.NOMINAL, FaultClass.MISFIRE, FaultClass.COOLING_FAULT]
        metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 1.0
        assert metrics.per_class_f1["NOMINAL"] == 1.0
        assert metrics.per_class_f1["MISFIRE"] == 1.0
        assert metrics.per_class_f1["COOLING_FAULT"] == 1.0
        assert len(metrics.confusion_matrix) == 9

    def test_anomaly_metrics(self):
        y_true = [False, False, True, True]
        y_pred = [False, True, True, False]
        metrics = MetricsCalculator.calculate_anomaly_metrics(y_true, y_pred)
        assert metrics.detection_rate == 0.5
        assert metrics.false_positive_rate == 0.5

    def test_rul_metrics(self):
        y_true = [100.0, 80.0, 60.0]
        y_pred = [105.0, 78.0, 61.0]
        bounds = [(90.0, 110.0), (70.0, 85.0), (55.0, 65.0)]
        metrics = MetricsCalculator.calculate_rul_metrics(y_true, y_pred, bounds=bounds)
        assert metrics.sample_count == 3
        assert metrics.bounds_coverage_pct == 1.0


class TestDatasetAuditors:

    def test_quality_and_integrity_audits(self, sample_raw_record):
        manifest = DatasetManifest(
            dataset_id="ds_sim_03",
            source_type=SourceType.SIMULATED,
            intended_use=IntendedUse.DEVELOPMENT,
        )
        container = DatasetRecordContainer(manifest=manifest, records=[sample_raw_record])

        q_res = DatasetQualityValidator.audit_quality(container)
        assert q_res.passed is True
        assert q_res.nan_count == 0

        checksum = DatasetQualityValidator.compute_dataset_checksum(container)
        assert len(checksum) == 64  # SHA-256 hex string

        manifest.checksum_sha256 = checksum
        i_res = DatasetQualityValidator.audit_integrity(container)
        assert i_res.passed is True
