"""
Integration tests for ValidationHarness and Post-Inference Metrics (Module 22).
"""

import pytest
from datetime import datetime, timezone

from src.core.provenance import FaultClass, Provenance
from src.dataset_validation import (
    DatasetManifest,
    DatasetRecordContainer,
    IntendedUse,
    SourceType,
    ValidationHarness,
    ValidationStatus,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import ScenarioRunner, SimulationGroundTruth


class TestValidationHarnessIntegration:

    def test_validation_harness_with_simulated_scenario_ground_truth(self):
        runner = ScenarioRunner(seed=42)
        records, ground_truths, meta = runner.run_scenario(duration_s=5.0, dt_s=1.0)

        manifest = DatasetManifest(
            dataset_id="ds_sim_scenario_42",
            source_type=SourceType.SIMULATED,
            intended_use=IntendedUse.VALIDATION,
            ground_truth_available=True,
        )
        container = DatasetRecordContainer(
            manifest=manifest, records=records, ground_truths=ground_truths
        )

        harness = ValidationHarness()
        outcome = harness.evaluate_dataset(container)

        assert outcome.status == ValidationStatus.PASS
        assert outcome.sample_count == 5
        assert outcome.quality_audit_passed is True
        assert outcome.schema_audit_passed is True
        assert outcome.integrity_audit_passed is True
        assert outcome.ground_truth_leakage_detected is False

        assert outcome.physics_validation is not None
        assert outcome.physics_validation.rpm_metrics.mae >= 0.0
        assert outcome.fault_classification_metrics is not None
        assert outcome.fault_classification_metrics.sample_count == 5
        assert outcome.anomaly_metrics is not None

    def test_validation_harness_without_ground_truth_returns_partial(self):
        runner = ScenarioRunner(seed=42)
        records, _, _ = runner.run_scenario(duration_s=3.0, dt_s=1.0)

        manifest = DatasetManifest(
            dataset_id="ds_sim_no_gt",
            source_type=SourceType.SIMULATED,
            intended_use=IntendedUse.TEST,
            ground_truth_available=False,
        )
        container = DatasetRecordContainer(
            manifest=manifest, records=records, ground_truths=None
        )

        harness = ValidationHarness()
        outcome = harness.evaluate_dataset(container)

        assert outcome.status == ValidationStatus.PARTIAL
        assert len(outcome.unavailable_reasons) > 0
        assert "Reference ground truth unavailable or length mismatch" in outcome.unavailable_reasons[0]
