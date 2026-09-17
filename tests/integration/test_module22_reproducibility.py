"""
Integration tests for Reproducibility, ValidationRun, and ValidationReport (Module 22).
"""

import pytest
from datetime import datetime, timezone

from src.core.provenance import Provenance
from src.dataset_validation import (
    DatasetManifest,
    DatasetRecordContainer,
    IntendedUse,
    SourceType,
    ValidationHarness,
    ValidationReport,
    ValidationRun,
    ValidationStatus,
)
from src.l1_data.simulator.forward_simulator import ScenarioRunner


class TestReproducibilityAndReporting:

    def test_validation_run_record_creation(self):
        vrun = ValidationRun(
            validation_run_id="vrun_001",
            dataset_ids=["ds_sim_01"],
            seed=42,
            metrics={"accuracy": 1.0, "mae": 0.05},
            status=ValidationStatus.PASS,
        )
        assert vrun.validation_run_id == "vrun_001"
        assert vrun.status == ValidationStatus.PASS

    def test_deterministic_validation_harness_reproducibility(self):
        # Run 1
        runner1 = ScenarioRunner(seed=100)
        records1, gts1, _ = runner1.run_scenario(duration_s=5.0, dt_s=1.0)
        c1 = DatasetRecordContainer(
            manifest=DatasetManifest(dataset_id="ds_100", source_type=SourceType.SIMULATED, intended_use=IntendedUse.VALIDATION),
            records=records1,
            ground_truths=gts1,
        )
        harness1 = ValidationHarness()
        outcome1 = harness1.evaluate_dataset(c1)

        # Run 2 with identical seed & inputs
        runner2 = ScenarioRunner(seed=100)
        records2, gts2, _ = runner2.run_scenario(duration_s=5.0, dt_s=1.0)
        c2 = DatasetRecordContainer(
            manifest=DatasetManifest(dataset_id="ds_100", source_type=SourceType.SIMULATED, intended_use=IntendedUse.VALIDATION),
            records=records2,
            ground_truths=gts2,
        )
        harness2 = ValidationHarness()
        outcome2 = harness2.evaluate_dataset(c2)

        # Outcomes must be identical
        assert outcome1.status == outcome2.status
        assert outcome1.physics_validation.rpm_metrics.mae == outcome2.physics_validation.rpm_metrics.mae
        assert outcome1.fault_classification_metrics.accuracy == outcome2.fault_classification_metrics.accuracy

    def test_validation_report_aggregation(self):
        runner = ScenarioRunner(seed=50)
        records, gts, _ = runner.run_scenario(duration_s=4.0, dt_s=1.0)
        manifest = DatasetManifest(dataset_id="ds_50", source_type=SourceType.SIMULATED, intended_use=IntendedUse.VALIDATION)
        container = DatasetRecordContainer(manifest=manifest, records=records, ground_truths=gts)

        harness = ValidationHarness()
        outcome = harness.evaluate_dataset(container)

        report = ValidationReport.create_report(
            run_id="run_report_01",
            outcomes=[outcome],
            manifests=[manifest],
            seed=50,
        )

        assert report.run_id == "run_report_01"
        assert report.overall_status == ValidationStatus.PASS
        assert report.sample_counts["ds_50"] == 4
        assert report.leakage_audit_passed is True
        assert report.schema_audit_passed is True
        assert report.integrity_audit_passed is True
