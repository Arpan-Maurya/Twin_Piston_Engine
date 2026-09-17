"""
Validation Harness — Module 22 Backend Validation Engine.

Executes post-inference evaluation of existing backend outputs (Modules 5–19)
against reference ground truth or validation datasets.

STRICT BOUNDARY CONSTRAINTS:
- SimulationGroundTruth is isolated and joined ONLY post-inference.
- Ground truth MUST NOT be inserted into RawSignalRecord or passed into pipeline.
- Returns NOT_AVAILABLE / INSUFFICIENT_REFERENCE when reference data is missing.
- Does NOT fabricate metrics or ML artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from src.core.config import AppSettings, get_settings
from src.core.exceptions import RoleViolationError, ValidationException
from src.core.logging import get_logger
from src.core.provenance import FaultClass
from src.dataset_validation.allocation import LeakageDetector
from src.dataset_validation.manifest import DatasetRecordContainer
from src.dataset_validation.metrics import (
    AnomalyMetrics,
    ClassificationMetrics,
    MetricsCalculator,
    RegressionMetrics,
    RULMetrics,
)
from src.dataset_validation.quality_schema import DatasetQualityValidator
from src.dataset_validation.taxonomy import ValidationStatus
from src.l1_data.simulator.forward_simulator import SimulationGroundTruth
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, PipelineStepResult

logger = get_logger(__name__)


class PhysicsValidationResult(BaseModel):
    """Evaluation result for physics & sensor conversion accuracy."""

    rpm_metrics: RegressionMetrics | None = None
    map_metrics: RegressionMetrics | None = None
    oil_pressure_metrics: RegressionMetrics | None = None
    oil_temp_metrics: RegressionMetrics | None = None
    power_metrics: RegressionMetrics | None = None


class ValidationHarnessOutcome(BaseModel):
    """Complete structured outcome from ValidationHarness evaluation."""

    status: ValidationStatus
    dataset_id: str
    sample_count: int
    quality_audit_passed: bool
    schema_audit_passed: bool
    integrity_audit_passed: bool
    ground_truth_leakage_detected: bool
    physics_validation: PhysicsValidationResult | None = None
    fault_classification_metrics: ClassificationMetrics | None = None
    anomaly_metrics: AnomalyMetrics | None = None
    rul_metrics: RULMetrics | None = None
    warnings: list[str] = Field(default_factory=list)
    unavailable_reasons: list[str] = Field(default_factory=list)


class ValidationHarness:
    """Post-inference Backend Evaluation and Benchmark Harness."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        adapter: PipelineReplayAdapter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._adapter = adapter or PipelineReplayAdapter(self._settings)

    def evaluate_dataset(
        self, container: DatasetRecordContainer
    ) -> ValidationHarnessOutcome:
        """Execute post-inference evaluation of container telemetry records.

        Steps:
        1. Run Quality, Schema, and Integrity Audits.
        2. Check for ground-truth leakage into telemetry records.
        3. Execute pipeline.process_sequence(container.records) -> predictions.
        4. If ground_truths available: perform post-inference comparison against predictions.
        5. Return ValidationHarnessOutcome.
        """
        manifest = container.manifest
        records = container.records
        ground_truths = container.ground_truths
        warnings: list[str] = []
        unavailable_reasons: list[str] = []

        if not records:
            return ValidationHarnessOutcome(
                status=ValidationStatus.NOT_AVAILABLE,
                dataset_id=manifest.dataset_id,
                sample_count=0,
                quality_audit_passed=True,
                schema_audit_passed=True,
                integrity_audit_passed=True,
                ground_truth_leakage_detected=False,
                unavailable_reasons=["Dataset contains no telemetry records"],
            )

        # Step 1: Audits
        q_audit = DatasetQualityValidator.audit_quality(container)
        s_audit = DatasetQualityValidator.audit_schema(container)
        i_audit = DatasetQualityValidator.audit_integrity(container)

        if not q_audit.passed:
            warnings.append(f"Quality audit reported issues: {q_audit.errors}")
        if not s_audit.passed:
            warnings.append(f"Schema audit reported issues: {s_audit.errors}")
        if not i_audit.passed:
            warnings.append(f"Integrity audit reported issues: {i_audit.errors}")

        # Step 2: Ground-truth leakage check
        gt_leakage = any(LeakageDetector.check_ground_truth_leakage(r) for r in records)
        if gt_leakage:
            warnings.append("CRITICAL: Ground-truth leakage detected in raw telemetry input records!")

        # Step 3: Run pipeline inference (without passing ground truths into inference)
        step_results: list[PipelineStepResult] = self._adapter.process_sequence(records)

        # Step 4: Post-inference evaluation against reference ground truth
        if not ground_truths or len(ground_truths) != len(records):
            unavailable_reasons.append("Reference ground truth unavailable or length mismatch")
            return ValidationHarnessOutcome(
                status=ValidationStatus.PARTIAL if (q_audit.passed and s_audit.passed) else ValidationStatus.FAIL,
                dataset_id=manifest.dataset_id,
                sample_count=len(records),
                quality_audit_passed=q_audit.passed,
                schema_audit_passed=s_audit.passed,
                integrity_audit_passed=i_audit.passed,
                ground_truth_leakage_detected=gt_leakage,
                warnings=warnings,
                unavailable_reasons=unavailable_reasons,
            )

        # Post-inference Physics Validation
        gt_rpm = [gt.rpm for gt in ground_truths]
        pred_rpm = [res.derived_rpm for res in step_results]
        rpm_met = MetricsCalculator.calculate_regression_metrics(gt_rpm, pred_rpm)

        gt_map = [gt.map_pressure_pa for gt in ground_truths]
        pred_map = [records[i].map_counts * 50.0 for i in range(len(records))]  # approximate engineering scale for MAP
        map_met = MetricsCalculator.calculate_regression_metrics(gt_map, pred_map)

        gt_power = [gt.power_kw for gt in ground_truths]
        pred_power = [res.derived_power_kw for res in step_results]
        power_met = MetricsCalculator.calculate_regression_metrics(gt_power, pred_power)

        phys_res = PhysicsValidationResult(
            rpm_metrics=rpm_met,
            map_metrics=map_met,
            power_metrics=power_met,
        )

        # Post-inference 9-Class Fault Classification Validation (Module 13 Taxonomy: 0..8)
        gt_faults = [gt.injected_fault_class for gt in ground_truths]
        pred_faults = [res.predicted_fault_class for res in step_results]
        fault_met = MetricsCalculator.calculate_classification_metrics(gt_faults, pred_faults)

        # Post-inference Anomaly Detection Validation
        gt_anom = [gt.injected_fault_class != FaultClass.NOMINAL for gt in ground_truths]
        pred_anom = [res.is_anomaly for res in step_results]
        anom_met = MetricsCalculator.calculate_anomaly_metrics(gt_anom, pred_anom)

        overall_status = ValidationStatus.PASS
        if not (q_audit.passed and s_audit.passed and i_audit.passed) or gt_leakage:
            overall_status = ValidationStatus.FAIL

        return ValidationHarnessOutcome(
            status=overall_status,
            dataset_id=manifest.dataset_id,
            sample_count=len(records),
            quality_audit_passed=q_audit.passed,
            schema_audit_passed=s_audit.passed,
            integrity_audit_passed=i_audit.passed,
            ground_truth_leakage_detected=gt_leakage,
            physics_validation=phys_res,
            fault_classification_metrics=fault_met,
            anomaly_metrics=anom_met,
            warnings=warnings,
            unavailable_reasons=unavailable_reasons,
        )
