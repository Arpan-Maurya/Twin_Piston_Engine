"""
Validation Report & Reproducible Run Record — Module 22.

Defines ValidationRun and ValidationReport models for deterministic benchmark tracking,
provenance auditing, metrics reporting, and reproducible verification runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from src.dataset_validation.harness import ValidationHarnessOutcome
from src.dataset_validation.manifest import DatasetManifest
from src.dataset_validation.taxonomy import ValidationStatus


class ValidationRun(BaseModel):
    """Record of a single reproducible validation execution."""

    validation_run_id: str = Field(description="Unique identifier for the validation execution run")
    dataset_ids: list[str] = Field(description="List of dataset IDs evaluated in this run")
    config_version: str = Field(default="1.0.0")
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed: int | None = Field(default=None)
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: ValidationStatus = Field(default=ValidationStatus.PASS)


class ValidationReport(BaseModel):
    """Structured report aggregating dataset evaluation results and audits."""

    run_id: str = Field(description="Validation run identifier")
    dataset_manifest_references: list[DatasetManifest] = Field(default_factory=list)
    validation_scope: str = Field(default="Full Pipeline Modules 5-19")
    sample_counts: dict[str, int] = Field(default_factory=dict)
    outcomes: list[ValidationHarnessOutcome] = Field(default_factory=list)
    overall_status: ValidationStatus = Field(default=ValidationStatus.PASS)
    warnings: list[str] = Field(default_factory=list)
    unavailable_metrics: list[str] = Field(default_factory=list)
    leakage_audit_passed: bool = Field(default=True)
    schema_audit_passed: bool = Field(default=True)
    integrity_audit_passed: bool = Field(default=True)

    @classmethod
    def create_report(
        cls,
        run_id: str,
        outcomes: list[ValidationHarnessOutcome],
        manifests: list[DatasetManifest],
        seed: int | None = None,
    ) -> ValidationReport:
        """Construct a ValidationReport from harness outcomes."""
        all_passed = True
        leakage_passed = True
        schema_passed = True
        integrity_passed = True

        all_warnings = []
        all_unavailable = []
        sample_counts = {}

        for out in outcomes:
            sample_counts[out.dataset_id] = out.sample_count
            all_warnings.extend(out.warnings)
            all_unavailable.extend(out.unavailable_reasons)

            if out.status == ValidationStatus.FAIL:
                all_passed = False
            if out.ground_truth_leakage_detected:
                leakage_passed = False
            if not out.schema_audit_passed:
                schema_passed = False
            if not out.integrity_audit_passed:
                integrity_passed = False

        status = ValidationStatus.PASS if all_passed else ValidationStatus.FAIL
        if not outcomes or any(o.status == ValidationStatus.NOT_AVAILABLE for o in outcomes):
            if all_passed:
                status = ValidationStatus.PARTIAL

        return cls(
            run_id=run_id,
            dataset_manifest_references=manifests,
            sample_counts=sample_counts,
            outcomes=outcomes,
            overall_status=status,
            warnings=all_warnings,
            unavailable_metrics=all_unavailable,
            leakage_audit_passed=leakage_passed,
            schema_audit_passed=schema_passed,
            integrity_audit_passed=integrity_passed,
        )
