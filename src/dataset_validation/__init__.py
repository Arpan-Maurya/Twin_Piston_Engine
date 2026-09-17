"""
Module 22 — Dataset/Source Allocation and Validation Harness Package.

Provides dataset manifests, taxonomy definitions, train/val/test allocation and scenario splitting,
leakage detection, quality/schema/integrity auditing, validation metrics calculation, post-inference
ValidationHarness, and ValidationReport generators.
"""

from src.dataset_validation.allocation import (
    DataAllocator,
    DatasetSplitter,
    LeakageDetector,
)
from src.dataset_validation.harness import (
    PhysicsValidationResult,
    ValidationHarness,
    ValidationHarnessOutcome,
)
from src.dataset_validation.manifest import DatasetManifest, DatasetRecordContainer
from src.dataset_validation.metrics import (
    AnomalyMetrics,
    ClassificationMetrics,
    MetricsCalculator,
    RegressionMetrics,
    RULMetrics,
)
from src.dataset_validation.quality_schema import (
    DatasetQualityValidator,
    IntegrityAuditResult,
    QualityAuditResult,
    SchemaAuditResult,
)
from src.dataset_validation.report import ValidationReport, ValidationRun
from src.dataset_validation.taxonomy import IntendedUse, SourceType, ValidationStatus

__all__ = [
    "SourceType",
    "IntendedUse",
    "ValidationStatus",
    "DatasetManifest",
    "DatasetRecordContainer",
    "DataAllocator",
    "DatasetSplitter",
    "LeakageDetector",
    "DatasetQualityValidator",
    "QualityAuditResult",
    "SchemaAuditResult",
    "IntegrityAuditResult",
    "MetricsCalculator",
    "RegressionMetrics",
    "ClassificationMetrics",
    "AnomalyMetrics",
    "RULMetrics",
    "ValidationHarness",
    "PhysicsValidationResult",
    "ValidationHarnessOutcome",
    "ValidationRun",
    "ValidationReport",
]
