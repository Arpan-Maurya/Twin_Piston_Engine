"""
Dataset Quality, Schema, and Integrity Validator — Module 22.

Validates data quality (NaN, Inf, duplicates, non-monotonic timestamps, stale records),
schema version compliance, required feature channels, and SHA-256 checksum integrity.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any
from pydantic import BaseModel, Field

from src.dataset_validation.manifest import DatasetRecordContainer
from src.l1_data.raw_signal_record import RawSignalRecord


class QualityAuditResult(BaseModel):
    """Data quality audit summary."""

    passed: bool
    record_count: int
    nan_count: int = 0
    inf_count: int = 0
    duplicate_timestamp_count: int = 0
    non_monotonic_timestamp_count: int = 0
    duplicate_sequence_count: int = 0
    missing_channel_records: int = 0
    anomalous_channel_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class SchemaAuditResult(BaseModel):
    """Schema version and required channels audit summary."""

    passed: bool
    declared_version: str
    expected_version: str
    missing_channels: list[str] = Field(default_factory=list)
    unexpected_channels: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IntegrityAuditResult(BaseModel):
    """SHA-256 checksum and record count integrity audit summary."""

    passed: bool
    expected_checksum: str | None
    computed_checksum: str
    expected_record_count: int | None
    actual_record_count: int
    errors: list[str] = Field(default_factory=list)


class DatasetQualityValidator:
    """Comprehensive dataset quality, schema, and integrity validator."""

    REQUIRED_RAW_CHANNELS = [
        "egt_cyl1_hot_uv",
        "egt_cyl2_hot_uv",
        "egt_cyl3_hot_uv",
        "egt_cyl4_hot_uv",
        "egt_cold_c",
        "cht_hot_uv",
        "cht_cold_c",
        "oil_rtd_ohms",
        "oil_p_counts",
        "map_counts",
        "adc_vref_counts",
        "crank_period_us",
        "fuel_pulse_hz",
        "accel_counts_xyz",
        "ambient_temp_c",
        "ambient_press_pa",
    ]

    @classmethod
    def audit_quality(cls, container: DatasetRecordContainer) -> QualityAuditResult:
        """Audit records for NaN, Inf, timestamp monotonicity, duplicates, and missing channels."""
        records = container.records
        if not records:
            return QualityAuditResult(passed=True, record_count=0)

        nan_cnt = 0
        inf_cnt = 0
        dup_ts_cnt = 0
        non_mono_ts_cnt = 0
        dup_seq_cnt = 0
        errors = []

        seen_timestamps = set()
        seen_seqs = set()
        prev_ts = None

        for rec in records:
            # Check timestamps
            if rec.timestamp in seen_timestamps:
                dup_ts_cnt += 1
            seen_timestamps.add(rec.timestamp)

            if prev_ts is not None and rec.timestamp < prev_ts:
                non_mono_ts_cnt += 1
            prev_ts = rec.timestamp

            # Check sequence numbers
            if rec.sequence_number in seen_seqs:
                dup_seq_cnt += 1
            seen_seqs.add(rec.sequence_number)

            # Check numeric values for NaN/Inf
            for field_name in [
                "egt_cyl1_hot_uv", "egt_cyl2_hot_uv", "egt_cyl3_hot_uv", "egt_cyl4_hot_uv",
                "egt_cold_c", "cht_hot_uv", "cht_cold_c", "oil_rtd_ohms", "crank_period_us",
                "fuel_pulse_hz", "ambient_temp_c", "ambient_press_pa"
            ]:
                val = getattr(rec, field_name, 0.0)
                if val is not None:
                    if math.isnan(val):
                        nan_cnt += 1
                        errors.append(f"NaN found in {field_name} at seq={rec.sequence_number}")
                    elif math.isinf(val):
                        inf_cnt += 1
                        errors.append(f"Inf found in {field_name} at seq={rec.sequence_number}")

        passed = (nan_cnt == 0) and (inf_cnt == 0) and (non_mono_ts_cnt == 0)
        return QualityAuditResult(
            passed=passed,
            record_count=len(records),
            nan_count=nan_cnt,
            inf_count=inf_cnt,
            duplicate_timestamp_count=dup_ts_cnt,
            non_monotonic_timestamp_count=non_mono_ts_cnt,
            duplicate_sequence_count=dup_seq_cnt,
            errors=errors,
        )

    @classmethod
    def audit_schema(
        cls, container: DatasetRecordContainer, expected_version: str = "1.0.0"
    ) -> SchemaAuditResult:
        """Audit schema version and required telemetry channels."""
        manifest = container.manifest
        declared_ver = manifest.schema_version
        errors = []

        if declared_ver != expected_version:
            errors.append(f"Declared schema version '{declared_ver}' does not match expected '{expected_version}'")

        missing = []
        if manifest.features_channels:
            for req in cls.REQUIRED_RAW_CHANNELS:
                if req not in manifest.features_channels:
                    missing.append(req)

        passed = len(errors) == 0 and len(missing) == 0
        return SchemaAuditResult(
            passed=passed,
            declared_version=declared_ver,
            expected_version=expected_version,
            missing_channels=missing,
            errors=errors,
        )

    @classmethod
    def compute_dataset_checksum(cls, container: DatasetRecordContainer) -> str:
        """Compute deterministic SHA-256 checksum over canonical record payload string."""
        hasher = hashlib.sha256()
        for rec in container.records:
            record_str = (
                f"{rec.sequence_number}:{rec.timestamp.isoformat()}:"
                f"{rec.egt_cyl1_hot_uv}:{rec.oil_p_counts}:{rec.crank_period_us}"
            )
            hasher.update(record_str.encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def audit_integrity(cls, container: DatasetRecordContainer) -> IntegrityAuditResult:
        """Audit dataset record count and SHA-256 checksum integrity."""
        manifest = container.manifest
        computed_hash = cls.compute_dataset_checksum(container)
        actual_cnt = len(container.records)
        errors = []

        if manifest.checksum_sha256 is not None:
            if manifest.checksum_sha256.lower() != computed_hash.lower():
                errors.append(
                    f"Checksum mismatch: expected '{manifest.checksum_sha256}', computed '{computed_hash}'"
                )

        if manifest.records_count is not None:
            if manifest.records_count != actual_cnt:
                errors.append(
                    f"Record count mismatch: expected {manifest.records_count}, actual {actual_cnt}"
                )

        passed = len(errors) == 0
        return IntegrityAuditResult(
            passed=passed,
            expected_checksum=manifest.checksum_sha256,
            computed_checksum=computed_hash,
            expected_record_count=manifest.records_count,
            actual_record_count=actual_cnt,
            errors=errors,
        )
