"""
Dataset Manifest — Module 22 Metadata Model.

Defines DatasetManifest and DatasetRecord for formal dataset registration,
provenance tracking, schema versioning, integrity checksums, and intended-use declarations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from pydantic import BaseModel, Field

from src.core.provenance import Provenance
from src.dataset_validation.taxonomy import IntendedUse, SourceType
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import SimulationGroundTruth


class DatasetManifest(BaseModel):
    """Formal dataset declaration and metadata manifest."""

    dataset_id: str = Field(description="Unique identifier for the dataset")
    version: str = Field(default="1.0.0", description="Semantic dataset version")
    source_type: SourceType = Field(description="Origin category (SIMULATED, REPLAY, LIVE, etc.)")
    schema_version: str = Field(default="1.0.0", description="RawSignalRecord schema version")
    provenance: Provenance = Field(default=Provenance.SIMULATED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = Field(default="", description="Human-readable description of dataset contents")
    records_count: int | None = Field(default=None, ge=0)
    time_start: datetime | None = Field(default=None)
    time_end: datetime | None = Field(default=None)
    sampling_rate_hz: float | None = Field(default=None, gt=0.0)
    features_channels: list[str] = Field(default_factory=list)
    ground_truth_available: bool = Field(default=False)
    intended_use: IntendedUse = Field(default=IntendedUse.DEVELOPMENT)
    checksum_sha256: str | None = Field(default=None)
    file_size_bytes: int | None = Field(default=None, ge=0)
    scenario_id: str | None = Field(default=None)
    mission_id: str | None = Field(default=None)


class DatasetRecordContainer(BaseModel):
    """Container pairing a dataset manifest with telemetry records and optional ground truths."""

    manifest: DatasetManifest
    records: list[RawSignalRecord] = Field(default_factory=list)
    ground_truths: list[SimulationGroundTruth] | None = Field(
        default=None,
        description="Isolated ground truth reference list available ONLY for post-inference validation"
    )

    def model_post_init(self, __context: Any) -> None:
        """Update manifest bounds from records if not explicitly set."""
        if self.records and self.manifest.records_count is None:
            self.manifest.records_count = len(self.records)
            if self.records[0].timestamp:
                self.manifest.time_start = self.records[0].timestamp
            if self.records[-1].timestamp:
                self.manifest.time_end = self.records[-1].timestamp
        if self.ground_truths is not None and len(self.ground_truths) > 0:
            self.manifest.ground_truth_available = True
