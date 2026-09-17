"""
Edge Telemetry Envelope — Module 21 Edge/Ground Data Contract.

Transport-safe envelope wrapping canonical RawSignalRecord for Edge -> Ground transmission.
Preserves timestamp, sequence number, provenance, quality metadata, HMAC signature, and schema version.
Strictly shields simulator internals, ground truth, and analytical states from transport.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord


class EdgeTelemetryEnvelope(BaseModel):
    """Canonical Edge Telemetry Transport Envelope."""

    record: RawSignalRecord = Field(description="Canonical raw signal record payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = Field(ge=0)
    source_type: Provenance = Field(default=Provenance.SIMULATED)
    quality: SignalQuality = Field(description="L1 data quality assessment metadata")
    signature: str | None = Field(default=None, description="HMAC packet signature")
    schema_version: str = Field(default="1.0.0", description="Envelope schema version")
    node_id: str = Field(default="edge_node_01", description="Originating edge node identifier")

    @classmethod
    def from_record(
        cls,
        record: RawSignalRecord,
        signature: str | None = None,
        node_id: str = "edge_node_01"
    ) -> EdgeTelemetryEnvelope:
        """Construct a transport envelope from a RawSignalRecord."""
        return cls(
            record=record,
            timestamp=record.timestamp,
            sequence_number=record.sequence_number,
            source_type=record.source_type,
            quality=record.signal_quality,
            signature=signature or getattr(record, "signature", getattr(record, "integrity_hash", None)),
            node_id=node_id,
        )
