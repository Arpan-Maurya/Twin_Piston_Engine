"""
Data Lineage — Provenance and traceability tagging.

Maintains a lineage record for each NormalizedSignalRecord:
    - Source adapter ID
    - Timestamp of ingestion
    - Sequence number chain
    - Record hash for integrity verification

This enables full traceability from any derived value back to
its original data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.provenance import Provenance
from src.l1_data.signal_record import NormalizedSignalRecord


@dataclass(frozen=True)
class LineageEntry:
    """A single lineage record for one NormalizedSignalRecord."""

    source_id: str
    provenance: Provenance
    timestamp: datetime
    sequence_number: int
    integrity_hash: str
    valid_channel_count: int
    invalid_channel_count: int
    ingestion_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class LineageTracker:
    """Tracks data lineage for all ingested records.

    Maintains an ordered history of lineage entries for audit
    and traceability purposes.
    """

    def __init__(self, max_history: int = 10000) -> None:
        """
        Args:
            max_history: Maximum number of entries to retain.
        """
        self._history: list[LineageEntry] = []
        self._max_history = max_history

    def record(self, signal_record: NormalizedSignalRecord) -> LineageEntry:
        """Create a lineage entry for a signal record.

        Args:
            signal_record: The record to track.

        Returns:
            The created LineageEntry.
        """
        entry = LineageEntry(
            source_id=signal_record.source_id,
            provenance=signal_record.provenance,
            timestamp=signal_record.timestamp,
            sequence_number=signal_record.sequence_number,
            integrity_hash=signal_record.integrity_hash,
            valid_channel_count=signal_record.valid_channel_count,
            invalid_channel_count=signal_record.invalid_channel_count,
        )

        self._history.append(entry)

        # Trim history if it exceeds max
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return entry

    @property
    def history(self) -> list[LineageEntry]:
        """The lineage history (oldest first)."""
        return list(self._history)

    @property
    def last_entry(self) -> LineageEntry | None:
        """The most recent lineage entry."""
        return self._history[-1] if self._history else None

    @property
    def total_records_processed(self) -> int:
        """Total number of records that have been tracked."""
        return len(self._history)
