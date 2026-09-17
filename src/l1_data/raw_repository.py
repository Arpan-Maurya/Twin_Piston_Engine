"""
Raw Telemetry Persistence & History Module — Module 4.

Stores validated RawSignalRecord objects without modifying payload, timestamps,
sequence numbers, raw channel values, integrity metadata, or signal quality metadata.

Provides:
    - RawTelemetryRepositoryProtocol: Clean repository abstraction.
    - InMemoryRawTelemetryRepository: Fast, deterministic, in-memory repository.
    - SQLiteRawTelemetryRepository: Prototype-safe local SQLite persistence store.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


@runtime_checkable
class RawTelemetryRepositoryProtocol(Protocol):
    """Protocol abstraction for RawSignalRecord persistence and retrieval."""

    def save(self, record: RawSignalRecord) -> bool:
        """Persist a validated RawSignalRecord.

        Returns True if saved successfully, False if persistence failed.
        """
        ...

    def get_by_sequence(self, sequence_number: int) -> RawSignalRecord | None:
        """Retrieve record by sequence number."""
        ...

    def get_by_timestamp_range(
        self, start_time: datetime, end_time: datetime
    ) -> list[RawSignalRecord]:
        """Retrieve records within chronological timestamp range [start_time, end_time]."""
        ...

    def get_latest(self) -> RawSignalRecord | None:
        """Retrieve the most recently persisted RawSignalRecord."""
        ...

    def get_history(self, limit: int = 100, offset: int = 0) -> list[RawSignalRecord]:
        """Retrieve chronological history of persisted records with pagination."""
        ...

    def count(self) -> int:
        """Return total number of persisted records."""
        ...

    def clear(self) -> None:
        """Clear all persisted telemetry state."""
        ...


class InMemoryRawTelemetryRepository:
    """In-memory, deterministic implementation of RawTelemetryRepositoryProtocol."""

    def __init__(self, max_records: int = 100000) -> None:
        self.max_records = max_records
        self._records: dict[int, RawSignalRecord] = {}
        self._order: list[int] = []  # Maintains chronological/sequence order

    def save(self, record: RawSignalRecord) -> bool:
        """Save RawSignalRecord to memory.

        Catches errors gracefully and returns False without crashing callers.
        """
        try:
            seq = record.sequence_number
            if seq in self._records:
                logger.warning("Duplicate sequence number %d in raw repository — overwriting", seq)
            else:
                self._order.append(seq)

            self._records[seq] = record

            # Evict oldest if exceeding capacity
            if len(self._order) > self.max_records:
                oldest_seq = self._order.pop(0)
                self._records.pop(oldest_seq, None)

            return True
        except Exception as e:
            logger.error("Failed to persist raw telemetry record: %s", e)
            return False

    def get_by_sequence(self, sequence_number: int) -> RawSignalRecord | None:
        return self._records.get(sequence_number)

    def get_by_timestamp_range(
        self, start_time: datetime, end_time: datetime
    ) -> list[RawSignalRecord]:
        # Ensure start_time and end_time are timezone-aware UTC for comparison
        st = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
        et = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)

        results = []
        for seq in self._order:
            rec = self._records[seq]
            rec_ts = rec.timestamp if rec.timestamp.tzinfo else rec.timestamp.replace(tzinfo=timezone.utc)
            if st <= rec_ts <= et:
                results.append(rec)
        return results

    def get_latest(self) -> RawSignalRecord | None:
        if not self._order:
            return None
        return self._records[self._order[-1]]

    def get_history(self, limit: int = 100, offset: int = 0) -> list[RawSignalRecord]:
        if offset < 0 or limit <= 0:
            return []
        slice_seqs = self._order[offset : offset + limit]
        return [self._records[seq] for seq in slice_seqs]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._order.clear()


class SQLiteRawTelemetryRepository:
    """SQLite-backed local persistence store for RawSignalRecord objects."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_telemetry (
                    sequence_number INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def save(self, record: RawSignalRecord) -> bool:
        try:
            payload_json = record.model_dump_json()
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO raw_telemetry
                    (sequence_number, timestamp, source_type, integrity_hash, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.sequence_number,
                        record.timestamp.isoformat(),
                        record.source_type.value,
                        record.integrity_hash,
                        payload_json,
                    ),
                )
            return True
        except Exception as e:
            logger.error("SQLite failure persisting raw telemetry record %d: %s", record.sequence_number, e)
            return False

    def get_by_sequence(self, sequence_number: int) -> RawSignalRecord | None:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM raw_telemetry WHERE sequence_number = ?",
                (sequence_number,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return RawSignalRecord.model_validate_json(row[0])
        except Exception as e:
            logger.error("SQLite error getting sequence %d: %s", sequence_number, e)
            return None

    def get_by_timestamp_range(
        self, start_time: datetime, end_time: datetime
    ) -> list[RawSignalRecord]:
        try:
            st_iso = (start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)).isoformat()
            et_iso = (end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)).isoformat()

            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT payload_json FROM raw_telemetry
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY sequence_number ASC
                """,
                (st_iso, et_iso),
            )
            rows = cursor.fetchall()
            return [RawSignalRecord.model_validate_json(r[0]) for r in rows]
        except Exception as e:
            logger.error("SQLite error getting timestamp range: %s", e)
            return []

    def get_latest(self) -> RawSignalRecord | None:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM raw_telemetry ORDER BY sequence_number DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                return None
            return RawSignalRecord.model_validate_json(row[0])
        except Exception as e:
            logger.error("SQLite error getting latest record: %s", e)
            return None

    def get_history(self, limit: int = 100, offset: int = 0) -> list[RawSignalRecord]:
        try:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM raw_telemetry ORDER BY sequence_number ASC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
            return [RawSignalRecord.model_validate_json(r[0]) for r in rows]
        except Exception as e:
            logger.error("SQLite error getting history: %s", e)
            return []

    def count(self) -> int:
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_telemetry")
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error("SQLite error counting records: %s", e)
            return 0

    def clear(self) -> None:
        try:
            with self._conn:
                self._conn.execute("DELETE FROM raw_telemetry")
        except Exception as e:
            logger.error("SQLite error clearing raw_telemetry table: %s", e)


def create_raw_repository(settings: AppSettings | None = None) -> RawTelemetryRepositoryProtocol:
    """Factory creating configured RawTelemetryRepositoryProtocol instance."""
    cfg = settings or get_settings()
    backend = cfg.persistence.storage_backend.lower()

    if backend == "sqlite":
        return SQLiteRawTelemetryRepository(cfg.persistence.storage_path)
    return InMemoryRawTelemetryRepository(cfg.persistence.max_records)
