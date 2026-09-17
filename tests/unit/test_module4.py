"""
Unit & Integration Tests for Original Module 4 — Raw Telemetry Persistence and History.

Validates all Module 4 requirements:
    - Save and retrieve valid RawSignalRecord
    - Multiple-record history & chronological ordering
    - Sequence lookup & timestamp range lookup
    - Latest record query & empty history behavior
    - Missing sequence lookup behavior
    - Duplicate handling contract
    - Persistence failure handling (graceful error capture without crashing pipeline)
    - Raw payload & metadata 100% preservation (no silent modifications)
    - Both InMemory and SQLite repository backends
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.core.config import AppSettings, load_settings
from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_repository import (
    InMemoryRawTelemetryRepository,
    RawTelemetryRepositoryProtocol,
    SQLiteRawTelemetryRepository,
    create_raw_repository,
)
from src.l1_data.raw_signal_record import RawSignalRecord


@pytest.fixture
def make_raw_record():
    """Factory fixture for generating test RawSignalRecords."""
    def _create(seq: int, ts: datetime | None = None) -> RawSignalRecord:
        timestamp = ts or datetime(2026, 9, 17, 12, 0, seq, tzinfo=timezone.utc)
        rec = RawSignalRecord(
            timestamp=timestamp,
            sequence_number=seq,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0 + seq * 10,
            egt_cyl2_hot_uv=30500.0 + seq * 10,
            egt_cyl3_hot_uv=29800.0 + seq * 10,
            egt_cyl4_hot_uv=30100.0 + seq * 10,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0 + seq * 5,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0 + seq * 0.1,
            oil_p_counts=2048 + seq,
            map_counts=2048 + seq,
            adc_vref_counts=4095,
            crank_period_us=15000.0 - seq,
            fuel_pulse_hz=100.0 + seq * 0.5,
            accel_counts_xyz=(10, 15, 980 + seq),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )
        return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})

    return _create


@pytest.fixture(params=["in_memory", "sqlite"])
def repository(request, tmp_path: Path) -> RawTelemetryRepositoryProtocol:
    """Parametrized fixture supplying both InMemory and SQLite repository backends."""
    if request.param == "in_memory":
        return InMemoryRawTelemetryRepository()
    else:
        db_file = tmp_path / f"test_raw_{request.node.name}.db"
        return SQLiteRawTelemetryRepository(db_file)


class TestRawTelemetryPersistence:
    """Comprehensive test suite for Module 4 persistence."""

    def test_save_and_retrieve_record(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        rec = make_raw_record(1)
        assert repository.save(rec) is True

        retrieved = repository.get_by_sequence(1)
        assert retrieved is not None
        assert retrieved.sequence_number == 1
        assert retrieved.egt_cyl1_hot_uv == rec.egt_cyl1_hot_uv
        assert retrieved.integrity_hash == rec.integrity_hash
        assert retrieved.signal_quality.score == rec.signal_quality.score

    def test_raw_payload_preservation_fidelity(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        rec = make_raw_record(42)
        repository.save(rec)

        retrieved = repository.get_by_sequence(42)
        assert retrieved == rec
        assert retrieved.timestamp == rec.timestamp
        assert retrieved.source_type == rec.source_type
        assert retrieved.accel_counts_xyz == rec.accel_counts_xyz

    def test_multiple_record_history_and_ordering(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        for seq in range(1, 11):
            repository.save(make_raw_record(seq))

        assert repository.count() == 10

        history = repository.get_history(limit=5, offset=0)
        assert len(history) == 5
        assert [r.sequence_number for r in history] == [1, 2, 3, 4, 5]

        history_offset = repository.get_history(limit=5, offset=5)
        assert len(history_offset) == 5
        assert [r.sequence_number for r in history_offset] == [6, 7, 8, 9, 10]

    def test_timestamp_range_lookup(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        r1 = make_raw_record(1, t0)
        r2 = make_raw_record(2, t0 + timedelta(seconds=5))
        r3 = make_raw_record(3, t0 + timedelta(seconds=10))

        repository.save(r1)
        repository.save(r2)
        repository.save(r3)

        range_records = repository.get_by_timestamp_range(
            start_time=t0 + timedelta(seconds=2),
            end_time=t0 + timedelta(seconds=8),
        )
        assert len(range_records) == 1
        assert range_records[0].sequence_number == 2

    def test_get_latest_record(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        repository.save(make_raw_record(1))
        repository.save(make_raw_record(5))
        repository.save(make_raw_record(3))

        latest = repository.get_latest()
        assert latest is not None
        assert latest.sequence_number in (3, 5)

    def test_empty_history_behavior(self, repository: RawTelemetryRepositoryProtocol) -> None:
        assert repository.count() == 0
        assert repository.get_latest() is None
        assert repository.get_by_sequence(999) is None
        assert repository.get_history() == []
        assert repository.get_by_timestamp_range(
            datetime.now(timezone.utc), datetime.now(timezone.utc)
        ) == []

    def test_duplicate_sequence_overwrite_contract(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        rec1 = make_raw_record(10)
        repository.save(rec1)

        rec2 = rec1.model_copy(update={"egt_cyl1_hot_uv": 50000.0})
        repository.save(rec2)

        retrieved = repository.get_by_sequence(10)
        assert retrieved is not None
        assert retrieved.egt_cyl1_hot_uv == 50000.0

    def test_clear_repository(self, repository: RawTelemetryRepositoryProtocol, make_raw_record) -> None:
        repository.save(make_raw_record(1))
        repository.save(make_raw_record(2))
        assert repository.count() == 2

        repository.clear()
        assert repository.count() == 0
        assert repository.get_latest() is None

    def test_persistence_failure_handling_gracefully(self, monkeypatch) -> None:
        """Verify repository catches internal storage failures without crashing callers."""
        repo = SQLiteRawTelemetryRepository(":memory:")
        # Close DB connection forcibly to simulate I/O or storage failure
        repo._conn.close()

        rec = RawSignalRecord(
            sequence_number=1,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0,
            egt_cyl2_hot_uv=30000.0,
            egt_cyl3_hot_uv=30000.0,
            egt_cyl4_hot_uv=30000.0,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0,
            oil_p_counts=2000,
            map_counts=2000,
            adc_vref_counts=4095,
            crank_period_us=15000.0,
            fuel_pulse_hz=100.0,
            accel_counts_xyz=(0, 0, 1000),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
        )

        # Failure should return False cleanly without throwing unhandled exception
        result = repo.save(rec)
        assert result is False

    def test_factory_creates_repository_from_config(self) -> None:
        settings = load_settings()
        repo = create_raw_repository(settings)
        assert isinstance(repo, RawTelemetryRepositoryProtocol)
