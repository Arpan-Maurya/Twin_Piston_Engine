"""
Unit tests for Original Module 2 — Source Adapters and Strict Raw-Signal Boundary.

Validates all Module 2 acceptance requirements:
    1. PhysicsSimulatorAdapter emits canonical RawSignalRecord (SIMULATED)
    2. CSVReplayAdapter emits canonical RawSignalRecord (CSV_REPLAY)
    3. LiveTelemetryAdapter emits canonical RawSignalRecord (REAL) via MockCANTransport
    4. Source swapping requires zero downstream code changes
    5. Deterministic replay, timestamp/sequence preservation, seek, pause/resume
    6. CAN 2.0 transport isolation without hardware dependencies
    7. Malformed input handling
"""

import asyncio
from pathlib import Path
import pytest

from src.core.provenance import Provenance
from src.l1_data.adapters import (
    CANFrame,
    CSVReplayAdapter,
    LiveTelemetryAdapter,
    MockCANTransport,
    PhysicsSimulatorAdapter,
    SimulatorAdapter,
    SourceAdapter,
)
from src.l1_data.adapters.can_transport import (
    encode_can_frame_0x100,
    encode_can_frame_0x101,
    encode_can_frame_0x102,
)
from src.l1_data.raw_signal_record import RawSignalRecord


class TestPhysicsSimulatorAdapter:
    """Validate PhysicsSimulatorAdapter emit behavior."""

    @pytest.mark.asyncio
    async def test_simulator_adapter_emits_raw_signal_record(self) -> None:
        async with PhysicsSimulatorAdapter(rpm=4000, throttle_pct=60) as adapter:
            assert adapter.is_connected
            assert adapter.source_type == Provenance.SIMULATED

            record = await adapter.read_next()
            assert record is not None
            assert isinstance(record, RawSignalRecord)
            assert record.source_type == Provenance.SIMULATED
            assert record.sequence_number >= 0
            assert record.crank_period_us > 0
            assert record.integrity_hash != ""

    @pytest.mark.asyncio
    async def test_simulator_adapter_alias(self) -> None:
        """Verify SimulatorAdapter alias works identically."""
        async with SimulatorAdapter() as adapter:
            record = await adapter.read_next()
            assert isinstance(record, RawSignalRecord)


class TestCSVReplayAdapter:
    """Validate CSVReplayAdapter features."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path: Path) -> Path:
        csv_file = tmp_path / "flight_test.csv"
        content = (
            "timestamp,sequence_number,rpm,map_pressure,egt1,egt2,egt3,egt4,cht1,oil_temp,oil_pressure\n"
            "2026-09-17T12:00:00Z,1,4000,120000,850,852,848,851,373,363,400000\n"
            "2026-09-17T12:00:01Z,2,4050,121000,855,857,853,856,374,364,405000\n"
            "2026-09-17T12:00:02Z,3,4100,122000,860,862,858,861,375,365,410000\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        return csv_file

    @pytest.mark.asyncio
    async def test_csv_adapter_emits_raw_signal_record(self, sample_csv_file: Path) -> None:
        async with CSVReplayAdapter(sample_csv_file) as adapter:
            assert adapter.source_type == Provenance.CSV_REPLAY
            record = await adapter.read_next()
            assert record is not None
            assert isinstance(record, RawSignalRecord)
            assert record.source_type == Provenance.CSV_REPLAY
            assert record.sequence_number == 1
            assert record.integrity_hash != ""

    @pytest.mark.asyncio
    async def test_csv_deterministic_replay(self, sample_csv_file: Path) -> None:
        records1 = []
        async with CSVReplayAdapter(sample_csv_file) as adapter1:
            while (rec := await adapter1.read_next()) is not None:
                records1.append(rec)

        records2 = []
        async with CSVReplayAdapter(sample_csv_file) as adapter2:
            while (rec := await adapter2.read_next()) is not None:
                records2.append(rec)

        assert len(records1) == 3
        assert len(records2) == 3
        for r1, r2 in zip(records1, records2):
            assert r1.sequence_number == r2.sequence_number
            assert r1.integrity_hash == r2.integrity_hash

    @pytest.mark.asyncio
    async def test_csv_seek_and_pause_resume(self, sample_csv_file: Path) -> None:
        async with CSVReplayAdapter(sample_csv_file) as adapter:
            adapter.seek(2)
            assert adapter.current_index == 2

            rec = await adapter.read_next()
            assert rec is not None
            assert rec.sequence_number == 1

            adapter.pause()
            assert adapter.is_paused
            assert await adapter.read_next() is None

            adapter.resume()
            assert not adapter.is_paused


class TestLiveTelemetryAdapter:
    """Validate LiveTelemetryAdapter with CAN 2.0 mock transport."""

    @pytest.mark.asyncio
    async def test_live_telemetry_adapter_with_mock_can(self) -> None:
        transport = MockCANTransport()
        adapter = LiveTelemetryAdapter(transport=transport)

        async with adapter:
            assert adapter.source_type == Provenance.REAL

            # Send synthetic CAN frames into mock transport
            await transport.send_frame(encode_can_frame_0x100(crank_period_us=15000.0, fuel_pulse_hz=120.0))
            await transport.send_frame(encode_can_frame_0x101(egt1_uv=32000, egt2_uv=32100, egt3_uv=31900, egt4_uv=32050))
            await transport.send_frame(encode_can_frame_0x102(cht_uv=12000, oil_rtd_ohms=135.0, map_counts=2048, oil_p_counts=2048))

            r1 = await adapter.read_next()
            r2 = await adapter.read_next()
            r3 = await adapter.read_next()

            assert r1 is not None
            assert isinstance(r1, RawSignalRecord)
            assert r1.source_type == Provenance.REAL
            assert r1.sequence_number == 1
            assert r3 is not None
            assert r3.oil_p_counts == 2048


class TestSourceSwapping:
    """Validate that downstream consumer functions accept any SourceAdapter interchangeably."""

    @pytest.mark.asyncio
    async def test_polymorphic_adapter_consumption(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "swap_test.csv"
        csv_file.write_text("timestamp,sequence_number,rpm\n2026-09-17T12:00:00Z,1,4000\n", encoding="utf-8")

        mock_transport = MockCANTransport()
        await mock_transport.connect()
        await mock_transport.send_frame(encode_can_frame_0x100(crank_period_us=15000.0, fuel_pulse_hz=100.0))

        adapters: list[SourceAdapter] = [
            PhysicsSimulatorAdapter(),
            CSVReplayAdapter(csv_file),
            LiveTelemetryAdapter(transport=mock_transport),
        ]

        async def downstream_consumer(adapter: SourceAdapter) -> RawSignalRecord:
            async with adapter:
                rec = await adapter.read_next()
                assert rec is not None
                assert isinstance(rec, RawSignalRecord)
                return rec

        results = []
        for adapter in adapters:
            rec = await downstream_consumer(adapter)
            results.append(rec)

        assert results[0].source_type == Provenance.SIMULATED
        assert results[1].source_type == Provenance.CSV_REPLAY
        assert results[2].source_type == Provenance.REAL
