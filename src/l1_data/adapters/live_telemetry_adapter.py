"""
Live Telemetry Adapter — Interface & CAN 2.0 parser for live engine hardware.

Parses incoming CAN 2.0 bus frames (or mock transport) into canonical
RawSignalRecord instances tagged with Provenance.REAL.

Zero hardware dependency in downstream layers — works with real hardware interfaces
or isolated MockCANTransport.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone

from src.core.exceptions import AdapterConnectionError
from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.adapters.can_transport import CANFrame, CANTransportProtocol, MockCANTransport
from src.l1_data.raw_signal_record import RawSignalRecord


class LiveTelemetryAdapter(SourceAdapter):
    """Adapter for live engine telemetry via CAN 2.0 bus or mock transport.

    All output is emitted as canonical RawSignalRecords tagged with Provenance.REAL.

    Usage:
        transport = MockCANTransport()
        adapter = LiveTelemetryAdapter(transport=transport)
        async with adapter:
            raw_record = await adapter.read_next()
    """

    def __init__(
        self,
        transport: CANTransportProtocol | None = None,
        protocol: str = "can2.0",
        interface: str = "can0",
        baud_rate: int = 500000,
    ) -> None:
        self._transport = transport or MockCANTransport()
        self._protocol = protocol
        self._interface = interface
        self._baud_rate = baud_rate
        self._connected = False
        self._sequence = 0

        # Accumulated frame state for assembling complete RawSignalRecord
        self._latest_crank_period_us: float = 15000.0
        self._latest_fuel_pulse_hz: float = 100.0
        self._latest_egt1_uv: float = 30000.0
        self._latest_egt2_uv: float = 30000.0
        self._latest_egt3_uv: float = 30000.0
        self._latest_egt4_uv: float = 30000.0
        self._latest_cht_uv: float = 12000.0
        self._latest_oil_rtd_ohms: float = 135.0
        self._latest_map_counts: int = 2048
        self._latest_oil_p_counts: int = 2048

    async def connect(self) -> None:
        """Establish connection to CAN bus interface."""
        try:
            await self._transport.connect()
            self._connected = True
        except Exception as e:
            raise AdapterConnectionError(f"Failed to connect live telemetry transport: {e}") from e

    async def read_next(self) -> RawSignalRecord | None:
        """Read and parse incoming CAN frame into canonical RawSignalRecord.

        Returns:
            RawSignalRecord with Provenance.REAL, or None if no frame is available.
        """
        if not self._connected:
            return None

        frame = await self._transport.read_frame()
        if frame is None:
            return None

        self._parse_can_frame(frame)
        self._sequence += 1

        raw_rec = RawSignalRecord(
            timestamp=frame.timestamp,
            sequence_number=self._sequence,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=self._latest_egt1_uv,
            egt_cyl2_hot_uv=self._latest_egt2_uv,
            egt_cyl3_hot_uv=self._latest_egt3_uv,
            egt_cyl4_hot_uv=self._latest_egt4_uv,
            egt_cold_c=25.0,
            cht_hot_uv=self._latest_cht_uv,
            cht_cold_c=25.0,
            oil_rtd_ohms=self._latest_oil_rtd_ohms,
            oil_p_counts=self._latest_oil_p_counts,
            map_counts=self._latest_map_counts,
            adc_vref_counts=4095,
            crank_period_us=self._latest_crank_period_us,
            fuel_pulse_hz=self._latest_fuel_pulse_hz,
            accel_counts_xyz=(0, 0, 1000),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=1.0),
        )

        return raw_rec.model_copy(update={"integrity_hash": raw_rec.compute_integrity_hash()})

    async def disconnect(self) -> None:
        """Disconnect from CAN bus transport."""
        await self._transport.disconnect()
        self._connected = False

    def _parse_can_frame(self, frame: CANFrame) -> None:
        """Unpack raw binary payload based on CAN ID."""
        if frame.can_id == 0x100 and len(frame.data) >= 6:
            period_int, hz_int = struct.unpack(">IH", frame.data[:6])
            self._latest_crank_period_us = float(period_int)
            self._latest_fuel_pulse_hz = float(hz_int) / 10.0

        elif frame.can_id == 0x101 and len(frame.data) >= 8:
            e1, e2, e3, e4 = struct.unpack(">HHHH", frame.data[:8])
            self._latest_egt1_uv = float(e1 * 10)
            self._latest_egt2_uv = float(e2 * 10)
            self._latest_egt3_uv = float(e3 * 10)
            self._latest_egt4_uv = float(e4 * 10)

        elif frame.can_id == 0x102 and len(frame.data) >= 8:
            c, rtd, m_cnt, o_cnt = struct.unpack(">HHHH", frame.data[:8])
            self._latest_cht_uv = float(c * 10)
            self._latest_oil_rtd_ohms = float(rtd / 10.0)
            self._latest_map_counts = m_cnt
            self._latest_oil_p_counts = o_cnt

    @property
    def source_type(self) -> Provenance:
        return Provenance.REAL

    @property
    def source_id(self) -> str:
        return f"live_telemetry:{self._protocol}:{self._interface}"

    @property
    def is_connected(self) -> bool:
        return self._connected
