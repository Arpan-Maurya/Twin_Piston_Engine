"""
CAN 2.0 Transport Abstraction & Mock Transport.

Isolates CAN bus protocol parsing from physical hardware interfaces.
Provides an in-memory MockCANTransport for unit testing and offline replay
without requiring socketcan or CAN hardware drivers.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CANFrame:
    """Standard CAN 2.0 Data Frame (8 bytes payload max)."""

    can_id: int                          # 11-bit standard or 29-bit extended arbitration ID
    data: bytes                          # Up to 8 bytes payload
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if len(self.data) > 8:
            raise ValueError("CAN 2.0 payload exceeds maximum length of 8 bytes")


@runtime_checkable
class CANTransportProtocol(Protocol):
    """Protocol for CAN bus network interface abstraction."""

    async def connect(self) -> None:
        """Connect to CAN interface."""
        ...

    async def read_frame(self) -> CANFrame | None:
        """Read next CAN frame from bus."""
        ...

    async def send_frame(self, frame: CANFrame) -> None:
        """Send a CAN frame over the bus."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from CAN bus."""
        ...


class MockCANTransport:
    """In-memory Mock CAN Transport for testing without CAN hardware."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[CANFrame] = asyncio.Queue()
        self._connected: bool = False

    async def connect(self) -> None:
        self._connected = True

    async def read_frame(self) -> CANFrame | None:
        if not self._connected or self._queue.empty():
            return None
        return await self._queue.get()

    async def send_frame(self, frame: CANFrame) -> None:
        if not self._connected:
            raise RuntimeError("MockCANTransport is not connected")
        await self._queue.put(frame)

    async def disconnect(self) -> None:
        self._connected = False
        while not self._queue.empty():
            self._queue.get_nowait()

    @property
    def is_connected(self) -> bool:
        return self._connected


# Standard CAN 2.0 Signal Frame Encoders / Decoders for Rotax 915 iS ECU
# CAN ID 0x100: Crank Period (us, 32-bit uint) + Fuel Pulse (Hz, 16-bit uint) + Reserved (16-bit)
# CAN ID 0x101: EGT Cyl 1-4 (microvolts / 10, 4 x 16-bit uint)
# CAN ID 0x102: CHT (uV / 10, 16-bit), Oil RTD (ohms * 10, 16-bit), MAP counts (16-bit), Oil P counts (16-bit)

def encode_can_frame_0x100(crank_period_us: float, fuel_pulse_hz: float) -> CANFrame:
    """Encode speed & fuel parameters into CAN ID 0x100."""
    period_int = int(min(max(0, crank_period_us), 4294967295))
    hz_int = int(min(max(0, fuel_pulse_hz * 10.0), 65535))
    data = struct.pack(">IH", period_int, hz_int) + b"\x00\x00"
    return CANFrame(can_id=0x100, data=data)


def encode_can_frame_0x101(egt1_uv: float, egt2_uv: float, egt3_uv: float, egt4_uv: float) -> CANFrame:
    """Encode EGT thermocouple microvolts into CAN ID 0x101."""
    e1 = int(min(max(0, egt1_uv / 10.0), 65535))
    e2 = int(min(max(0, egt2_uv / 10.0), 65535))
    e3 = int(min(max(0, egt3_uv / 10.0), 65535))
    e4 = int(min(max(0, egt4_uv / 10.0), 65535))
    data = struct.pack(">HHHH", e1, e2, e3, e4)
    return CANFrame(can_id=0x101, data=data)


def encode_can_frame_0x102(cht_uv: float, oil_rtd_ohms: float, map_counts: int, oil_p_counts: int) -> CANFrame:
    """Encode CHT, Oil RTD, MAP & Oil pressure counts into CAN ID 0x102."""
    c = int(min(max(0, cht_uv / 10.0), 65535))
    rtd = int(min(max(0, oil_rtd_ohms * 10.0), 65535))
    m_cnt = int(min(max(0, map_counts), 65535))
    o_cnt = int(min(max(0, oil_p_counts), 65535))
    data = struct.pack(">HHHH", c, rtd, m_cnt, o_cnt)
    return CANFrame(can_id=0x102, data=data)
