"""
L1 Data Adapters — Source adapters for physics simulator, CSV replay, and live telemetry.
"""

from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.adapters.can_transport import (
    CANFrame,
    CANTransportProtocol,
    MockCANTransport,
)
from src.l1_data.adapters.csv_replay_adapter import CSVReplayAdapter
from src.l1_data.adapters.live_telemetry_adapter import LiveTelemetryAdapter
from src.l1_data.adapters.simulator_adapter import (
    PhysicsSimulatorAdapter,
    SimulatorAdapter,
)

__all__ = [
    "CANFrame",
    "CANTransportProtocol",
    "CSVReplayAdapter",
    "LiveTelemetryAdapter",
    "MockCANTransport",
    "PhysicsSimulatorAdapter",
    "SimulatorAdapter",
    "SourceAdapter",
]
