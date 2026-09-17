"""
Physics Simulator Adapter — Wraps the physics simulator as an L1 data source adapter.

Converts the RotaxEngineSimulator ground truth state through the sensor-forward
path into canonical RawSignalRecord instances tagged with Provenance.SIMULATED.
"""

from __future__ import annotations

from src.core.provenance import FlightPhase, Provenance
from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.sensor_forward import normalized_to_raw
from src.l1_data.simulator.engine_model import EngineOperatingPoint, RotaxEngineSimulator
from src.l1_data.simulator.fault_injection import FaultScenario


class PhysicsSimulatorAdapter(SourceAdapter):
    """Source adapter wrapping the Rotax 915 iS simulator.

    Usage:
        async with PhysicsSimulatorAdapter() as adapter:
            raw_record = await adapter.read_next()
    """

    def __init__(
        self,
        rpm: float = 4000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 1000.0,
        flight_phase: FlightPhase = FlightPhase.CRUISE,
        fault_scenario: FaultScenario | None = None,
        sample_rate_hz: float = 1.0,
    ) -> None:
        self._simulator = RotaxEngineSimulator()
        self._operating_point = EngineOperatingPoint(
            rpm=rpm,
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
            flight_phase=flight_phase,
        )
        self._fault_scenario = fault_scenario
        self._sample_rate_hz = sample_rate_hz
        self._connected = False
        self._engine_hours = 0.0

    async def connect(self) -> None:
        """Initialize the simulator."""
        if self._fault_scenario:
            self._simulator.configure_fault(self._fault_scenario)
        self._connected = True

    async def read_next(self) -> RawSignalRecord | None:
        """Run one simulation step and convert output to RawSignalRecord.

        Returns:
            RawSignalRecord with Provenance.SIMULATED.
        """
        if not self._connected:
            return None

        self._operating_point.engine_hours = self._engine_hours

        # Generate normalized telemetry from simulator
        norm_record = self._simulator.step(self._operating_point)

        # Advance engine hours
        self._engine_hours += 1.0 / (self._sample_rate_hz * 3600.0)

        # Pass through sensor forward model to produce canonical RawSignalRecord
        raw_record = normalized_to_raw(
            record=norm_record,
            sequence_number=norm_record.sequence_number,
            source_type=Provenance.SIMULATED,
        )

        return raw_record

    async def disconnect(self) -> None:
        """Clean up simulator resources."""
        self._connected = False

    @property
    def source_type(self) -> Provenance:
        return Provenance.SIMULATED

    @property
    def source_id(self) -> str:
        return "rotax_915is_simulator"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_operating_point(
        self,
        rpm: float | None = None,
        throttle_pct: float | None = None,
        altitude_m: float | None = None,
        flight_phase: FlightPhase | None = None,
    ) -> None:
        """Update the operating point for subsequent steps."""
        if rpm is not None:
            self._operating_point.rpm = rpm
        if throttle_pct is not None:
            self._operating_point.throttle_pct = throttle_pct
        if altitude_m is not None:
            self._operating_point.altitude_m = altitude_m
        if flight_phase is not None:
            self._operating_point.flight_phase = flight_phase


# Backward-compatible alias
SimulatorAdapter = PhysicsSimulatorAdapter
