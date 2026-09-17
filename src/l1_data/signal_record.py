"""
NormalizedSignalRecord — The universal telemetry contract.

This is the SOLE artifact that crosses the L1 → L2 boundary.
Every data source adapter must produce instances of this schema.
L2 (Digital Twin) receives only this — never simulator internals,
raw CSV rows, or hardware-specific packets.

38 channels covering all monitored engine parameters for the Rotax 915 iS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.provenance import ChannelValidity, FlightPhase, Provenance


class ChannelValue(BaseModel):
    """A single telemetry channel measurement with provenance and validity.

    Every channel value carries metadata about:
    - provenance: where the value came from
    - valid: whether it should be trusted
    - quality: signal quality score [0.0, 1.0]
    - fault_flag: human-readable reason if invalid

    The system must NEVER silently replace an invalid value.
    """

    value: float
    provenance: Provenance
    valid: bool = True
    validity_reason: ChannelValidity = ChannelValidity.VALID
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    fault_flag: str | None = None

    model_config = ConfigDict(frozen=True)

    def as_invalid(self, reason: ChannelValidity, fault_flag: str) -> ChannelValue:
        """Return a copy marked as invalid with the given reason.

        The original value is preserved (not replaced) — downstream
        consumers can see what the sensor reported, but know not to
        trust it.
        """
        return self.model_copy(
            update={
                "valid": False,
                "validity_reason": reason,
                "quality": 0.0,
                "fault_flag": fault_flag,
            }
        )


class NormalizedSignalRecord(BaseModel):
    """Universal telemetry record — the L1 → L2 contract.

    Contains 38 channels covering all monitored parameters of the
    Rotax 915 iS engine. Each channel is a ChannelValue with provenance.

    This record is IMMUTABLE once created (frozen model). Adapters
    produce it; L2 consumes it; nothing modifies it in transit.

    Channel list (all in SI units):
        rpm, map_pressure, throttle_position,
        egt_cyl_1..4 (K), cht_cyl_1..4 (K),
        oil_temp (K), oil_pressure (Pa), coolant_temp (K),
        fuel_flow (kg/s), fuel_pressure (Pa),
        intake_air_temp (K), ambient_pressure (Pa), ambient_temp (K),
        voltage (V), current (A),
        vibration_x/y/z/rms (m/s²),
        propeller_speed (rev/min), boost_pressure (Pa),
        wastegate_duty (%), lambda_sensor (ratio),
        ignition_timing_cyl_1..4 (deg BTDC),
        altitude (m), flight_phase (enum), engine_hours (h)
    """

    # --- Record metadata ---
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the measurement",
    )
    sequence_number: int = Field(ge=0, description="Monotonically increasing packet counter")
    source_id: str = Field(description="Identifier of the adapter/source that produced this record")
    provenance: Provenance = Field(description="Overall record provenance (REAL/SIMULATED/CSV_REPLAY)")

    # --- Engine speed & load ---
    rpm: ChannelValue = Field(description="Engine speed [rev/min]")
    map_pressure: ChannelValue = Field(description="Manifold absolute pressure [Pa]")
    throttle_position: ChannelValue = Field(description="Throttle position [%]")

    # --- Exhaust gas temperatures (per cylinder) ---
    egt_cyl_1: ChannelValue = Field(description="Exhaust gas temperature, cylinder 1 [K]")
    egt_cyl_2: ChannelValue = Field(description="Exhaust gas temperature, cylinder 2 [K]")
    egt_cyl_3: ChannelValue = Field(description="Exhaust gas temperature, cylinder 3 [K]")
    egt_cyl_4: ChannelValue = Field(description="Exhaust gas temperature, cylinder 4 [K]")

    # --- Cylinder head temperatures (per cylinder) ---
    cht_cyl_1: ChannelValue = Field(description="Cylinder head temperature, cylinder 1 [K]")
    cht_cyl_2: ChannelValue = Field(description="Cylinder head temperature, cylinder 2 [K]")
    cht_cyl_3: ChannelValue = Field(description="Cylinder head temperature, cylinder 3 [K]")
    cht_cyl_4: ChannelValue = Field(description="Cylinder head temperature, cylinder 4 [K]")

    # --- Oil system ---
    oil_temp: ChannelValue = Field(description="Oil temperature [K]")
    oil_pressure: ChannelValue = Field(description="Oil pressure [Pa]")

    # --- Cooling ---
    coolant_temp: ChannelValue = Field(description="Coolant temperature [K]")

    # --- Fuel system ---
    fuel_flow: ChannelValue = Field(description="Fuel mass flow rate [kg/s]")
    fuel_pressure: ChannelValue = Field(description="Fuel pressure [Pa]")

    # --- Intake ---
    intake_air_temp: ChannelValue = Field(description="Intake air temperature (post-compressor) [K]")

    # --- Ambient ---
    ambient_pressure: ChannelValue = Field(description="Ambient atmospheric pressure [Pa]")
    ambient_temp: ChannelValue = Field(description="Ambient (outside) air temperature [K]")

    # --- Electrical ---
    voltage: ChannelValue = Field(description="Electrical system voltage [V]")
    current: ChannelValue = Field(description="Electrical system current draw [A]")

    # --- Vibration ---
    vibration_x: ChannelValue = Field(description="Vibration acceleration, X axis [m/s²]")
    vibration_y: ChannelValue = Field(description="Vibration acceleration, Y axis [m/s²]")
    vibration_z: ChannelValue = Field(description="Vibration acceleration, Z axis [m/s²]")
    vibration_rms: ChannelValue = Field(description="Vibration RMS magnitude [m/s²]")

    # --- Propeller ---
    propeller_speed: ChannelValue = Field(description="Propeller speed [rev/min]")

    # --- Turbocharger ---
    boost_pressure: ChannelValue = Field(description="Boost (compressor outlet) pressure [Pa]")
    wastegate_duty: ChannelValue = Field(description="Wastegate duty cycle [%]")

    # --- Combustion ---
    lambda_sensor: ChannelValue = Field(description="Lambda (O₂) sensor reading [ratio, 1.0 = stoich]")

    # --- Ignition timing (per cylinder) ---
    ignition_timing_cyl_1: ChannelValue = Field(description="Ignition timing, cylinder 1 [deg BTDC]")
    ignition_timing_cyl_2: ChannelValue = Field(description="Ignition timing, cylinder 2 [deg BTDC]")
    ignition_timing_cyl_3: ChannelValue = Field(description="Ignition timing, cylinder 3 [deg BTDC]")
    ignition_timing_cyl_4: ChannelValue = Field(description="Ignition timing, cylinder 4 [deg BTDC]")

    # --- Flight context ---
    altitude: ChannelValue = Field(description="Altitude above sea level [m]")
    flight_phase: FlightPhase = Field(
        default=FlightPhase.GROUND,
        description="Current mission flight phase",
    )
    engine_hours: ChannelValue = Field(description="Cumulative engine operating hours [h]")

    # --- Integrity ---
    integrity_hash: str = Field(
        default="",
        description="SHA-256 hash of the raw payload for integrity verification",
    )

    model_config = ConfigDict(frozen=True)

    # --- Convenience accessors ---

    @property
    def egt_channels(self) -> list[ChannelValue]:
        """All four EGT channels as a list (cylinder order 1-4)."""
        return [self.egt_cyl_1, self.egt_cyl_2, self.egt_cyl_3, self.egt_cyl_4]

    @property
    def cht_channels(self) -> list[ChannelValue]:
        """All four CHT channels as a list (cylinder order 1-4)."""
        return [self.cht_cyl_1, self.cht_cyl_2, self.cht_cyl_3, self.cht_cyl_4]

    @property
    def ignition_timing_channels(self) -> list[ChannelValue]:
        """All four ignition timing channels as a list (cylinder order 1-4)."""
        return [
            self.ignition_timing_cyl_1, self.ignition_timing_cyl_2,
            self.ignition_timing_cyl_3, self.ignition_timing_cyl_4,
        ]

    @property
    def all_channel_values(self) -> dict[str, ChannelValue]:
        """All channels as a name → ChannelValue dict.

        Useful for iteration in validation and derivation pipelines.
        """
        channels: dict[str, ChannelValue] = {}
        for field_name, field_info in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            if isinstance(value, ChannelValue):
                channels[field_name] = value
        return channels

    @property
    def valid_channel_count(self) -> int:
        """Number of channels currently marked as valid."""
        return sum(1 for ch in self.all_channel_values.values() if ch.valid)

    @property
    def invalid_channel_count(self) -> int:
        """Number of channels currently marked as invalid."""
        return sum(1 for ch in self.all_channel_values.values() if not ch.valid)

    @property
    def invalid_channel_names(self) -> list[str]:
        """Names of channels currently marked as invalid."""
        return [name for name, ch in self.all_channel_values.items() if not ch.valid]


def make_channel(
    value: float,
    provenance: Provenance,
    valid: bool = True,
    quality: float = 1.0,
) -> ChannelValue:
    """Convenience factory for creating a ChannelValue."""
    return ChannelValue(
        value=value,
        provenance=provenance,
        valid=valid,
        quality=quality,
    )
