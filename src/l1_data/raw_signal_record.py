"""
Canonical RawSignalRecord — Acquisition-level telemetry schema.

Contains ONLY raw sensor/acquisition quantities (microvolts, ohms, ADC counts,
microseconds, pulse frequencies), NEVER derived engineering quantities (RPM, Pa, K).

This record preserves physical raw signal fidelity and carries complete
provenance and signal quality metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality, TimestampedRecord


class RawSignalRecord(TimestampedRecord):
    """Canonical raw signal record for hardware acquisition layer.

    All fields are raw acquisition units:
        - Thermocouples: microvolts [uV] and cold-junction temp [°C]
        - RTD: resistance [ohms]
        - Pressure / ADC: raw ADC counts
        - Engine speed: crank period [microseconds]
        - Fuel flow: pulse frequency [Hz]
        - Accelerometer: 3-axis ADC counts
        - Ambient: ambient temp [°C] and pressure [Pa]
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of hardware sampling",
    )
    sequence_number: int = Field(ge=0, description="Monotonically increasing packet counter")
    source_type: Provenance = Field(
        default=Provenance.REAL,
        description="Origin of raw data (REAL / SIMULATED / CSV_REPLAY)",
    )

    # Thermocouple Exhaust Gas Temperature raw signals (microvolts)
    egt_cyl1_hot_uv: float = Field(description="Cylinder 1 EGT thermocouple hot junction [uV]")
    egt_cyl2_hot_uv: float = Field(description="Cylinder 2 EGT thermocouple hot junction [uV]")
    egt_cyl3_hot_uv: float = Field(description="Cylinder 3 EGT thermocouple hot junction [uV]")
    egt_cyl4_hot_uv: float = Field(description="Cylinder 4 EGT thermocouple hot junction [uV]")
    egt_cold_c: float = Field(description="EGT cold junction reference temperature [°C]")

    # Thermocouple Cylinder Head Temperature raw signals (microvolts)
    cht_hot_uv: float = Field(description="CHT thermocouple hot junction [uV]")
    cht_cold_c: float = Field(description="CHT cold junction reference temperature [°C]")

    # RTD Resistance (Oil temperature)
    oil_rtd_ohms: float = Field(description="Oil temperature RTD resistance [ohms]")

    # ADC counts for pressure and reference
    oil_p_counts: int = Field(description="Oil pressure sensor ADC counts")
    map_counts: int = Field(description="Manifold absolute pressure sensor ADC counts")
    adc_vref_counts: int = Field(description="ADC reference voltage channel counts")

    # Crankshaft timing and fuel pulse
    crank_period_us: float = Field(gt=0.0, description="Crankshaft revolution period [us]")
    fuel_pulse_hz: float = Field(ge=0.0, description="Fuel flow turbine pulse frequency [Hz]")

    # Accelerometer raw counts (X, Y, Z)
    accel_counts_xyz: tuple[int, int, int] = Field(description="3-axis accelerometer ADC counts (X, Y, Z)")

    # Ambient conditions
    ambient_temp_c: float = Field(description="Ambient air temperature [°C]")
    ambient_press_pa: float = Field(description="Ambient atmospheric pressure [Pa]")

    # Integrity & Quality metadata
    integrity_hash: str = Field(default="", description="SHA-256 payload integrity signature")
    signal_quality: SignalQuality = Field(default_factory=SignalQuality, description="Signal quality metadata")

    model_config = ConfigDict(frozen=True)

    def compute_integrity_hash(self) -> str:
        """Compute SHA-256 integrity hash of raw payload fields."""
        payload = {
            "seq": self.sequence_number,
            "egt1": self.egt_cyl1_hot_uv,
            "egt2": self.egt_cyl2_hot_uv,
            "egt3": self.egt_cyl3_hot_uv,
            "egt4": self.egt_cyl4_hot_uv,
            "egt_cold": self.egt_cold_c,
            "cht": self.cht_hot_uv,
            "cht_cold": self.cht_cold_c,
            "oil_rtd": self.oil_rtd_ohms,
            "oil_p": self.oil_p_counts,
            "map": self.map_counts,
            "vref": self.adc_vref_counts,
            "crank_us": self.crank_period_us,
            "fuel_hz": self.fuel_pulse_hz,
            "accel": list(self.accel_counts_xyz),
            "amb_t": self.ambient_temp_c,
            "amb_p": self.ambient_press_pa,
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()
