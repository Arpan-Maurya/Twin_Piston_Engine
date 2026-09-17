"""
Schemas — Base Pydantic models used across all layers.

Provides generic wrappers for provenance-tagged values, timestamped records,
and operating point definitions. All domain schemas inherit from these.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.core.provenance import (
    AlertLevel,
    ChannelValidity,
    DegradationState,
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    ModelLifecycleStatus,
    Provenance,
)

T = TypeVar("T")


class ProvenanceTaggedValue(BaseModel, Generic[T]):
    """A value paired with its provenance, validity, and quality metadata.

    Every numeric output in the system must be wrapped in this container
    so that downstream consumers always know:
      1. Where the value came from (provenance)
      2. Whether it should be trusted (valid / quality)
      3. Why it might be untrustworthy (fault_flag)
    """

    value: T
    provenance: Provenance
    valid: bool = True
    validity_reason: ChannelValidity = ChannelValidity.VALID
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    fault_flag: str | None = None

    model_config = ConfigDict(frozen=True)

    def as_invalid(self, reason: ChannelValidity, fault_flag: str) -> ProvenanceTaggedValue[T]:
        """Return a copy marked invalid with the given reason."""
        return self.model_copy(
            update={
                "valid": False,
                "validity_reason": reason,
                "quality": 0.0,
                "fault_flag": fault_flag,
            }
        )


class TimestampedRecord(BaseModel):
    """Base class for all time-indexed records."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)


class OperatingPoint(BaseModel):
    """Engine operating point — the independent variables that define the
    current engine state. Used as the lookup key for baseline expectation
    models in L2.

    All values in SI units.
    """

    rpm: float = Field(ge=0, le=6500, description="Engine speed [rev/min]")
    map_pressure_pa: float = Field(ge=20000, le=200000, description="Manifold absolute pressure [Pa]")
    altitude_m: float = Field(ge=0, le=15000, description="Altitude [m]")
    ambient_temp_k: float = Field(ge=210, le=330, description="Ambient temperature [K]")
    ambient_pressure_pa: float = Field(ge=30000, le=110000, description="Ambient pressure [Pa]")
    throttle_pct: float = Field(ge=0, le=100, description="Throttle position [%]")

    model_config = ConfigDict(frozen=True)


class ChannelRange(BaseModel):
    """Valid range for a telemetry channel, used in packet integrity checks."""

    min: float
    max: float

    def contains(self, value: float) -> bool:
        """Check if value falls within [min, max]."""
        return self.min <= value <= self.max


class ModelMetadata(BaseModel):
    """Metadata for a serialized ML model."""

    name: str
    version: str
    trained_at: datetime
    input_shape: list[int]
    output_shape: list[int]
    accuracy_metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""

    model_config = ConfigDict(frozen=True)


class SignalQuality(BaseModel):
    """Signal quality metadata for telemetric measurements."""

    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall quality score [0..1]")
    valid: bool = Field(default=True, description="Whether overall packet signals are valid")
    invalid_channels: list[str] = Field(default_factory=list, description="List of channels marked invalid")
    invalid_reasons: dict[str, str] = Field(default_factory=dict, description="Channel to invalidity reason map")
    noise_level: float = Field(default=0.0, ge=0.0, description="Estimated signal noise level")

    model_config = ConfigDict(frozen=True)


class DerivedEngineState(TimestampedRecord):
    """L2 Physics digital twin derived engineering parameters."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    bmep_pa: ProvenanceTaggedValue[float]
    imep_pa: ProvenanceTaggedValue[float | None]
    eta_thermal: ProvenanceTaggedValue[float]
    eta_volumetric: ProvenanceTaggedValue[float]
    afr: ProvenanceTaggedValue[float]
    brake_torque_nm: ProvenanceTaggedValue[float]
    brake_power_kw: ProvenanceTaggedValue[float]
    fmep_pa: ProvenanceTaggedValue[float]
    eta_mechanical: ProvenanceTaggedValue[float]
    mean_piston_speed_m_s: ProvenanceTaggedValue[float]

    model_config = ConfigDict(frozen=True)


class DiagnosticState(TimestampedRecord):
    """Subsystem & per-cylinder diagnostic status."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    per_cylinder_egt_dev_k: list[float] = Field(default_factory=list)
    per_cylinder_cht_dev_k: list[float] = Field(default_factory=list)
    misfire_detected: list[bool] = Field(default_factory=lambda: [False, False, False, False])
    vibration_rms_m_s2: float = 0.0
    lubrication_pressure_ok: bool = True
    cooling_temp_ok: bool = True

    # Extended Module 7 EGT fields
    per_cylinder_egt_k: list[float | None] = Field(default_factory=list)
    per_cylinder_egt_status: list[DiagnosticStatus] = Field(default_factory=list)
    egt_mean_k: float | None = None
    egt_spread_k: float | None = None
    max_egt_cylinder: int | None = None
    min_egt_cylinder: int | None = None
    egt_rate_of_change_k_s: list[float | None] = Field(default_factory=list)
    egt_overall_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)

    model_config = ConfigDict(frozen=True)


class LubricationState(TimestampedRecord):
    """L2 Physics digital twin derived lubrication system state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    oil_temperature_k: ProvenanceTaggedValue[float]
    oil_pressure_pa: ProvenanceTaggedValue[float]
    dynamic_viscosity_pa_s: ProvenanceTaggedValue[float]
    pressure_margin_pa: ProvenanceTaggedValue[float]
    temperature_margin_k: ProvenanceTaggedValue[float]
    status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)

    model_config = ConfigDict(frozen=True)


class VibrationState(TimestampedRecord):
    """L2 Physics digital twin derived vibration feature state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    rms_x_m_s2: ProvenanceTaggedValue[float]
    rms_y_m_s2: ProvenanceTaggedValue[float]
    rms_z_m_s2: ProvenanceTaggedValue[float]
    overall_rms_m_s2: ProvenanceTaggedValue[float]
    peak_m_s2: ProvenanceTaggedValue[float]
    crest_factor: ProvenanceTaggedValue[float]
    dominant_freq_hz: ProvenanceTaggedValue[float]
    dominant_amplitude_m_s2: ProvenanceTaggedValue[float]
    dominant_order: ProvenanceTaggedValue[float | None]

    model_config = ConfigDict(frozen=True)


class CombustionStabilityState(TimestampedRecord):
    """L2 Physics digital twin derived combustion stability & misfire diagnostic state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    misfire_detected: list[bool] = Field(default_factory=lambda: [False, False, False, False])
    misfire_status: list[DiagnosticStatus] = Field(default_factory=lambda: [DiagnosticStatus.NORMAL]*4)
    evidence_score: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    overall_combustion_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)

    model_config = ConfigDict(frozen=True)


class ResidualState(TimestampedRecord):
    """Physical actuals vs baseline expected residuals."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    operating_point: OperatingPoint
    residuals: dict[str, ProvenanceTaggedValue[float]] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class HealthState(TimestampedRecord):
    """L3 Supervision aggregate engine health metrics and degradation supervision."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    health_index: ProvenanceTaggedValue[float]
    degradation_state: DegradationState = Field(default=DegradationState.HEALTHY)
    component_health: dict[str, float] = Field(default_factory=dict)
    health_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)
    trend: str = Field(default="STABLE")
    health_rate: float = Field(default=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    contributing_fault: FaultClass | None = Field(default=None)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: str = Field(default="1.0.0")

    model_config = ConfigDict(frozen=True)


class RULState(TimestampedRecord):
    """Remaining Useful Life estimation with uncertainty boundaries."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    hours_remaining: float = Field(default=0.0, ge=0.0)
    lower_bound_hours: float | None = Field(default=None)
    upper_bound_hours: float | None = Field(default=None)
    confidence: float | None = Field(default=None)
    unit: str = Field(default="hours")
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_available: bool = Field(default=False)
    uncertainty_representation: dict[str, Any] | None = Field(default=None)
    degradation_state: DegradationState = Field(default=DegradationState.HEALTHY)
    trend: str = Field(default="STABLE")
    operating_assumption: str = Field(default="constant_operating_profile")
    model_name: str | None = Field(default=None)
    model_version: str = Field(default="1.0.0")
    feature_schema_version: str = Field(default="1.0.0")
    evidence: dict[str, Any] = Field(default_factory=dict)
    is_ml: bool = Field(default=False)

    model_config = ConfigDict(frozen=True)


class Alert(TimestampedRecord):
    """L4 Advisory alert notification."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    alert_level: AlertLevel = Field(default=AlertLevel.NOMINAL)
    fault_class: FaultClass = Field(default=FaultClass.NOMINAL)
    summary: str
    message: str
    recommended_action: str

    model_config = ConfigDict(frozen=True)


class AnomalyResult(TimestampedRecord):
    """Result of anomaly detection evaluation."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    is_anomaly: bool = Field(default=False)
    anomaly_score: float = Field(default=0.0, ge=0.0)
    threshold: float = Field(default=0.5, ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    model_metadata: ModelMetadata | None = Field(default=None)
    is_ml: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)


class FaultClassificationResult(TimestampedRecord):
    """Result of nine-class fault classification evaluation."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    predicted_class: FaultClass = Field(default=FaultClass.NOMINAL)
    class_id: int = Field(default=0, ge=0, le=8)
    class_name: str = Field(default="NOMINAL")
    confidence: float | None = Field(default=None)
    probabilities: dict[str, float] | None = Field(default=None)
    feature_schema_version: str = Field(default="1.0.0")
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] | None = Field(default=None)
    model_metadata: ModelMetadata | None = Field(default=None)
    is_ml: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)


class MissionState(TimestampedRecord):
    """Mission phase and analytical risk assessment."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    flight_phase: FlightPhase = Field(default=FlightPhase.GROUND)
    mission_hours: float = Field(default=0.0, ge=0.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    go_no_go_recommendation: str = "GO"
    mission_id: str | None = Field(default=None)
    phase_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    previous_phase: FlightPhase | str | None = Field(default=None)
    phase_transition: str | None = Field(default=None)
    risk_level: str = Field(default="LOW")
    risk_trend: str = Field(default="STABLE")
    health_index: float | None = Field(default=None)
    degradation_state: DegradationState | None = Field(default=None)
    rul_hours: float | None = Field(default=None)
    contributing_fault: FaultClass | None = Field(default=None)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


def make_tagged(
    value: Any,
    provenance: Provenance,
    valid: bool = True,
    quality: float = 1.0,
) -> ProvenanceTaggedValue[Any]:
    """Convenience factory for creating provenance-tagged values."""
    return ProvenanceTaggedValue(
        value=value,
        provenance=provenance,
        valid=valid,
        quality=quality,
    )

