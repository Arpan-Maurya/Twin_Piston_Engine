"""
API Transport Schemas — Pydantic Request, Response, and Event Models (Module 20).

Contains clean serialization DTOs for REST endpoints and WebSocket events.
Keeps transport concerns decoupled from domain models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import SignalQuality

T = TypeVar("T")


class APIErrorResponse(BaseModel):
    """Standardized API error response envelope."""

    error_code: str = Field(description="Machine-readable error category code")
    message: str = Field(description="Human-readable error description")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context or validation details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheckResponse(BaseModel):
    """REST API service availability health response."""

    status: str = Field(default="ok", description="Service status (ok / degraded)")
    service_name: str = Field(default="Piston Engine Digital Twin API")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    """Overall system operational status response."""

    service_status: str = Field(description="API service health status")
    telemetry_pipeline_status: str = Field(description="L1 Ingestion pipeline status")
    digital_twin_status: str = Field(description="L2 Digital Twin status")
    ml_supervision_status: str = Field(description="L3 ML Supervision status")
    active_connections: int = Field(default=0, description="Active WebSocket clients")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryIngestRequest(BaseModel):
    """Request DTO for raw telemetry ingestion."""

    sequence_number: int = Field(ge=0)
    source_type: Provenance = Field(default=Provenance.SIMULATED)
    egt_cyl1_hot_uv: float
    egt_cyl2_hot_uv: float
    egt_cyl3_hot_uv: float
    egt_cyl4_hot_uv: float
    egt_cold_c: float
    cht_hot_uv: float
    cht_cold_c: float
    oil_rtd_ohms: float
    oil_p_counts: int = Field(ge=0, le=4095)
    map_counts: int = Field(ge=0, le=4095)
    adc_vref_counts: int = Field(ge=0, le=4095)
    crank_period_us: float = Field(gt=0.0)
    fuel_pulse_hz: float = Field(ge=0.0)
    accel_counts_xyz: tuple[int, int, int]
    ambient_temp_c: float
    ambient_press_pa: float
    timestamp: datetime | None = None
    integrity_hash: str | None = None
    signature: str | None = None


class TelemetryIngestResponse(BaseModel):
    """Response DTO for raw telemetry ingestion."""

    accepted: bool
    sequence_number: int
    rejection_reason: str = ""
    security_verified: bool = False
    invalid_channels: list[str] = Field(default_factory=list)
    quality: SignalQuality
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EngineHealthResponse(BaseModel):
    """Engine Health Index & degradation state response DTO."""

    health_index: float
    degradation_state: str
    trend: str
    component_health: dict[str, float]
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime


class DiagnosticSummaryResponse(BaseModel):
    """Summary DTO of all L2 physics diagnostics."""

    rpm: float
    map_pa: float
    egt_mean_k: float
    egt_spread_k: float
    oil_temp_k: float
    oil_pressure_pa: float
    vibration_rms_m_s2: float
    misfire_detected: list[bool]
    overall_status: str
    timestamp: datetime
    provenance: Provenance


class AnomalyResponse(BaseModel):
    """Anomaly detection response DTO."""

    is_anomaly: bool
    anomaly_score: float
    threshold: float
    status: str
    evidence: dict[str, Any]
    quality: float
    provenance: Provenance
    timestamp: datetime


class FaultClassificationResponse(BaseModel):
    """Nine-class fault classification response DTO."""

    predicted_class: str
    class_id: int
    confidence: float | None
    probabilities: dict[str, float] | None
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime


class RULResponse(BaseModel):
    """Remaining Useful Life estimation response DTO."""

    hours_remaining: float
    lower_bound_hours: float | None
    upper_bound_hours: float | None
    unit: str = "hours"
    trend: str
    operating_assumption: str
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime


class MissionRiskResponse(BaseModel):
    """Mission phase and risk assessment response DTO."""

    flight_phase: str
    risk_score: float
    risk_level: str
    risk_trend: str
    health_index: float | None
    rul_hours: float | None
    contributing_fault: str | None
    quality: float
    provenance: Provenance
    timestamp: datetime


class AdvisoryResponse(BaseModel):
    """Decision-support advisory response DTO."""

    advisory_id: str
    category: str
    priority: str
    title: str
    message: str
    subsystem: str
    confidence: float
    provenance: Provenance
    timestamp: datetime
    limitations: str


class ExplanationResponse(BaseModel):
    """Human-readable explanation response DTO."""

    explanation_id: str
    finding: str
    observation: str
    interpretation: str
    evidence: list[dict[str, Any]]
    uncertainty: str
    limitation: str
    provenance: Provenance
    quality: float
    timestamp: datetime


class DiagnosticQueryRequest(BaseModel):
    """Operator diagnostic query request DTO."""

    question_type: str = Field(description="Query category (e.g. HEALTH_STATUS, FAULT_STATUS, WHAT_CHANGED)")
    time_window_s: float | None = Field(default=None, description="Optional time window in seconds")


class DiagnosticQueryResponse(BaseModel):
    """Operator diagnostic query answer DTO."""

    question_type: str
    answer: str
    evidence: list[dict[str, Any]]
    limitations: str
    quality: float
    provenance: Provenance
    timestamp: datetime


class ReplayActionRequest(BaseModel):
    """Request DTO for scenario replay control."""

    scenario_id: str = Field(default="default_scenario")
    position: int | None = Field(default=None, ge=0)
    target_time: datetime | None = None


class ReplayStateResponse(BaseModel):
    """Current state of scenario replay response DTO."""

    scenario_id: str
    replay_position: int
    total_records: int
    status: str
    current_timestamp: datetime | None
    start_time: datetime | None
    end_time: datetime | None
    sequence_number: int
    speed_multiplier: float
    provenance: Provenance


class WhatIfRequest(BaseModel):
    """Request DTO for creating and executing a what-if scenario."""

    baseline_scenario_id: str = Field(default="baseline_01")
    modifications: dict[str, Any] = Field(description="Allowed modification keys (rpm_profile, fault_scenarios, seed, etc.)")
    duration_s: float = Field(default=30.0, gt=0.0, le=600.0)
    dt_s: float = Field(default=1.0, ge=0.01, le=10.0)


class WhatIfResponse(BaseModel):
    """Response DTO for what-if scenario execution."""

    what_if_id: str
    parent_scenario_id: str
    sample_count: int
    seed: int
    provenance: Provenance
    created_at: datetime


class ScenarioCompareRequest(BaseModel):
    """Request DTO for comparing baseline and what-if scenario pipeline outputs."""

    baseline_scenario_id: str
    what_if_id: str


class ScenarioCompareResponse(BaseModel):
    """Response DTO for scenario comparison results."""

    baseline_id: str
    what_if_id: str
    metric_deltas: dict[str, Any]
    state_changes: dict[str, Any]
    timestamp_alignment_info: dict[str, Any]
    ground_truth_validation: dict[str, Any] | None
    provenance: Provenance


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response container."""

    items: list[T]
    total_items: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=1000)
    total_pages: int


class EventEnvelope(BaseModel):
    """Real-time WebSocket event message envelope."""

    event_type: str = Field(description="Event category (e.g. telemetry_update, health_update, fault_update)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = Field(default=0)
    payload: dict[str, Any]
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance = Field(default=Provenance.DERIVED)
