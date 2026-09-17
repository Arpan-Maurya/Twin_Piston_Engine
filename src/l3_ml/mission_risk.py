"""
Mission Phase Identification and Mission Risk Analytics — Original Module 16.

Fifth stage of L3 ML & Supervision Layer.
Provides deterministic mission phase classification and analytical mission risk scoring.

STRICT BOUNDARY CONSTRAINTS:
    - Analytical risk scoring ONLY. Zero autonomous flight control or engine actuation.
    - ML & physics inputs MUST NOT consume RawSignalRecord directly.
    - Preserves exact FlightPhase enum taxonomy (GROUND, TAKEOFF, CLIMB, CRUISE, DESCENT, LANDING).
    - risk_score is a relative analytical index in [0,1], NOT a failure probability.
    - Missing RUL or missing ML models are handled safely without assuming zero risk or perfect safety.
    - Zero maintenance recommendation text or advisory alert text (Module 19 owns advisory).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    DegradationState,
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    DerivedEngineState,
    FaultClassificationResult,
    HealthState,
    MissionState,
    RULState,
    ResidualState,
)
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


class MissionPhaseClassifier:
    """Deterministic, rule-based mission phase classifier."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._config = self._settings.mission

        # State for phase transition tracking and debounce
        self._previous_phase: FlightPhase | str = FlightPhase.GROUND
        self._candidate_phase: FlightPhase | str = FlightPhase.GROUND
        self._candidate_count: int = 0
        self._last_transition_time: datetime | None = None

    def reset_state(self) -> None:
        """Reset internal phase transition state."""
        self._previous_phase = FlightPhase.GROUND
        self._candidate_phase = FlightPhase.GROUND
        self._candidate_count = 0
        self._last_transition_time = None

    def classify_phase(
        self,
        rpm: float | None = None,
        map_pa: float | None = None,
        throttle_pct: float | None = None,
        altitude_m: float | None = None,
        vertical_rate_m_s: float | None = None,
        derived_state: DerivedEngineState | None = None,
        residual_state: ResidualState | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[FlightPhase | str, float, str | None]:
        """Determine current mission phase from engine & flight parameters.

        Returns (phase, phase_quality, phase_transition_description).
        Raises TypeError if caller passes RawSignalRecord directly.
        """
        for arg in [derived_state, residual_state]:
            if isinstance(arg, RawSignalRecord):
                raise TypeError("STRICT BOUNDARY VIOLATION: MissionPhaseClassifier MUST NOT consume RawSignalRecord directly.")

        ts = timestamp or (derived_state.timestamp if derived_state else None) or datetime.now(timezone.utc)

        # Extract values with fallbacks to derived_state / residual_state
        r_rpm = rpm if rpm is not None else (derived_state.rpm.value if (derived_state and derived_state.rpm.valid) else None)
        r_map = map_pa if map_pa is not None else (derived_state.map_pressure_pa.value if (derived_state and derived_state.map_pressure_pa.valid) else None)
        r_throttle = throttle_pct if throttle_pct is not None else (derived_state.operating_point.throttle_pct if derived_state else None)
        r_alt = altitude_m if altitude_m is not None else (derived_state.operating_point.altitude_m if derived_state else None)
        r_vrate = vertical_rate_m_s

        # Validate non-finite inputs
        for val in [r_rpm, r_map, r_throttle, r_alt, r_vrate]:
            if val is not None and (math.isnan(val) or math.isinf(val)):
                return "UNKNOWN", 0.0, None

        # Rule evaluation
        raw_phase: FlightPhase | str = "UNKNOWN"
        quality = 1.0

        if r_rpm is None:
            raw_phase = "UNKNOWN"
            quality = 0.0
        elif r_rpm < self._config.ground_max_rpm and (r_throttle is None or r_throttle < self._config.ground_max_throttle_pct):
            raw_phase = FlightPhase.GROUND
        elif r_rpm >= self._config.takeoff_min_rpm and (r_map is None or r_map >= self._config.takeoff_min_map_pa) and (r_alt is None or r_alt < 500.0):
            raw_phase = FlightPhase.TAKEOFF
        elif r_rpm >= self._config.climb_min_rpm and (r_vrate is not None and r_vrate >= self._config.climb_min_vrate_m_s):
            raw_phase = FlightPhase.CLIMB
        elif (r_alt is not None and r_alt < self._config.landing_max_altitude_m) and (r_vrate is not None and r_vrate < -0.5) and (r_rpm < 3500.0):
            raw_phase = FlightPhase.LANDING
        elif r_rpm < self._config.descent_max_rpm and (r_vrate is not None and r_vrate <= self._config.descent_max_vrate_m_s):
            raw_phase = FlightPhase.DESCENT
        elif 3500.0 <= r_rpm <= 5000.0:
            raw_phase = FlightPhase.CRUISE
        else:
            raw_phase = FlightPhase.CRUISE
            quality = 0.7

        # Apply phase transition debounce
        final_phase, transition_desc = self._apply_debounce(raw_phase, ts)
        return final_phase, quality, transition_desc

    def _apply_debounce(self, raw_phase: FlightPhase | str, ts: datetime) -> tuple[FlightPhase | str, str | None]:
        """Apply debounce filter to prevent rapid phase switching."""
        if self._last_transition_time is None:
            # First classification sample: adopt initial phase directly
            old_p = self._previous_phase
            self._previous_phase = raw_phase
            self._candidate_phase = raw_phase
            self._candidate_count = 1
            self._last_transition_time = ts
            transition_desc = f"Initial phase set to {raw_phase}" if raw_phase != old_p else None
            return raw_phase, transition_desc

        if raw_phase == self._previous_phase:
            self._candidate_phase = raw_phase
            self._candidate_count = 0
            return self._previous_phase, None

        if raw_phase == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = raw_phase
            self._candidate_count = 1

        if self._candidate_count >= self._config.phase_debounce_samples:
            old_p = self._previous_phase
            self._previous_phase = raw_phase
            self._last_transition_time = ts
            transition_desc = f"Transitioned from {old_p} to {raw_phase}"
            return raw_phase, transition_desc

        return self._previous_phase, None


class MissionRiskEngine:
    """Engine for analytical mission risk assessment integrating Modules 13-15."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._config = self._settings.mission

        # State for risk trend supervision
        self._previous_risk_score: float | None = None

    def reset_state(self) -> None:
        """Reset historical risk state."""
        self._previous_risk_score = None

    def evaluate_mission_risk(
        self,
        flight_phase: FlightPhase | str = FlightPhase.GROUND,
        health_state: HealthState | None = None,
        rul_state: RULState | None = None,
        anomaly_result: AnomalyResult | None = None,
        fault_result: FaultClassificationResult | None = None,
        mission_id: str | None = None,
        mission_elapsed_hours: float = 0.0,
        phase_quality: float = 1.0,
        previous_phase: FlightPhase | str | None = None,
        phase_transition: str | None = None,
        timestamp: datetime | None = None,
    ) -> MissionState:
        """Evaluate analytical mission risk score and risk level.

        Raises TypeError if caller passes RawSignalRecord directly.
        """
        for arg in [health_state, rul_state, anomaly_result, fault_result]:
            if isinstance(arg, RawSignalRecord):
                raise TypeError("STRICT BOUNDARY VIOLATION: MissionRiskEngine MUST NOT consume RawSignalRecord directly.")

        ts = (
            timestamp
            or (health_state.timestamp if health_state else None)
            or (rul_state.timestamp if rul_state else None)
            or datetime.now(timezone.utc)
        )

        evidence: dict[str, Any] = {}
        qualities: list[float] = [phase_quality]

        # 1. Health Index contribution
        risk_hi = 0.0
        hi_val: float | None = None
        deg_state: DegradationState | None = None
        if health_state:
            qualities.append(health_state.quality)
            if health_state.health_index.valid and health_state.health_index.value is not None:
                hi_val = health_state.health_index.value
                deg_state = health_state.degradation_state
                risk_hi = 1.0 - hi_val
                evidence["health_index"] = hi_val

                # Degradation state adjustment
                deg_map = {
                    DegradationState.HEALTHY: 0.0,
                    DegradationState.WATCH: 0.20,
                    DegradationState.CAUTION: 0.45,
                    DegradationState.WARNING: 0.70,
                    DegradationState.CRITICAL: 0.95,
                }
                risk_hi = max(risk_hi, deg_map.get(deg_state, 0.5))

        # 2. Anomaly & Fault contribution
        risk_fault = 0.0
        contrib_fault: FaultClass | None = None
        if anomaly_result:
            qualities.append(anomaly_result.quality)
            if anomaly_result.status == InferenceStatus.SUCCESS:
                evidence["anomaly_score"] = anomaly_result.anomaly_score
                if anomaly_result.is_anomaly:
                    risk_fault += min(0.5, anomaly_result.anomaly_score * 0.5)

        if fault_result:
            qualities.append(fault_result.quality)
            if fault_result.status == InferenceStatus.SUCCESS:
                evidence["predicted_class"] = fault_result.class_name
                if fault_result.predicted_class != FaultClass.NOMINAL:
                    contrib_fault = fault_result.predicted_class
                    conf = fault_result.confidence if fault_result.confidence is not None else 1.0
                    risk_fault += min(0.5, 0.4 * conf)

        # 3. RUL contribution
        risk_rul = 0.0
        rul_val: float | None = None
        if rul_state:
            qualities.append(rul_state.quality)
            if rul_state.status == InferenceStatus.SUCCESS:
                rul_val = rul_state.hours_remaining
                evidence["rul_hours"] = rul_val
                if rul_val < 50.0:
                    risk_rul = min(1.0, (50.0 - rul_val) / 50.0)
            elif rul_state.status in (InferenceStatus.MODEL_UNAVAILABLE, InferenceStatus.INSUFFICIENT_HISTORY):
                evidence["rul_status"] = str(rul_state.status)
                risk_rul = 0.10  # Mild uncertainty penalty for missing RUL

        # 4. Phase Risk Weight Multiplier
        phase_str = str(flight_phase.value) if isinstance(flight_phase, FlightPhase) else str(flight_phase)
        phase_mult = self._config.phase_risk_weights.get(phase_str, 1.2)
        evidence["phase_multiplier"] = phase_mult

        # Combine weighted raw risk score
        raw_risk = (0.35 * risk_hi + 0.35 * risk_fault + 0.30 * risk_rul) * phase_mult
        risk_score = max(0.0, min(1.0, raw_risk))

        # Quality aggregation
        overall_quality = sum(qualities) / len(qualities) if qualities else 1.0

        # Risk level determination
        if overall_quality < 0.20:
            risk_level = "UNKNOWN"
            risk_score = 0.0
        elif risk_score >= self._config.high_risk_threshold:
            risk_level = "CRITICAL"
        elif risk_score >= self._config.moderate_risk_threshold:
            risk_level = "HIGH"
        elif risk_score >= self._config.low_risk_threshold:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Risk trend supervision
        trend = "STABLE"
        if self._previous_risk_score is not None and overall_quality >= 0.20:
            d_risk = risk_score - self._previous_risk_score
            if d_risk > 0.05:
                trend = "INCREASING"
            elif d_risk < -0.05:
                trend = "DECREASING"

        if overall_quality >= 0.20:
            self._previous_risk_score = risk_score

        # Convert flight phase enum for schema compatibility
        fp_enum = flight_phase if isinstance(flight_phase, FlightPhase) else FlightPhase.GROUND

        return MissionState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            flight_phase=fp_enum,
            mission_hours=mission_elapsed_hours,
            risk_score=risk_score,
            go_no_go_recommendation="GO" if risk_score < 0.50 else "NO_GO",
            mission_id=mission_id,
            phase_quality=phase_quality,
            previous_phase=previous_phase,
            phase_transition=phase_transition,
            risk_level=risk_level,
            risk_trend=trend,
            health_index=hi_val,
            degradation_state=deg_state,
            rul_hours=rul_val,
            contributing_fault=contrib_fault,
            quality=overall_quality,
            evidence=evidence,
        )
