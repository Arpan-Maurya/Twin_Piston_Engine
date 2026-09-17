"""
Lubrication Model — Original Module 8.

Fourth processing stage of L2 Digital Twin.
Derives physics-based lubrication state (viscosity, pressure margins, temperature margins,
operating status, and pressure-temperature interaction) from NormalizedSignalRecord
and optional DerivedEngineState.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (from Module 5) and optional DerivedEngineState (Module 6)
    - Output: LubricationState (canonical schema) and LubricationModelResult
    - Zero simulator internals or ground truth dependencies
    - Zero ML, Advisory, or API dependencies
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, LubricationConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import DerivedEngineState, LubricationState, ProvenanceTaggedValue, make_tagged
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)


def _severity_rank(status: DiagnosticStatus) -> int:
    """Helper for severity comparison."""
    ranks = {
        DiagnosticStatus.INVALID: 0,
        DiagnosticStatus.NORMAL: 1,
        DiagnosticStatus.WARNING: 2,
        DiagnosticStatus.CRITICAL: 3,
    }
    return ranks.get(status, 0)


def _max_status(s1: DiagnosticStatus, s2: DiagnosticStatus) -> DiagnosticStatus:
    """Return the status with higher severity."""
    if s1 == DiagnosticStatus.INVALID and s2 != DiagnosticStatus.INVALID:
        return s2
    if s2 == DiagnosticStatus.INVALID and s1 != DiagnosticStatus.INVALID:
        return s1
    return s1 if _severity_rank(s1) >= _severity_rank(s2) else s2


class LubricationModelResult(BaseModel):
    """Detailed result container for the lubrication physics model."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    oil_temperature_k: float | None
    oil_pressure_pa: float | None
    dynamic_viscosity_pa_s: float | None
    pressure_margin_pa: float | None
    temperature_margin_k: float | None
    expected_pressure_pa: float | None
    dp_dt_pa_s: float | None
    dt_dt_k_s: float | None
    status: DiagnosticStatus
    reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class LubricationModel:
    """Deterministic physics-based lubrication model for Rotax 915 iS."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: LubricationConfig = self._settings.lubrication
        self._prev_timestamp: datetime | None = None
        self._prev_oil_p_pa: float | None = None
        self._prev_oil_t_k: float | None = None

        # Precalculate nominal viscosity at nominal temperature (90°C = 363.15 K)
        nom_t_k = self._cfg.nominal_temp_c + 273.15
        self._nom_viscosity_pa_s = self.compute_viscosity(nom_t_k) or 0.0433

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_oil_p_pa = None
        self._prev_oil_t_k = None

    def compute_viscosity(self, temp_k: float) -> float | None:
        """Compute dynamic oil viscosity using the Vogel equation:
        mu(T) = A * exp(B / (T - C)) [Pa*s].
        """
        if temp_k <= self._cfg.vogel_c:
            return None
        try:
            val = self._cfg.vogel_a * math.exp(self._cfg.vogel_b / (temp_k - self._cfg.vogel_c))
            return val if not (math.isnan(val) or math.isinf(val)) else None
        except (OverflowError, ValueError, ZeroDivisionError):
            return None

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
    ) -> tuple[LubricationState, LubricationModelResult]:
        """Evaluate lubrication model on a NormalizedSignalRecord.

        Returns canonical LubricationState and detailed LubricationModelResult.
        """
        reasons: list[str] = []
        status = DiagnosticStatus.NORMAL

        # Calculate temporal dt
        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        # ---------------------------------------------------------------------
        # 1. Oil Temperature
        # ---------------------------------------------------------------------
        oil_t_ch = record.oil_temp
        oil_t_valid = oil_t_ch.valid and oil_t_ch.value > 0.0

        if oil_t_valid:
            oil_t_k = oil_t_ch.value
            oil_t_c = oil_t_k - 273.15
            oil_temp_tv = make_tagged(oil_t_k, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)

            # Temp margin relative to critical max (130°C = 403.15 K)
            crit_max_k = self._cfg.crit_high_temp_c + 273.15
            temp_margin_k = crit_max_k - oil_t_k
            temp_margin_tv = make_tagged(temp_margin_k, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)

            # Viscosity calculation
            viscosity_pa_s = self.compute_viscosity(oil_t_k)
            if viscosity_pa_s is not None:
                viscosity_tv = make_tagged(viscosity_pa_s, Provenance.DERIVED, valid=True, quality=oil_t_ch.quality)
            else:
                viscosity_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

            # Temp status check
            if oil_t_c >= self._cfg.crit_high_temp_c or oil_t_c <= self._cfg.crit_low_temp_c:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Oil temperature ({oil_t_c:.1f}°C) at critical threshold")
            elif oil_t_c >= self._cfg.warn_high_temp_c or oil_t_c <= self._cfg.warn_low_temp_c:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Oil temperature ({oil_t_c:.1f}°C) at warning threshold")
        else:
            oil_t_k = None
            oil_temp_tv = make_tagged(363.15, Provenance.DERIVED, valid=False, quality=0.0)
            temp_margin_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            viscosity_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            viscosity_pa_s = None

        # ---------------------------------------------------------------------
        # 2. Oil Pressure
        # ---------------------------------------------------------------------
        oil_p_ch = record.oil_pressure
        oil_p_valid = oil_p_ch.valid and oil_p_ch.value >= 0.0

        if oil_p_valid:
            oil_p_pa = oil_p_ch.value
            oil_p_bar = oil_p_pa / 100000.0
            oil_p_tv = make_tagged(oil_p_pa, Provenance.DERIVED, valid=True, quality=oil_p_ch.quality)

            # Pressure margin relative to critical min (1.5 bar = 150,000 Pa)
            crit_min_pa = self._cfg.crit_low_pressure_bar * 100000.0
            press_margin_pa = oil_p_pa - crit_min_pa
            press_margin_tv = make_tagged(press_margin_pa, Provenance.DERIVED, valid=True, quality=oil_p_ch.quality)

            # Pressure status check
            if oil_p_bar <= self._cfg.crit_low_pressure_bar or oil_p_bar >= self._cfg.crit_high_pressure_bar:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Oil pressure ({oil_p_bar:.2f} bar) at critical threshold")
            elif oil_p_bar <= self._cfg.warn_low_pressure_bar or oil_p_bar >= self._cfg.warn_high_pressure_bar:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Oil pressure ({oil_p_bar:.2f} bar) at warning threshold")
        else:
            oil_p_pa = None
            oil_p_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            press_margin_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        if not oil_t_valid and not oil_p_valid:
            status = DiagnosticStatus.INVALID
            reasons.append("Both oil temperature and oil pressure telemetry are invalid")

        # ---------------------------------------------------------------------
        # 3. Pressure-Temperature-RPM Interaction
        # ---------------------------------------------------------------------
        expected_p_pa: float | None = None
        rpm_ch = record.rpm
        if oil_p_valid and oil_t_valid and viscosity_pa_s is not None and rpm_ch.valid and rpm_ch.value > 0:
            rpm_val = rpm_ch.value
            nom_p_pa = self._cfg.nominal_pressure_bar * 100000.0
            # Hydrodynamic pressure correlation: P_exp = P_nom * sqrt(N / N_nom) * (mu / mu_nom)^0.3
            expected_p_pa = nom_p_pa * math.sqrt(rpm_val / 4000.0) * math.pow(viscosity_pa_s / self._nom_viscosity_pa_s, 0.3)

            if oil_p_pa is not None and expected_p_pa > 0:
                p_ratio = oil_p_pa / expected_p_pa
                if p_ratio < 0.60:
                    status = _max_status(status, DiagnosticStatus.WARNING)
                    reasons.append(f"Oil pressure ({oil_p_pa / 1e5:.2f} bar) significantly below expected ({expected_p_pa / 1e5:.2f} bar) for current viscosity and RPM")

        # ---------------------------------------------------------------------
        # 4. Temporal Rates of Change
        # ---------------------------------------------------------------------
        dp_dt: float | None = None
        dt_dt: float | None = None

        if dt_s is not None and 0.0 < dt_s <= 10.0:
            if oil_p_valid and self._prev_oil_p_pa is not None:
                dp_dt = (oil_p_pa - self._prev_oil_p_pa) / dt_s
            if oil_t_valid and self._prev_oil_t_k is not None:
                dt_dt = (oil_t_k - self._prev_oil_t_k) / dt_s

        # Update temporal state
        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        if oil_p_valid:
            self._prev_oil_p_pa = oil_p_pa
        if oil_t_valid:
            self._prev_oil_t_k = oil_t_k

        # Build output objects
        lub_result = LubricationModelResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            oil_temperature_k=oil_t_k,
            oil_pressure_pa=oil_p_pa,
            dynamic_viscosity_pa_s=viscosity_pa_s,
            pressure_margin_pa=oil_p_pa - (self._cfg.crit_low_pressure_bar * 100000.0) if oil_p_valid else None,
            temperature_margin_k=(self._cfg.crit_high_temp_c + 273.15) - oil_t_k if oil_t_valid else None,
            expected_pressure_pa=expected_p_pa,
            dp_dt_pa_s=dp_dt,
            dt_dt_k_s=dt_dt,
            status=status,
            reasons=reasons,
        )

        lub_state = LubricationState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            oil_temperature_k=oil_temp_tv,
            oil_pressure_pa=oil_p_tv,
            dynamic_viscosity_pa_s=viscosity_tv,
            pressure_margin_pa=press_margin_tv,
            temperature_margin_k=temp_margin_tv,
            status=status,
        )

        return lub_state, lub_result


def evaluate_lubrication_model(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    settings: AppSettings | None = None,
) -> tuple[LubricationState, LubricationModelResult]:
    """Convenience function for evaluating Module 8 Lubrication Model."""
    model = LubricationModel(settings)
    return model.evaluate(record, derived_state)
