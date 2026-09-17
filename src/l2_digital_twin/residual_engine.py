"""
Healthy Expectation Models and Residual Engine — Original Module 11.

Seventh processing stage of L2 Digital Twin.
Generates deterministic physical healthy expectations based on operating point (RPM, MAP, ambient context)
and computes multi-signal physical residuals:
    Residual = Observed - Expected Healthy
    Normalized Residual = (Observed - Expected Healthy) / Scale

STRICT BOUNDARY CONSTRAINTS:
    - Input: Modules 5–10 outputs (NormalizedSignalRecord, DerivedEngineState, DiagnosticState, etc.)
    - Output: ResidualState (canonical domain schema) & HealthyExpectationResult
    - Zero ML models, zero ML inference, zero ML training
    - Zero simulator internals or ground truth dependencies
    - Zero diagnostic fault labels ("bearing failure", "misfire", etc.)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, HealthyBaselineConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import (
    DerivedEngineState,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    make_tagged,
)
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)

# Standard ISA Sea Level Constants
STD_PRESSURE_PA = 101325.0
STD_TEMP_K = 288.15


class QuantityResidual(BaseModel):
    """Detailed residual container for a single physical quantity."""

    name: str
    observed: float | None
    expected: float | None
    raw_residual: float | None
    absolute_residual: float | None
    normalized_residual: float | None
    valid: bool
    quality: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class HealthyExpectationResult(BaseModel):
    """Container for expected healthy state and computed residuals."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    operating_point: OperatingPoint
    expected_values: dict[str, float]
    residuals: dict[str, QuantityResidual]

    model_config = ConfigDict(frozen=True)


class HealthyExpectationModel:
    """Deterministic healthy baseline model based on operating context."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: HealthyBaselineConfig = self._settings.healthy_baseline

    def expected_egt_k(self, op: OperatingPoint) -> float:
        """Expected healthy EGT based on MAP, RPM, and ambient temperature."""
        delta_p = op.map_pressure_pa - STD_PRESSURE_PA
        delta_rpm = max(0.0, op.rpm - 1000.0)
        temp_effect = (op.ambient_temp_k - STD_TEMP_K) * 0.5
        return self._cfg.base_egt_k + (self._cfg.k_egt_map_pa * delta_p) + (self._cfg.k_egt_rpm * delta_rpm) + temp_effect

    def expected_oil_pressure_pa(self, op: OperatingPoint, oil_temp_k: float = 363.15) -> float:
        """Expected healthy oil pressure based on RPM and oil temperature."""
        nom_p = 400000.0  # 4.0 bar
        rpm_factor = math.sqrt(max(0.1, op.rpm) / 4000.0)
        c = 140.0
        visc_ratio = 1.0
        if oil_temp_k > c:
            num = math.exp(1200.0 / (oil_temp_k - c))
            den = math.exp(1200.0 / (363.15 - c))
            visc_ratio = math.pow(num / den, 0.3) if den > 0 else 1.0

        return nom_p * rpm_factor * visc_ratio

    def expected_oil_temp_k(self, op: OperatingPoint) -> float:
        """Expected healthy oil temperature based on load and ambient temperature."""
        load_factor = (op.map_pressure_pa / STD_PRESSURE_PA) * (op.rpm / 5800.0)
        return op.ambient_temp_k + (self._cfg.base_oil_temp_rise_k * load_factor)

    def expected_vibration_rms_m_s2(self, op: OperatingPoint) -> float:
        """Expected healthy vibration RMS based on engine speed."""
        speed_ratio = op.rpm / 5800.0
        return self._cfg.base_vibration_rms_m_s2 + (self._cfg.k_vib_rpm * (speed_ratio ** 2))

    def expected_brake_power_kw(self, op: OperatingPoint) -> float:
        """Expected healthy brake power based on MAP and RPM."""
        rated_p_kw = self._settings.engine.rated_power_kw
        return rated_p_kw * (op.map_pressure_pa / 140000.0) * (op.rpm / 5800.0)


class ResidualEngine:
    """Reusable residual calculation engine for physical actuals vs expected healthy baselines."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: HealthyBaselineConfig = self._settings.healthy_baseline
        self.expectation_model = HealthyExpectationModel(self._settings)

    def compute_residual(
        self,
        name: str,
        observed: float | None,
        observed_valid: bool,
        expected: float | None,
        scale: float,
        quality: float = 1.0,
    ) -> QuantityResidual:
        """Compute raw, absolute, and normalized residual for a single quantity."""
        valid = observed_valid and (observed is not None) and (expected is not None) and not (math.isnan(observed) or math.isnan(expected))
        if not valid or scale <= 0.0:
            return QuantityResidual(
                name=name,
                observed=observed,
                expected=expected,
                raw_residual=None,
                absolute_residual=None,
                normalized_residual=None,
                valid=False,
                quality=0.0,
            )

        raw_res = observed - expected
        abs_res = abs(raw_res)
        norm_res = raw_res / scale

        return QuantityResidual(
            name=name,
            observed=observed,
            expected=expected,
            raw_residual=raw_res,
            absolute_residual=abs_res,
            normalized_residual=norm_res,
            valid=True,
            quality=quality,
        )

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
    ) -> tuple[ResidualState, HealthyExpectationResult]:
        """Evaluate healthy expectations and generate residuals from telemetry."""
        rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value >= 0) else 0.0
        map_val = record.map_pressure.value if (record.map_pressure.valid and record.map_pressure.value > 0) else STD_PRESSURE_PA
        amb_p_val = record.ambient_pressure.value if (record.ambient_pressure.valid and record.ambient_pressure.value > 0) else STD_PRESSURE_PA
        amb_t_val = record.ambient_temp.value if (record.ambient_temp.valid and record.ambient_temp.value > 0) else STD_TEMP_K
        throttle_val = record.throttle_position.value if record.throttle_position.valid else 50.0

        altitude_m = max(0.0, 44330.0 * (1.0 - math.pow(amb_p_val / STD_PRESSURE_PA, 0.1903)))

        op = OperatingPoint(
            rpm=rpm_val,
            map_pressure_pa=map_val,
            altitude_m=altitude_m,
            ambient_temp_k=amb_t_val,
            ambient_pressure_pa=amb_p_val,
            throttle_pct=throttle_val,
        )

        exp_egt = self.expectation_model.expected_egt_k(op)
        exp_oil_p = self.expectation_model.expected_oil_pressure_pa(op, record.oil_temp.value if record.oil_temp.valid else 363.15)
        exp_oil_t = self.expectation_model.expected_oil_temp_k(op)
        exp_vib = self.expectation_model.expected_vibration_rms_m_s2(op)
        exp_power = self.expectation_model.expected_brake_power_kw(op)

        expected_dict = {
            "egt_cyl1": exp_egt,
            "egt_cyl2": exp_egt,
            "egt_cyl3": exp_egt,
            "egt_cyl4": exp_egt,
            "oil_pressure": exp_oil_p,
            "oil_temp": exp_oil_t,
            "vibration_rms": exp_vib,
            "brake_power_kw": exp_power,
        }

        res_dict: dict[str, QuantityResidual] = {}
        schema_residuals: dict[str, ProvenanceTaggedValue[float]] = {}

        egt_channels = [
            ("egt_cyl1", record.egt_cyl_1),
            ("egt_cyl2", record.egt_cyl_2),
            ("egt_cyl3", record.egt_cyl_3),
            ("egt_cyl4", record.egt_cyl_4),
        ]

        for name, ch in egt_channels:
            q_res = self.compute_residual(
                name=name,
                observed=ch.value if ch.valid else None,
                observed_valid=ch.valid,
                expected=exp_egt,
                scale=self._cfg.scale_egt_k,
                quality=ch.quality,
            )
            res_dict[name] = q_res
            if q_res.valid and q_res.raw_residual is not None:
                schema_residuals[name] = make_tagged(q_res.raw_residual, Provenance.DERIVED, valid=True, quality=q_res.quality)
            else:
                schema_residuals[name] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        oil_p_res = self.compute_residual(
            name="oil_pressure",
            observed=record.oil_pressure.value if record.oil_pressure.valid else None,
            observed_valid=record.oil_pressure.valid,
            expected=exp_oil_p,
            scale=self._cfg.scale_oil_pressure_pa,
            quality=record.oil_pressure.quality,
        )
        res_dict["oil_pressure"] = oil_p_res
        if oil_p_res.valid and oil_p_res.raw_residual is not None:
            schema_residuals["oil_pressure"] = make_tagged(oil_p_res.raw_residual, Provenance.DERIVED, valid=True, quality=oil_p_res.quality)
        else:
            schema_residuals["oil_pressure"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        oil_t_res = self.compute_residual(
            name="oil_temp",
            observed=record.oil_temp.value if record.oil_temp.valid else None,
            observed_valid=record.oil_temp.valid,
            expected=exp_oil_t,
            scale=self._cfg.scale_oil_temp_k,
            quality=record.oil_temp.quality,
        )
        res_dict["oil_temp"] = oil_t_res
        if oil_t_res.valid and oil_t_res.raw_residual is not None:
            schema_residuals["oil_temp"] = make_tagged(oil_t_res.raw_residual, Provenance.DERIVED, valid=True, quality=oil_t_res.quality)
        else:
            schema_residuals["oil_temp"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        vib_res = self.compute_residual(
            name="vibration_rms",
            observed=record.vibration_rms.value if record.vibration_rms.valid else None,
            observed_valid=record.vibration_rms.valid,
            expected=exp_vib,
            scale=self._cfg.scale_vibration_m_s2,
            quality=record.vibration_rms.quality,
        )
        res_dict["vibration_rms"] = vib_res
        if vib_res.valid and vib_res.raw_residual is not None:
            schema_residuals["vibration_rms"] = make_tagged(vib_res.raw_residual, Provenance.DERIVED, valid=True, quality=vib_res.quality)
        else:
            schema_residuals["vibration_rms"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        if derived_state is not None and derived_state.brake_power_kw.valid:
            obs_power = derived_state.brake_power_kw.value
            p_res = self.compute_residual(
                name="brake_power_kw",
                observed=obs_power,
                observed_valid=True,
                expected=exp_power,
                scale=self._cfg.scale_power_kw,
                quality=derived_state.brake_power_kw.quality,
            )
            res_dict["brake_power_kw"] = p_res
            if p_res.valid and p_res.raw_residual is not None:
                schema_residuals["brake_power_kw"] = make_tagged(p_res.raw_residual, Provenance.DERIVED, valid=True, quality=p_res.quality)
            else:
                schema_residuals["brake_power_kw"] = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        exp_result = HealthyExpectationResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            operating_point=op,
            expected_values=expected_dict,
            residuals=res_dict,
        )

        residual_state = ResidualState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            operating_point=op,
            residuals=schema_residuals,
        )

        return residual_state, exp_result


def evaluate_residual_engine(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    settings: AppSettings | None = None,
) -> tuple[ResidualState, HealthyExpectationResult]:
    """Convenience function for Module 11 residual engine evaluation."""
    engine = ResidualEngine(settings)
    return engine.evaluate(record, derived_state)
