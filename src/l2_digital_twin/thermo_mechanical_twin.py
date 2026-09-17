"""
Thermodynamic and Mechanical Digital Twin — Original Module 6.

Second processing stage of L2 Digital Twin.
Derives thermodynamic and mechanical engine parameters from engineering-unit
NormalizedSignalRecord produced by Module 5.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (from Module 5)
    - Output: DerivedEngineState (canonical domain schema)
    - Zero simulator internals or ground truth dependencies
    - Zero ML, Advisory, or API dependencies
    - Physical derivations are independent of simulator generator equations
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import DerivedEngineState, ProvenanceTaggedValue, make_tagged
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)

# Physical Constants
AIR_GAS_CONSTANT_J_KG_K = 287.058  # Specific gas constant for dry air
STANDARD_SEA_LEVEL_PRESSURE_PA = 101325.0
STANDARD_SEA_LEVEL_TEMP_K = 288.15


class EngineGeometry:
    """Calculates and encapsulates Rotax 915 iS engine geometry."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        cfg = (settings or get_settings()).engine
        self.bore_m: float = cfg.bore_mm / 1000.0
        self.stroke_m: float = cfg.stroke_mm / 1000.0
        self.displacement_m3: float = cfg.displacement_cc * 1e-6
        self.num_cylinders: int = cfg.num_cylinders
        self.compression_ratio: float = cfg.compression_ratio
        self.conrod_length_m: float = cfg.connecting_rod_mm / 1000.0
        self.rated_power_kw: float = cfg.rated_power_kw
        self.rated_rpm: int = cfg.rated_rpm
        self.lhv_j_kg: float = cfg.lhv_mj_per_kg * 1e6
        self.stoichiometric_afr: float = cfg.stoichiometric_afr

        # Derived geometry
        self.cylinder_displacement_m3: float = self.displacement_m3 / self.num_cylinders
        self.cylinder_area_m2: float = (math.pi / 4.0) * (self.bore_m ** 2)
        self.crank_radius_m: float = self.stroke_m / 2.0
        self.conrod_ratio: float = self.crank_radius_m / self.conrod_length_m

        # Clearance volume per cylinder: V_c = V_cyl / (r_c - 1)
        self.clearance_volume_m3: float = self.cylinder_displacement_m3 / (self.compression_ratio - 1.0)

    def mean_piston_speed(self, rpm: float) -> float:
        """Mean piston speed S_p = 2 * stroke * (RPM / 60) [m/s]."""
        if rpm < 0:
            return 0.0
        return 2.0 * self.stroke_m * (rpm / 60.0)

    def max_piston_speed(self, rpm: float) -> float:
        """Peak piston speed V_p,max = (pi/2) * S_p * (1 + lambda_crank) [m/s]."""
        s_p = self.mean_piston_speed(rpm)
        return (math.pi / 2.0) * s_p * (1.0 + self.conrod_ratio)


class ThermodynamicResults(BaseModel):
    """Detailed thermodynamic twin intermediate results."""

    boost_pressure_pa: ProvenanceTaggedValue[float]
    pressure_ratio: ProvenanceTaggedValue[float]
    air_density_kg_m3: ProvenanceTaggedValue[float]
    air_mass_flow_kg_s: ProvenanceTaggedValue[float]
    fuel_energy_rate_w: ProvenanceTaggedValue[float]
    afr: ProvenanceTaggedValue[float]
    eta_volumetric: ProvenanceTaggedValue[float]

    model_config = ConfigDict(frozen=True)


class MechanicalResults(BaseModel):
    """Detailed mechanical twin intermediate results."""

    crank_angular_velocity_rad_s: ProvenanceTaggedValue[float]
    crank_angular_acceleration_rad_s2: ProvenanceTaggedValue[float]
    mean_piston_speed_m_s: ProvenanceTaggedValue[float]
    max_piston_speed_m_s: ProvenanceTaggedValue[float]
    brake_torque_nm: ProvenanceTaggedValue[float]
    brake_power_kw: ProvenanceTaggedValue[float]
    fmep_pa: ProvenanceTaggedValue[float]

    model_config = ConfigDict(frozen=True)


class ThermodynamicTwin:
    """Thermodynamic Digital Twin calculations."""

    def __init__(self, geometry: EngineGeometry) -> None:
        self.geom = geometry

    def compute(self, record: NormalizedSignalRecord) -> ThermodynamicResults:
        """Derive thermodynamic parameters from normalized telemetry."""
        # 1. Boost Pressure & Pressure Ratio
        p_map = record.map_pressure
        p_amb = record.ambient_pressure

        if p_map.valid and p_amb.valid and p_amb.value > 0:
            boost = p_map.value - p_amb.value
            pr = p_map.value / p_amb.value
            boost_tv = make_tagged(boost, Provenance.DERIVED, valid=True, quality=min(p_map.quality, p_amb.quality))
            pr_tv = make_tagged(pr, Provenance.DERIVED, valid=True, quality=min(p_map.quality, p_amb.quality))
        else:
            boost_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            pr_tv = make_tagged(1.0, Provenance.DERIVED, valid=False, quality=0.0)

        # 2. Air Density
        t_iat = record.intake_air_temp if record.intake_air_temp.valid else record.ambient_temp
        if p_map.valid and t_iat.valid and t_iat.value > 0:
            rho_air = p_map.value / (AIR_GAS_CONSTANT_J_KG_K * t_iat.value)
            rho_tv = make_tagged(rho_air, Provenance.DERIVED, valid=True, quality=min(p_map.quality, t_iat.quality))
        else:
            rho_tv = make_tagged(1.225, Provenance.DERIVED, valid=False, quality=0.0)

        # 3. Air-Fuel Ratio (AFR)
        lam = record.lambda_sensor
        if lam.valid and lam.value > 0:
            afr_val = lam.value * self.geom.stoichiometric_afr
            afr_tv = make_tagged(afr_val, Provenance.DERIVED, valid=True, quality=lam.quality)
        else:
            afr_tv = make_tagged(self.geom.stoichiometric_afr, Provenance.DERIVED, valid=False, quality=0.0)

        # 4. Fuel Energy Rate
        fuel = record.fuel_flow
        if fuel.valid and fuel.value >= 0:
            q_in = fuel.value * self.geom.lhv_j_kg
            q_in_tv = make_tagged(q_in, Provenance.DERIVED, valid=True, quality=fuel.quality)
        else:
            q_in_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # 5. Air Mass Flow & Volumetric Efficiency
        rpm = record.rpm
        if fuel.valid and fuel.value >= 0 and afr_tv.valid:
            m_dot_air = fuel.value * afr_tv.value
            m_dot_air_tv = make_tagged(m_dot_air, Provenance.DERIVED, valid=True, quality=min(fuel.quality, afr_tv.quality))
        elif rho_tv.valid and rpm.valid and rpm.value > 0:
            # Speed-density estimate if fuel flow is invalid
            vol_disp_rate = self.geom.displacement_m3 * (rpm.value / 120.0)
            m_dot_air_est = rho_tv.value * vol_disp_rate * 0.85
            m_dot_air_tv = make_tagged(m_dot_air_est, Provenance.DERIVED, valid=True, quality=0.7)
        else:
            m_dot_air_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # Volumetric Efficiency
        if m_dot_air_tv.valid and rho_tv.valid and rho_tv.value > 0 and rpm.valid and rpm.value > 0:
            theoretical_m_dot_air = rho_tv.value * self.geom.displacement_m3 * (rpm.value / 120.0)
            if theoretical_m_dot_air > 0:
                eta_v = m_dot_air_tv.value / theoretical_m_dot_air
                valid_eta = 0.0 <= eta_v <= 1.5
                eta_v_tv = make_tagged(
                    eta_v if valid_eta else 0.0,
                    Provenance.DERIVED,
                    valid=valid_eta,
                    quality=m_dot_air_tv.quality if valid_eta else 0.0,
                )
            else:
                eta_v_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
        else:
            eta_v_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        return ThermodynamicResults(
            boost_pressure_pa=boost_tv,
            pressure_ratio=pr_tv,
            air_density_kg_m3=rho_tv,
            air_mass_flow_kg_s=m_dot_air_tv,
            fuel_energy_rate_w=q_in_tv,
            afr=afr_tv,
            eta_volumetric=eta_v_tv,
        )


class MechanicalTwin:
    """Mechanical Digital Twin calculations."""

    def __init__(self, geometry: EngineGeometry) -> None:
        self.geom = geometry

    def compute(
        self,
        record: NormalizedSignalRecord,
        prev_omega: float | None = None,
        dt_s: float | None = None,
    ) -> MechanicalResults:
        """Derive mechanical parameters from normalized telemetry."""
        rpm = record.rpm
        if rpm.valid and rpm.value >= 0:
            omega = (2.0 * math.pi * rpm.value) / 60.0
            omega_tv = make_tagged(omega, Provenance.DERIVED, valid=True, quality=rpm.quality)
            s_p = self.geom.mean_piston_speed(rpm.value)
            s_p_tv = make_tagged(s_p, Provenance.DERIVED, valid=True, quality=rpm.quality)
            v_p_max = self.geom.max_piston_speed(rpm.value)
            v_p_max_tv = make_tagged(v_p_max, Provenance.DERIVED, valid=True, quality=rpm.quality)
        else:
            omega_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            s_p_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            v_p_max_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # Angular acceleration alpha = d(omega)/dt
        if omega_tv.valid and prev_omega is not None and dt_s is not None and 0.0 < dt_s <= 10.0:
            alpha = (omega_tv.value - prev_omega) / dt_s
            alpha_tv = make_tagged(alpha, Provenance.DERIVED, valid=True, quality=rpm.quality)
        else:
            alpha_tv = make_tagged(0.0, Provenance.DERIVED, valid=True, quality=1.0)

        # FMEP via Chen-Flynn model: FMEP = 40000 + 4000 * S_p + 200 * S_p^2 [Pa]
        if s_p_tv.valid:
            fmep = 40000.0 + 4000.0 * s_p_tv.value + 200.0 * (s_p_tv.value ** 2)
            fmep_tv = make_tagged(fmep, Provenance.DERIVED, valid=True, quality=s_p_tv.quality)
        else:
            fmep_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        # Torque & Power derivations
        fuel = record.fuel_flow
        lam = record.lambda_sensor
        p_map = record.map_pressure

        bmep = 0.0
        power_kw = 0.0
        torque_nm = 0.0
        power_valid = False
        power_quality = 0.0

        if rpm.valid and rpm.value > 0:
            if fuel.valid and fuel.value > 0:
                afr_val = (lam.value * self.geom.stoichiometric_afr) if (lam.valid and lam.value > 0) else self.geom.stoichiometric_afr
                q_in = fuel.value * self.geom.lhv_j_kg
                p_i = q_in * 0.42  # Indicated thermal efficiency ~42%
                imep = (120.0 * p_i) / (self.geom.displacement_m3 * rpm.value)
                bmep = imep - fmep_tv.value if fmep_tv.valid else imep * 0.85
                power_w = (bmep * self.geom.displacement_m3 * rpm.value) / 120.0
                power_kw = power_w / 1000.0
                torque_nm = (bmep * self.geom.displacement_m3) / (4.0 * math.pi)
                power_valid = True
                power_quality = min(rpm.quality, fuel.quality)
            elif p_map.valid and p_map.value > 0:
                # Speed-density estimate
                t_iat = record.intake_air_temp.value if (record.intake_air_temp.valid and record.intake_air_temp.value > 0) else 298.15
                rho_intake = p_map.value / (AIR_GAS_CONSTANT_J_KG_K * t_iat)
                m_dot_air_est = rho_intake * self.geom.displacement_m3 * (rpm.value / 120.0) * 0.85
                m_dot_fuel_est = m_dot_air_est / self.geom.stoichiometric_afr
                q_in = m_dot_fuel_est * self.geom.lhv_j_kg
                p_i = q_in * 0.42
                imep = (120.0 * p_i) / (self.geom.displacement_m3 * rpm.value)
                bmep = imep - fmep_tv.value if fmep_tv.valid else imep * 0.85
                power_w = (bmep * self.geom.displacement_m3 * rpm.value) / 120.0
                power_kw = power_w / 1000.0
                torque_nm = (bmep * self.geom.displacement_m3) / (4.0 * math.pi)
                power_valid = True
                power_quality = 0.7  # quality-limited estimate

        power_tv = make_tagged(power_kw if power_valid else 0.0, Provenance.DERIVED, valid=power_valid, quality=power_quality)
        torque_tv = make_tagged(torque_nm if power_valid else 0.0, Provenance.DERIVED, valid=power_valid, quality=power_quality)

        return MechanicalResults(
            crank_angular_velocity_rad_s=omega_tv,
            crank_angular_acceleration_rad_s2=alpha_tv,
            mean_piston_speed_m_s=s_p_tv,
            max_piston_speed_m_s=v_p_max_tv,
            brake_torque_nm=torque_tv,
            brake_power_kw=power_tv,
            fmep_pa=fmep_tv,
        )


class ThermodynamicMechanicalTwin:
    """Integrated L2 Digital Twin for Original Module 6."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.geometry = EngineGeometry(self._settings)
        self.thermo_twin = ThermodynamicTwin(self.geometry)
        self.mech_twin = MechanicalTwin(self.geometry)
        self._prev_timestamp: datetime | None = None
        self._prev_omega: float | None = None

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_omega = None

    def evaluate(self, record: NormalizedSignalRecord) -> DerivedEngineState:
        """Evaluate digital twin from normalized telemetry to produce DerivedEngineState."""
        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        thermo_res = self.thermo_twin.compute(record)
        mech_res = self.mech_twin.compute(record, prev_omega=self._prev_omega, dt_s=dt_s)

        # Update temporal state
        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        if mech_res.crank_angular_velocity_rad_s.valid:
            self._prev_omega = mech_res.crank_angular_velocity_rad_s.value

        # Calculate BMEP, IMEP, Thermal Efficiency, Mechanical Efficiency
        rpm = record.rpm
        fmep_tv = mech_res.fmep_pa
        power_tv = mech_res.brake_power_kw
        q_in_tv = thermo_res.fuel_energy_rate_w

        if rpm.valid and rpm.value > 0 and power_tv.valid and power_tv.value > 0:
            bmep_val = (120.0 * (power_tv.value * 1000.0)) / (self.geometry.displacement_m3 * rpm.value)
            bmep_tv = make_tagged(bmep_val, Provenance.DERIVED, valid=True, quality=power_tv.quality)
            imep_val = bmep_val + fmep_tv.value if fmep_tv.valid else bmep_val / 0.85
            imep_tv = make_tagged(imep_val, Provenance.DERIVED, valid=True, quality=power_tv.quality)

            if q_in_tv.valid and q_in_tv.value > 0:
                eta_th_val = (power_tv.value * 1000.0) / q_in_tv.value
                valid_eta_th = 0.0 <= eta_th_val <= 1.0
                eta_th_tv = make_tagged(eta_th_val if valid_eta_th else 0.0, Provenance.DERIVED, valid=valid_eta_th, quality=power_tv.quality if valid_eta_th else 0.0)
            else:
                eta_th_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

            if imep_val > 0:
                eta_mech_val = bmep_val / imep_val
                valid_eta_m = 0.0 <= eta_mech_val <= 1.0
                eta_mech_tv = make_tagged(eta_mech_val if valid_eta_m else 0.0, Provenance.DERIVED, valid=valid_eta_m, quality=power_tv.quality if valid_eta_m else 0.0)
            else:
                eta_mech_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
        else:
            bmep_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            imep_tv = make_tagged(None, Provenance.DERIVED, valid=False, quality=0.0)
            eta_th_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)
            eta_mech_tv = make_tagged(0.0, Provenance.DERIVED, valid=False, quality=0.0)

        return DerivedEngineState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            bmep_pa=bmep_tv,
            imep_pa=imep_tv,
            eta_thermal=eta_th_tv,
            eta_volumetric=thermo_res.eta_volumetric,
            afr=thermo_res.afr,
            brake_torque_nm=mech_res.brake_torque_nm,
            brake_power_kw=mech_res.brake_power_kw,
            fmep_pa=fmep_tv,
            eta_mechanical=eta_mech_tv,
            mean_piston_speed_m_s=mech_res.mean_piston_speed_m_s,
        )


def evaluate_digital_twin(
    record: NormalizedSignalRecord,
    settings: AppSettings | None = None,
) -> DerivedEngineState:
    """Convenience function to evaluate digital twin on a NormalizedSignalRecord."""
    twin = ThermodynamicMechanicalTwin(settings)
    return twin.evaluate(record)
