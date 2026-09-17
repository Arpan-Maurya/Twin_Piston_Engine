"""
Rotax 915 iS Engine Parameters — Geometric, thermal, and material constants.

All parameters sourced from:
- Rotax 915 iS Installation Manual (IM), Rev. 0, 2017
- Rotax 915 iS Operator's Manual (OM), Rev. 0, 2017
- Heywood, J.B., "Internal Combustion Engine Fundamentals", 2nd ed., 2018
- Estimated values noted explicitly in docstrings

These are used ONLY by the L1 simulator and must NOT be imported by L2.
L2 gets its engine parameters from the externalized config system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rotax915iSParams:
    """Complete physical parameters for the Rotax 915 iS engine.

    Geometric Parameters:
        Source: Rotax 915 iS Installation Manual, §2.1
        - 4-cylinder horizontally opposed (boxer)
        - Liquid-cooled
        - Turbocharged with intercooler and wastegate
        - Electronic fuel injection (Rotax E-TEC)
        - Dual electronic ignition
    """

    # --- Fundamental geometry ---
    bore_m: float = 0.084
    """Bore diameter [m]. Source: Rotax 915 iS IM §2.1 — 84 mm"""

    stroke_m: float = 0.061
    """Stroke [m]. Source: Rotax 915 iS IM §2.1 — 61 mm"""

    num_cylinders: int = 4
    """Number of cylinders. Source: Rotax 915 iS IM §2.1"""

    compression_ratio: float = 10.5
    """Geometric compression ratio. Source: Rotax 915 iS IM §2.1 — 10.5:1"""

    connecting_rod_m: float = 0.105
    """Connecting rod length [m]. Estimated from typical Rotax 9-series geometry."""

    firing_order: tuple[int, ...] = (1, 3, 2, 4)
    """Firing order. Source: Rotax 915 iS OM §5.2 — boxer configuration."""

    # --- Derived geometry ---
    @property
    def bore_area_m2(self) -> float:
        """Piston crown area [m²]: A = π/4 · B²"""
        return math.pi / 4.0 * self.bore_m ** 2

    @property
    def stroke_volume_m3(self) -> float:
        """Displacement per cylinder [m³]: V_d1 = A · S"""
        return self.bore_area_m2 * self.stroke_m

    @property
    def total_displacement_m3(self) -> float:
        """Total engine displacement [m³]: V_d = n · V_d1"""
        return self.stroke_volume_m3 * self.num_cylinders

    @property
    def clearance_volume_m3(self) -> float:
        """Clearance volume per cylinder [m³]: V_c = V_d1 / (r_c - 1)

        Source: Heywood (2018), §2.2
        """
        return self.stroke_volume_m3 / (self.compression_ratio - 1.0)

    @property
    def crank_radius_m(self) -> float:
        """Crank radius [m]: r = S / 2"""
        return self.stroke_m / 2.0

    @property
    def rod_ratio(self) -> float:
        """Rod ratio λ = r / l (crank radius / connecting rod length).

        Source: Heywood (2018), §2.2
        """
        return self.crank_radius_m / self.connecting_rod_m

    # --- Valve timing ---
    ivc_btdc_deg: float = 50.0
    """Intake valve close angle [deg BTDC of compression].
    Estimated from typical SI engine timing."""

    evo_bbdc_deg: float = 55.0
    """Exhaust valve open angle [deg BBDC of expansion].
    Estimated from typical SI engine timing."""

    # --- Mass properties ---
    piston_mass_kg: float = 0.35
    """Piston mass [kg]. Estimated for Rotax 9-series aluminium pistons."""

    conrod_mass_kg: float = 0.30
    """Connecting rod mass [kg]. Estimated."""

    @property
    def reciprocating_mass_kg(self) -> float:
        """Reciprocating mass [kg]: m_recip ≈ m_piston + m_conrod/3

        Source: Taylor, C.F., "The Internal Combustion Engine in Theory
        and Practice", Vol. 2, 1985, Ch. 8
        """
        return self.piston_mass_kg + self.conrod_mass_kg / 3.0

    # --- Performance limits ---
    rated_power_w: float = 105000.0
    """Rated power [W]. Source: Rotax 915 iS IM §2.1 — 105 kW at 5800 RPM"""

    rated_rpm: float = 5800.0
    """Rated speed [rev/min]. Source: Rotax 915 iS IM §2.1"""

    max_rpm: float = 5800.0
    """Maximum continuous RPM. Source: Rotax 915 iS OM §4.2"""

    idle_rpm: float = 1200.0
    """Idle RPM. Estimated from Rotax 9-series typical idle."""

    # --- Combustion ---
    wiebe_a: float = 6.908
    """Wiebe function parameter 'a'. Source: Heywood (2018), §9.2"""

    wiebe_m: float = 2.0
    """Wiebe function form factor 'm'. Source: Heywood (2018), §9.2"""

    combustion_duration_deg: float = 50.0
    """Total combustion duration [crank degrees].
    Source: Typical SI engine, Heywood (2018), §9.2 — range 40–60°"""

    ignition_timing_btdc_deg: float = 25.0
    """Baseline ignition timing [deg BTDC].
    Source: Typical turbocharged SI engine — conservative due to turbo."""

    # --- Fuel ---
    lhv_j_per_kg: float = 43.5e6
    """Lower heating value of Avgas 100LL [J/kg].
    Source: CRC Handbook, "Aviation Fuel Properties", Report No. 530"""

    stoichiometric_afr: float = 14.7
    """Stoichiometric air-fuel ratio. Source: Heywood (2018), §3.4"""

    target_lambda: float = 0.92
    """Target lambda at full power (slightly rich for cooling).
    Source: Typical turbocharged aero engine practice."""

    # --- Turbocharger ---
    max_boost_pa: float = 145000.0
    """Maximum manifold pressure [Pa]. Source: Rotax 915 iS IM — 1.45 bar"""

    wastegate_target_pa: float = 135000.0
    """Wastegate target pressure [Pa]. Source: Rotax 915 iS OM — 1.35 bar nominal"""

    compressor_efficiency: float = 0.72
    """Compressor isentropic efficiency. Estimated — typical small turbo."""

    turbine_efficiency: float = 0.68
    """Turbine isentropic efficiency. Estimated — typical small turbo."""

    turbo_inertia_kg_m2: float = 5.0e-5
    """Turbo shaft moment of inertia [kg·m²]. Estimated."""

    # --- Cooling ---
    coolant_flow_kg_s: float = 1.0
    """Coolant mass flow rate [kg/s]. Estimated."""

    coolant_cp_j_kg_k: float = 3500.0
    """Coolant specific heat [J/(kg·K)]. 50/50 ethylene glycol."""

    thermostat_open_k: float = 358.15
    """Thermostat opening temperature [K] (85°C)."""

    nominal_cht_k: float = 373.15
    """Nominal CHT [K] (100°C). Source: Rotax 915 iS OM §4.3"""

    max_cht_k: float = 408.15
    """Maximum allowable CHT [K] (135°C). Source: Rotax 915 iS OM §4.3"""

    # --- Lubrication ---
    nominal_oil_pressure_pa: float = 400000.0
    """Nominal oil pressure [Pa] (4 bar). Source: Rotax 915 iS OM §4.4"""

    min_oil_pressure_pa: float = 150000.0
    """Minimum oil pressure [Pa] (1.5 bar). Source: Rotax 915 iS OM §4.4"""

    nominal_oil_temp_k: float = 363.15
    """Nominal oil temperature [K] (90°C). Source: Rotax 915 iS OM §4.4"""

    max_oil_temp_k: float = 403.15
    """Maximum oil temperature [K] (130°C). Source: Rotax 915 iS OM §4.4"""

    # --- Heat transfer — Woschni model constants ---
    woschni_c1: float = 3.26
    """Woschni leading coefficient. Source: Woschni (1967), SAE 670931"""

    wall_temp_k: float = 450.0
    """Average cylinder wall temperature [K]. Estimated for liquid-cooled engine."""

    # --- Friction — Chen-Flynn model ---
    cf_c0_pa: float = 40000.0
    """Chen-Flynn constant term [Pa]. Calibrated to Rotax 9-series."""

    cf_c1: float = 0.006
    """Chen-Flynn peak pressure coefficient. Source: Chen & Flynn (1965)"""

    cf_c2_pa_s_m: float = 600.0
    """Chen-Flynn linear piston speed coefficient [Pa·s/m]."""

    cf_c3_pa_s2_m2: float = 50.0
    """Chen-Flynn quadratic piston speed coefficient [Pa·s²/m²]."""


# Module-level singleton instance
ROTAX_915IS = Rotax915iSParams()
