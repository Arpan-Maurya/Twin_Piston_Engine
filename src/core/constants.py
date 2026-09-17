"""
Constants — Physical constants with source references.

Every constant must identify its source in the docstring. This is a mandatory
architectural rule: traceability of physical equations and parameters.

All values in SI units unless otherwise noted.
"""

import math

# =============================================================================
# Universal Constants
# =============================================================================

R_UNIVERSAL: float = 8.314462618
"""Universal gas constant [J/(mol·K)].
Source: NIST CODATA 2018, https://physics.nist.gov/cgi-bin/cuu/Value?r
"""

AVOGADRO: float = 6.02214076e23
"""Avogadro's number [1/mol].
Source: NIST CODATA 2018
"""

BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant [J/K].
Source: NIST CODATA 2018
"""

STEFAN_BOLTZMANN: float = 5.670374419e-8
"""Stefan-Boltzmann constant [W/(m²·K⁴)].
Source: NIST CODATA 2018
"""

G_STANDARD: float = 9.80665
"""Standard gravitational acceleration [m/s²].
Source: ISO 80000-3:2006
"""

# =============================================================================
# Atmospheric Constants
# =============================================================================

P_ATM: float = 101325.0
"""Standard atmospheric pressure at sea level [Pa].
Source: ISO 2533:1975, International Standard Atmosphere (ISA)
"""

T_ATM: float = 288.15
"""Standard atmospheric temperature at sea level [K] (15 °C).
Source: ISO 2533:1975, ISA
"""

LAPSE_RATE: float = 0.0065
"""Temperature lapse rate in troposphere [K/m].
Source: ISO 2533:1975, ISA
"""

TROPOPAUSE_ALT: float = 11000.0
"""Tropopause altitude [m].
Source: ISO 2533:1975, ISA
"""

# =============================================================================
# Air Properties
# =============================================================================

GAMMA_AIR: float = 1.4
"""Ratio of specific heats for air (γ = c_p / c_v) at standard conditions.
Source: Heywood, J.B., "Internal Combustion Engine Fundamentals", 2nd ed.,
       McGraw-Hill, 2018, Table 3.3
"""

R_AIR: float = 287.058
"""Specific gas constant for dry air [J/(kg·K)].
R_air = R_universal / M_air, where M_air = 28.97 g/mol.
Source: Heywood (2018), §3.3
"""

CP_AIR: float = 1005.0
"""Specific heat of air at constant pressure [J/(kg·K)].
Source: Heywood (2018), Table 3.3
"""

CV_AIR: float = 718.0
"""Specific heat of air at constant volume [J/(kg·K)].
c_v = c_p / γ = 1005 / 1.4 ≈ 718 J/(kg·K)
Source: Derived from Heywood (2018), Table 3.3
"""

RHO_AIR_STD: float = 1.225
"""Density of air at standard conditions [kg/m³].
ρ = P / (R_air · T) = 101325 / (287.058 × 288.15)
Source: ISO 2533:1975
"""

M_AIR: float = 0.02897
"""Molar mass of dry air [kg/mol].
Source: Heywood (2018), §3.3
"""

# =============================================================================
# Fuel Properties — Avgas 100LL
# =============================================================================

LHV_AVGAS: float = 43.5e6
"""Lower heating value of Avgas 100LL [J/kg].
Source: CRC Handbook, "Aviation Fuel Properties", CRC Report No. 530, 1983
       Typical range: 43.1–43.9 MJ/kg
"""

STOICHIOMETRIC_AFR: float = 14.7
"""Stoichiometric air-fuel ratio for Avgas (by mass).
Source: Heywood (2018), §3.4, Table 3.4
"""

RHO_AVGAS: float = 721.0
"""Density of Avgas 100LL at 15°C [kg/m³].
Source: ASTM D910 specification
"""

FUEL_MOLECULAR_FORMULA: str = "C8H18"
"""Approximate molecular formula for Avgas (modeled as iso-octane).
Source: Heywood (2018), §3.4
"""

M_FUEL: float = 0.11423
"""Molar mass of iso-octane (Avgas model) [kg/mol].
Source: CRC Handbook of Chemistry and Physics
"""

# =============================================================================
# Combustion Constants
# =============================================================================

WIEBE_A: float = 6.908
"""Wiebe function parameter 'a' — gives 99.9% burn at x_b = 1.
a = -ln(1 - 0.999) = 6.908
Source: Heywood (2018), §9.2, Eq. 9.29
"""

WIEBE_M: float = 2.0
"""Wiebe function form factor 'm' — typical for spark-ignition engines.
Source: Heywood (2018), §9.2, typical range 1.5–3.0
"""

# =============================================================================
# Woschni Heat Transfer Constants
# =============================================================================

WOSCHNI_C1: float = 3.26
"""Woschni correlation leading coefficient.
Source: Woschni, G., "A Universally Applicable Equation for the Instantaneous
       Heat Transfer Coefficient in the Internal Combustion Engine",
       SAE Technical Paper 670931, 1967
"""

WOSCHNI_C2_COMPRESSION: float = 2.28
"""Woschni velocity coefficient during compression.
Source: Woschni (1967), SAE 670931
"""

WOSCHNI_C2_COMBUSTION: float = 3.24e-3
"""Woschni velocity coefficient during combustion/expansion.
Source: Woschni (1967), SAE 670931
"""

# =============================================================================
# Chen-Flynn Friction Model Constants (SI engine baseline)
# =============================================================================

CHEN_FLYNN_C0: float = 0.4e5
"""Chen-Flynn constant term [Pa].
Source: Chen, S.K. and Flynn, P.F., "Development of a Single Cylinder
       Compression Ignition Research Engine", SAE 650733, 1965
       (adapted for SI engines — coefficients calibrated to typical
       Rotax 9-series friction data)
"""

CHEN_FLYNN_C1: float = 0.006
"""Chen-Flynn peak pressure coefficient [dimensionless].
Source: Chen & Flynn (1965), SAE 650733
"""

CHEN_FLYNN_C2: float = 600.0
"""Chen-Flynn mean piston speed linear coefficient [Pa·s/m].
Source: Chen & Flynn (1965), SAE 650733
"""

CHEN_FLYNN_C3: float = 50.0
"""Chen-Flynn mean piston speed squared coefficient [Pa·s²/m²].
Source: Chen & Flynn (1965), SAE 650733
"""

# =============================================================================
# Mathematical Constants
# =============================================================================

TWO_PI: float = 2.0 * math.pi
"""2π — frequently used in rotational calculations."""

DEG_TO_RAD: float = math.pi / 180.0
"""Conversion factor: degrees to radians."""

RAD_TO_DEG: float = 180.0 / math.pi
"""Conversion factor: radians to degrees."""
