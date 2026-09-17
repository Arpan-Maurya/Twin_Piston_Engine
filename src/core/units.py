"""
Units — Pure-function unit conversion utilities.

All conversions are stateless and reversible.  Business logic should call
these rather than embedding conversion factors inline.
"""

import math

from src.core.constants import DEG_TO_RAD, G_STANDARD, LAPSE_RATE, P_ATM, RAD_TO_DEG, T_ATM

# =============================================================================
# Temperature
# =============================================================================


def celsius_to_kelvin(t_c: float) -> float:
    """Convert Celsius to Kelvin."""
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    """Convert Kelvin to Celsius."""
    return t_k - 273.15


def fahrenheit_to_kelvin(t_f: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def kelvin_to_fahrenheit(t_k: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (t_k - 273.15) * 9.0 / 5.0 + 32.0


# =============================================================================
# Pressure
# =============================================================================


def bar_to_pa(p_bar: float) -> float:
    """Convert bar to Pascal."""
    return p_bar * 1.0e5


def pa_to_bar(p_pa: float) -> float:
    """Convert Pascal to bar."""
    return p_pa * 1.0e-5


def psi_to_pa(p_psi: float) -> float:
    """Convert psi to Pascal."""
    return p_psi * 6894.757


def pa_to_psi(p_pa: float) -> float:
    """Convert Pascal to psi."""
    return p_pa / 6894.757


def inhg_to_pa(p_inhg: float) -> float:
    """Convert inches of mercury to Pascal."""
    return p_inhg * 3386.389


def pa_to_inhg(p_pa: float) -> float:
    """Convert Pascal to inches of mercury."""
    return p_pa / 3386.389


def atm_to_pa(p_atm: float) -> float:
    """Convert atmospheres to Pascal."""
    return p_atm * 101325.0


def pa_to_atm(p_pa: float) -> float:
    """Convert Pascal to atmospheres."""
    return p_pa / 101325.0


# =============================================================================
# Rotational
# =============================================================================


def rpm_to_rad_s(rpm: float) -> float:
    """Convert revolutions per minute to radians per second."""
    return rpm * math.pi / 30.0


def rad_s_to_rpm(omega: float) -> float:
    """Convert radians per second to revolutions per minute."""
    return omega * 30.0 / math.pi


def rpm_to_hz(rpm: float) -> float:
    """Convert RPM to frequency in Hz."""
    return rpm / 60.0


def hz_to_rpm(freq: float) -> float:
    """Convert frequency in Hz to RPM."""
    return freq * 60.0


# =============================================================================
# Angular
# =============================================================================


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * DEG_TO_RAD


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * RAD_TO_DEG


# =============================================================================
# Length
# =============================================================================


def mm_to_m(mm: float) -> float:
    """Convert millimetres to metres."""
    return mm * 1.0e-3


def m_to_mm(m: float) -> float:
    """Convert metres to millimetres."""
    return m * 1.0e3


def cc_to_m3(cc: float) -> float:
    """Convert cubic centimetres to cubic metres."""
    return cc * 1.0e-6


def m3_to_cc(m3: float) -> float:
    """Convert cubic metres to cubic centimetres."""
    return m3 * 1.0e6


# =============================================================================
# Mass / Flow
# =============================================================================


def kg_per_hr_to_kg_per_s(flow_kg_hr: float) -> float:
    """Convert mass flow from kg/hr to kg/s."""
    return flow_kg_hr / 3600.0


def kg_per_s_to_kg_per_hr(flow_kg_s: float) -> float:
    """Convert mass flow from kg/s to kg/hr."""
    return flow_kg_s * 3600.0


# =============================================================================
# Power / Energy
# =============================================================================


def kw_to_w(p_kw: float) -> float:
    """Convert kilowatts to watts."""
    return p_kw * 1000.0


def w_to_kw(p_w: float) -> float:
    """Convert watts to kilowatts."""
    return p_w / 1000.0


def hp_to_w(p_hp: float) -> float:
    """Convert mechanical horsepower to watts."""
    return p_hp * 745.69987


def w_to_hp(p_w: float) -> float:
    """Convert watts to mechanical horsepower."""
    return p_w / 745.69987


# =============================================================================
# Atmosphere — ISA Model
# =============================================================================


def isa_temperature(altitude_m: float) -> float:
    """ISA temperature at a given altitude [K].

    Valid in the troposphere (0–11 000 m).

    Source: ISO 2533:1975, International Standard Atmosphere.
        T(h) = T₀ − L·h
        where T₀ = 288.15 K, L = 0.0065 K/m

    Args:
        altitude_m: Altitude above sea level [m].

    Returns:
        Temperature [K].
    """
    return T_ATM - LAPSE_RATE * min(altitude_m, 11000.0)


def isa_pressure(altitude_m: float) -> float:
    """ISA pressure at a given altitude [Pa].

    Valid in the troposphere (0–11 000 m).

    Source: ISO 2533:1975, International Standard Atmosphere.
        P(h) = P₀ · (1 − L·h/T₀)^(g/(R·L))
        where g = 9.80665 m/s², R_air = 287.058 J/(kg·K)

    Args:
        altitude_m: Altitude above sea level [m].

    Returns:
        Pressure [Pa].
    """
    h = min(altitude_m, 11000.0)
    exponent = G_STANDARD / (287.058 * LAPSE_RATE)  # ≈ 5.2559
    return P_ATM * (1.0 - LAPSE_RATE * h / T_ATM) ** exponent


def isa_density(altitude_m: float) -> float:
    """ISA air density at a given altitude [kg/m³].

    Uses ideal gas law: ρ = P / (R_air · T)

    Source: Derived from ISO 2533:1975.
    """
    t = isa_temperature(altitude_m)
    p = isa_pressure(altitude_m)
    return p / (287.058 * t)
