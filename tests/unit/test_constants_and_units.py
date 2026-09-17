"""
Unit tests for physical constants and unit conversion functions.
"""

from __future__ import annotations

import math

import pytest

from src.core import constants
from src.core.units import (
    bar_to_pa,
    celsius_to_kelvin,
    cc_to_m3,
    deg_to_rad,
    hp_to_w,
    isa_density,
    isa_pressure,
    isa_temperature,
    kelvin_to_celsius,
    kw_to_w,
    mm_to_m,
    pa_to_bar,
    rad_to_deg,
    rpm_to_rad_s,
    rad_s_to_rpm,
    w_to_hp,
    w_to_kw,
)


class TestPhysicalConstants:
    """Verify physical constants against known values."""

    def test_r_universal(self) -> None:
        """R = 8.314... J/(mol·K) — NIST CODATA 2018."""
        assert abs(constants.R_UNIVERSAL - 8.314462618) < 1e-6

    def test_p_atm(self) -> None:
        """Standard atmosphere = 101325 Pa."""
        assert constants.P_ATM == 101325.0

    def test_t_atm(self) -> None:
        """ISA sea level temperature = 288.15 K (15 °C)."""
        assert constants.T_ATM == 288.15

    def test_gamma_air(self) -> None:
        """γ_air = 1.4 for ideal diatomic gas."""
        assert constants.GAMMA_AIR == 1.4

    def test_r_air(self) -> None:
        """R_air = R_universal / M_air ≈ 287 J/(kg·K)."""
        computed = constants.R_UNIVERSAL / constants.M_AIR
        assert abs(constants.R_AIR - computed) < 0.5

    def test_lhv_avgas(self) -> None:
        """LHV of Avgas 100LL ≈ 43.5 MJ/kg."""
        assert constants.LHV_AVGAS == 43.5e6

    def test_stoichiometric_afr(self) -> None:
        """Stoichiometric AFR for gasoline/avgas ≈ 14.7."""
        assert constants.STOICHIOMETRIC_AFR == 14.7

    def test_wiebe_a(self) -> None:
        """Wiebe parameter a = -ln(1 - 0.999) ≈ 6.908."""
        computed = -math.log(1.0 - 0.999)
        assert abs(constants.WIEBE_A - computed) < 0.01


class TestTemperatureConversions:
    """Test temperature unit conversions."""

    def test_celsius_to_kelvin(self) -> None:
        assert celsius_to_kelvin(0.0) == 273.15
        assert celsius_to_kelvin(100.0) == 373.15
        assert celsius_to_kelvin(-273.15) == pytest.approx(0.0)

    def test_kelvin_to_celsius(self) -> None:
        assert kelvin_to_celsius(273.15) == 0.0
        assert kelvin_to_celsius(373.15) == 100.0

    def test_roundtrip(self) -> None:
        for t in [-40, 0, 15, 100, 500]:
            assert kelvin_to_celsius(celsius_to_kelvin(float(t))) == pytest.approx(float(t))


class TestPressureConversions:
    """Test pressure unit conversions."""

    def test_bar_to_pa(self) -> None:
        assert bar_to_pa(1.0) == 100000.0
        assert bar_to_pa(1.01325) == pytest.approx(101325.0)

    def test_pa_to_bar(self) -> None:
        assert pa_to_bar(100000.0) == 1.0

    def test_roundtrip(self) -> None:
        for p in [0.5, 1.0, 1.45, 5.0]:
            assert pa_to_bar(bar_to_pa(p)) == pytest.approx(p)


class TestRotationalConversions:
    """Test RPM and angular conversions."""

    def test_rpm_to_rad_s(self) -> None:
        # 60 RPM = 1 rev/s = 2π rad/s
        assert rpm_to_rad_s(60.0) == pytest.approx(2.0 * math.pi)

    def test_rad_s_to_rpm(self) -> None:
        assert rad_s_to_rpm(2.0 * math.pi) == pytest.approx(60.0)

    def test_roundtrip(self) -> None:
        for rpm in [800, 2000, 5800]:
            assert rad_s_to_rpm(rpm_to_rad_s(float(rpm))) == pytest.approx(float(rpm))


class TestAngularConversions:
    """Test degree/radian conversions."""

    def test_deg_to_rad(self) -> None:
        assert deg_to_rad(180.0) == pytest.approx(math.pi)
        assert deg_to_rad(360.0) == pytest.approx(2.0 * math.pi)

    def test_rad_to_deg(self) -> None:
        assert rad_to_deg(math.pi) == pytest.approx(180.0)


class TestLengthConversions:
    """Test length conversions."""

    def test_mm_to_m(self) -> None:
        assert mm_to_m(84.0) == pytest.approx(0.084)  # Rotax bore
        assert mm_to_m(61.0) == pytest.approx(0.061)  # Rotax stroke

    def test_cc_to_m3(self) -> None:
        assert cc_to_m3(1352.0) == pytest.approx(1.352e-3)  # Rotax displacement


class TestPowerConversions:
    """Test power conversions."""

    def test_kw_to_w(self) -> None:
        assert kw_to_w(105.0) == 105000.0  # Rotax rated power

    def test_w_to_kw(self) -> None:
        assert w_to_kw(105000.0) == 105.0

    def test_hp_to_w(self) -> None:
        # 1 HP ≈ 745.7 W
        assert hp_to_w(1.0) == pytest.approx(745.7, abs=0.1)

    def test_w_to_hp(self) -> None:
        # 105 kW ≈ 140.8 HP
        assert w_to_hp(105000.0) == pytest.approx(140.8, abs=0.5)


class TestISAModel:
    """Test International Standard Atmosphere model."""

    def test_sea_level_temperature(self) -> None:
        assert isa_temperature(0.0) == 288.15

    def test_sea_level_pressure(self) -> None:
        assert isa_pressure(0.0) == pytest.approx(101325.0, rel=1e-4)

    def test_sea_level_density(self) -> None:
        assert isa_density(0.0) == pytest.approx(1.225, rel=1e-3)

    def test_temperature_decreases_with_altitude(self) -> None:
        for alt in [1000, 3000, 5000, 8000]:
            assert isa_temperature(float(alt)) < isa_temperature(0.0)

    def test_pressure_decreases_with_altitude(self) -> None:
        for alt in [1000, 3000, 5000, 8000]:
            assert isa_pressure(float(alt)) < isa_pressure(0.0)

    def test_5000m_pressure(self) -> None:
        """At 5000 m, ISA pressure ≈ 54000 Pa (common reference)."""
        assert isa_pressure(5000.0) == pytest.approx(54020.0, rel=0.02)

    def test_lapse_rate(self) -> None:
        """Temperature drops 6.5 K per 1000 m."""
        t_0 = isa_temperature(0.0)
        t_1000 = isa_temperature(1000.0)
        assert t_0 - t_1000 == pytest.approx(6.5, abs=0.01)
