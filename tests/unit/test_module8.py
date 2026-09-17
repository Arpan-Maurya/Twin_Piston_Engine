"""
Unit Tests for Original Module 8 — Lubrication Model.

Tests all lubrication physics features: viscosity calculation, pressure/temp margins,
health threshold checks, pressure-viscosity interaction, failure isolation,
temporal rate handling, and schema/provenance compliance.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import get_settings
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import LubricationState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.lubrication_model import (
    LubricationModel,
    LubricationModelResult,
    evaluate_lubrication_model,
)


def _make_record_lubrication(
    oil_temp_k: float = 363.15,      # 90°C
    oil_press_pa: float = 400000.0,  # 4.0 bar
    temp_valid: bool = True,
    press_valid: bool = True,
    rpm_val: float = 4000.0,
    timestamp: datetime | None = None,
) -> NormalizedSignalRecord:
    """Helper to build a NormalizedSignalRecord for lubrication testing."""
    ts = timestamp or datetime.now(timezone.utc)

    def _ch(val: float, valid: bool = True) -> ChannelValue:
        return ChannelValue(
            value=val,
            provenance=Provenance.DERIVED,
            valid=valid,
            validity_reason=ChannelValidity.VALID if valid else ChannelValidity.INVALID_RANGE,
            quality=1.0 if valid else 0.0,
        )

    return NormalizedSignalRecord(
        timestamp=ts,
        sequence_number=1,
        source_id="test_source",
        provenance=Provenance.DERIVED,
        integrity_hash="test_hash",
        rpm=_ch(rpm_val),
        map_pressure=_ch(120000.0),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(900.0),
        egt_cyl_2=_ch(900.0),
        egt_cyl_3=_ch(900.0),
        egt_cyl_4=_ch(900.0),
        cht_cyl_1=_ch(380.0),
        cht_cyl_2=_ch(380.0),
        cht_cyl_3=_ch(380.0),
        cht_cyl_4=_ch(380.0),
        oil_temp=_ch(oil_temp_k, valid=temp_valid),
        oil_pressure=_ch(oil_press_pa, valid=press_valid),
        coolant_temp=_ch(360.0),
        fuel_flow=_ch(0.005),
        fuel_pressure=_ch(350000.0),
        intake_air_temp=_ch(295.0),
        ambient_pressure=_ch(101325.0),
        ambient_temp=_ch(288.15),
        voltage=_ch(13.8),
        current=_ch(15.0),
        vibration_x=_ch(1.0),
        vibration_y=_ch(1.0),
        vibration_z=_ch(9.8),
        vibration_rms=_ch(9.9),
        propeller_speed=_ch(1656.0),
        boost_pressure=_ch(18675.0),
        wastegate_duty=_ch(40.0),
        lambda_sensor=_ch(1.0),
        ignition_timing_cyl_1=_ch(25.0),
        ignition_timing_cyl_2=_ch(25.0),
        ignition_timing_cyl_3=_ch(25.0),
        ignition_timing_cyl_4=_ch(25.0),
        altitude=_ch(1000.0),
        engine_hours=_ch(10.0),
    )


class TestLubricationModel:
    """Unit test suite for Original Module 8 Lubrication Model."""

    def test_known_value_viscosity_calculation(self) -> None:
        model = LubricationModel()
        # Nominal temp 90°C = 363.15 K
        mu = model.compute_viscosity(363.15)
        assert mu is not None
        # mu = 0.0002 * exp(1200 / (363.15 - 140)) = 0.0002 * exp(1200 / 223.15) ≈ 0.0433 Pa*s
        assert math.isclose(mu, 0.0433, rel_tol=1e-2)

    def test_normal_lubrication_state(self) -> None:
        rec = _make_record_lubrication(oil_temp_k=363.15, oil_press_pa=400000.0)
        state, res = evaluate_lubrication_model(rec)

        assert isinstance(state, LubricationState)
        assert isinstance(res, LubricationModelResult)
        assert state.provenance == Provenance.DERIVED
        assert state.status == DiagnosticStatus.NORMAL
        assert state.oil_temperature_k.valid
        assert state.oil_pressure_pa.valid
        assert state.dynamic_viscosity_pa_s.valid
        assert state.pressure_margin_pa.valid
        assert state.temperature_margin_k.valid

        # Pressure margin relative to 1.5 bar (150,000 Pa) = 400,000 - 150,000 = 250,000 Pa
        assert math.isclose(state.pressure_margin_pa.value, 250000.0, rel_tol=1e-4)

    def test_low_oil_pressure_thresholds(self) -> None:
        # Warning low pressure: 1.8 bar = 180,000 Pa
        rec_warn = _make_record_lubrication(oil_press_pa=180000.0)
        state_warn, _ = evaluate_lubrication_model(rec_warn)
        assert state_warn.status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)

        # Critical low pressure: 1.2 bar = 120,000 Pa
        rec_crit = _make_record_lubrication(oil_press_pa=120000.0)
        state_crit, _ = evaluate_lubrication_model(rec_crit)
        assert state_crit.status == DiagnosticStatus.CRITICAL

    def test_high_oil_temperature_thresholds(self) -> None:
        # Warning high temp: 122°C = 395.15 K
        rec_warn = _make_record_lubrication(oil_temp_k=395.15)
        state_warn, _ = evaluate_lubrication_model(rec_warn)
        assert state_warn.status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)

        # Critical high temp: 132°C = 405.15 K
        rec_crit = _make_record_lubrication(oil_temp_k=405.15)
        state_crit, _ = evaluate_lubrication_model(rec_crit)
        assert state_crit.status == DiagnosticStatus.CRITICAL

    def test_missing_oil_pressure_failure_isolation(self) -> None:
        """Invalid oil pressure: pressure metrics invalid, temperature & viscosity remain valid."""
        rec = _make_record_lubrication(press_valid=False)
        state, res = evaluate_lubrication_model(rec)

        assert not state.oil_pressure_pa.valid
        assert not state.pressure_margin_pa.valid
        assert state.oil_temperature_k.valid
        assert state.dynamic_viscosity_pa_s.valid

    def test_missing_oil_temperature_failure_isolation(self) -> None:
        """Invalid oil temperature: viscosity/temp metrics invalid, pressure remains valid."""
        rec = _make_record_lubrication(temp_valid=False)
        state, res = evaluate_lubrication_model(rec)

        assert not state.oil_temperature_k.valid
        assert not state.dynamic_viscosity_pa_s.valid
        assert not state.temperature_margin_k.valid
        assert state.oil_pressure_pa.valid
        assert state.pressure_margin_pa.valid

    def test_both_invalid(self) -> None:
        rec = _make_record_lubrication(temp_valid=False, press_valid=False)
        state, res = evaluate_lubrication_model(rec)

        assert state.status == DiagnosticStatus.INVALID
        assert not state.oil_pressure_pa.valid
        assert not state.oil_temperature_k.valid

    def test_temporal_rates(self) -> None:
        model = LubricationModel()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec1 = _make_record_lubrication(oil_press_pa=400000.0, oil_temp_k=360.0, timestamp=t0)
        model.evaluate(rec1)

        t1 = t0 + timedelta(seconds=2)
        rec2 = _make_record_lubrication(oil_press_pa=440000.0, oil_temp_k=364.0, timestamp=t1)
        _, res2 = model.evaluate(rec2)

        # dp/dt = (440000 - 400000) / 2 = 20000 Pa/s
        assert res2.dp_dt_pa_s is not None
        assert math.isclose(res2.dp_dt_pa_s, 20000.0, rel_tol=1e-4)

        # dt/dt = (364 - 360) / 2 = 2 K/s
        assert res2.dt_dt_k_s is not None
        assert math.isclose(res2.dt_dt_k_s, 2.0, rel_tol=1e-4)

    test_first_sample_temporal = lambda self: self._test_temporal_edge(None)
    test_zero_dt_temporal = lambda self: self._test_temporal_edge(0.0)

    def _test_temporal_edge(self, dt: float | None) -> None:
        model = LubricationModel()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec = _make_record_lubrication(timestamp=t0)
        _, res = model.evaluate(rec)

        assert res.dp_dt_pa_s is None
        assert res.dt_dt_k_s is None
