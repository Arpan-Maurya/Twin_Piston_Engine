"""
Unit Tests for Original Module 11 — Healthy Expectation Models and Residual Engine.

Tests operating-context expectation generation, residual computation (raw, absolute, normalized),
per-cylinder EGT residuals, failure isolation, zero-scale protection, and robustness.
"""

import math
from datetime import datetime, timezone

import pytest

from src.core.config import get_settings
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import OperatingPoint, ResidualState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.residual_engine import (
    HealthyExpectationModel,
    HealthyExpectationResult,
    ResidualEngine,
    evaluate_residual_engine,
)


def _make_record_residual(
    egt1: float = 950.0,
    oil_press_pa: float = 400000.0,
    oil_temp_k: float = 360.0,
    vib_rms: float = 5.0,
    valid1: bool = True,
    press_valid: bool = True,
    timestamp: datetime | None = None,
) -> NormalizedSignalRecord:
    """Helper to build a NormalizedSignalRecord for residual testing."""
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
        rpm=_ch(4000.0),
        map_pressure=_ch(120000.0),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(egt1, valid=valid1),
        egt_cyl_2=_ch(950.0),
        egt_cyl_3=_ch(950.0),
        egt_cyl_4=_ch(950.0),
        cht_cyl_1=_ch(380.0),
        cht_cyl_2=_ch(380.0),
        cht_cyl_3=_ch(380.0),
        cht_cyl_4=_ch(380.0),
        oil_temp=_ch(oil_temp_k),
        oil_pressure=_ch(oil_press_pa, valid=press_valid),
        coolant_temp=_ch(360.0),
        fuel_flow=_ch(0.005),
        fuel_pressure=_ch(350000.0),
        intake_air_temp=_ch(295.0),
        ambient_pressure=_ch(101325.0),
        ambient_temp=_ch(288.15),
        voltage=_ch(13.8),
        current=_ch(15.0),
        vibration_x=_ch(vib_rms / math.sqrt(3)),
        vibration_y=_ch(vib_rms / math.sqrt(3)),
        vibration_z=_ch(vib_rms / math.sqrt(3)),
        vibration_rms=_ch(vib_rms),
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


class TestHealthyExpectationModel:
    """Unit test suite for Healthy Expectation Model."""

    def test_expectation_operating_point_dependence(self) -> None:
        model = HealthyExpectationModel()

        op_idle = OperatingPoint(
            rpm=1200.0,
            map_pressure_pa=40000.0,
            altitude_m=0.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=10.0,
        )
        op_cruise = OperatingPoint(
            rpm=5000.0,
            map_pressure_pa=130000.0,
            altitude_m=1000.0,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325.0,
            throttle_pct=75.0,
        )

        exp_egt_idle = model.expected_egt_k(op_idle)
        exp_egt_cruise = model.expected_egt_k(op_cruise)

        # Higher load and speed should yield higher expected EGT
        assert exp_egt_cruise > exp_egt_idle


class TestResidualEngine:
    """Unit test suite for Residual Engine."""

    def test_residual_zero_when_observed_equals_expected(self) -> None:
        engine = ResidualEngine()
        q_res = engine.compute_residual("test", observed=100.0, observed_valid=True, expected=100.0, scale=50.0)

        assert q_res.valid
        assert q_res.raw_residual == 0.0
        assert q_res.absolute_residual == 0.0
        assert q_res.normalized_residual == 0.0

    def test_residual_positive_and_negative(self) -> None:
        engine = ResidualEngine()

        q_pos = engine.compute_residual("test_pos", observed=110.0, observed_valid=True, expected=100.0, scale=10.0)
        assert q_pos.raw_residual == 10.0
        assert q_pos.absolute_residual == 10.0
        assert math.isclose(q_pos.normalized_residual, 1.0, rel_tol=1e-5)

        q_neg = engine.compute_residual("test_neg", observed=90.0, observed_valid=True, expected=100.0, scale=10.0)
        assert q_neg.raw_residual == -10.0
        assert q_neg.absolute_residual == 10.0
        assert math.isclose(q_neg.normalized_residual, -1.0, rel_tol=1e-5)

    def test_zero_scale_protection(self) -> None:
        engine = ResidualEngine()
        q_res = engine.compute_residual("test_zero_scale", observed=100.0, observed_valid=True, expected=100.0, scale=0.0)

        assert not q_res.valid
        assert q_res.raw_residual is None

    def test_evaluate_record_residuals(self) -> None:
        rec = _make_record_residual()
        state, res = evaluate_residual_engine(rec)

        assert isinstance(state, ResidualState)
        assert isinstance(res, HealthyExpectationResult)
        assert state.provenance == Provenance.DERIVED
        assert "egt_cyl1" in state.residuals
        assert state.residuals["egt_cyl1"].valid

    def test_per_cylinder_failure_isolation(self) -> None:
        """Cyl 1 invalid: Cyl 1 residual invalid, Cyl 2-4 residuals valid."""
        rec = _make_record_residual(valid1=False)
        state, res = evaluate_residual_engine(rec)

        assert not res.residuals["egt_cyl1"].valid
        assert res.residuals["egt_cyl2"].valid
        assert res.residuals["egt_cyl3"].valid
        assert res.residuals["egt_cyl4"].valid

    def test_missing_pressure_isolation(self) -> None:
        rec = _make_record_residual(press_valid=False)
        state, res = evaluate_residual_engine(rec)

        assert not res.residuals["oil_pressure"].valid
        assert res.residuals["egt_cyl1"].valid
        assert res.residuals["oil_temp"].valid
