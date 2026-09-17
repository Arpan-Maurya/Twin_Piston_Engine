"""
Unit Tests for Original Module 6 — Thermodynamic and Mechanical Digital Twin.

Tests all thermodynamic derivations, mechanical derivations, geometry calculations,
time & state handling, SI unit compliance, and quality/failure propagation.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import get_settings
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import DerivedEngineState, SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import (
    EngineGeometry,
    MechanicalTwin,
    ThermodynamicMechanicalTwin,
    ThermodynamicTwin,
    evaluate_digital_twin,
)


def _make_dummy_normalized_record(
    rpm_val: float = 4000.0,
    map_pa: float = 120000.0,
    amb_press_pa: float = 101325.0,
    amb_temp_k: float = 288.15,
    fuel_kg_s: float = 0.005,
    lambda_val: float = 1.0,
    rpm_valid: bool = True,
    fuel_valid: bool = True,
    timestamp: datetime | None = None,
) -> NormalizedSignalRecord:
    """Helper to construct a NormalizedSignalRecord for testing."""
    ts = timestamp or datetime.now(timezone.utc)

    def _ch(v: float, valid: bool = True, q: float = 1.0) -> ChannelValue:
        return ChannelValue(
            value=v,
            provenance=Provenance.DERIVED,
            valid=valid,
            validity_reason=ChannelValidity.VALID if valid else ChannelValidity.INVALID_RANGE,
            quality=q if valid else 0.0,
        )

    return NormalizedSignalRecord(
        timestamp=ts,
        sequence_number=1,
        source_id="test_source",
        provenance=Provenance.DERIVED,
        integrity_hash="test_hash",
        rpm=_ch(rpm_val, valid=rpm_valid),
        map_pressure=_ch(map_pa),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(900.0),
        egt_cyl_2=_ch(905.0),
        egt_cyl_3=_ch(895.0),
        egt_cyl_4=_ch(900.0),
        cht_cyl_1=_ch(380.0),
        cht_cyl_2=_ch(380.0),
        cht_cyl_3=_ch(380.0),
        cht_cyl_4=_ch(380.0),
        oil_temp=_ch(360.0),
        oil_pressure=_ch(400000.0),
        coolant_temp=_ch(360.0),
        fuel_flow=_ch(fuel_kg_s, valid=fuel_valid),
        fuel_pressure=_ch(350000.0),
        intake_air_temp=_ch(295.0),
        ambient_pressure=_ch(amb_press_pa),
        ambient_temp=_ch(amb_temp_k),
        voltage=_ch(13.8),
        current=_ch(15.0),
        vibration_x=_ch(1.0),
        vibration_y=_ch(1.0),
        vibration_z=_ch(9.8),
        vibration_rms=_ch(9.9),
        propeller_speed=_ch(rpm_val * 0.414),
        boost_pressure=_ch(map_pa - amb_press_pa),
        wastegate_duty=_ch(40.0),
        lambda_sensor=_ch(lambda_val),
        ignition_timing_cyl_1=_ch(25.0),
        ignition_timing_cyl_2=_ch(25.0),
        ignition_timing_cyl_3=_ch(25.0),
        ignition_timing_cyl_4=_ch(25.0),
        altitude=_ch(1000.0),
        engine_hours=_ch(10.0),
    )


class TestEngineGeometry:
    """Tests for Rotax 915 iS engine geometry calculations."""

    def test_geometry_values(self) -> None:
        geom = EngineGeometry()
        assert math.isclose(geom.bore_m, 0.084, rel_tol=1e-3)
        assert math.isclose(geom.stroke_m, 0.061, rel_tol=1e-3)
        assert math.isclose(geom.displacement_m3, 0.001352, rel_tol=1e-3)
        assert geom.num_cylinders == 4
        assert geom.compression_ratio == 10.5

    def test_piston_speeds(self) -> None:
        geom = EngineGeometry()
        # Mean piston speed at 5800 RPM: 2 * 0.061 * (5800/60) = 11.793 m/s
        sp = geom.mean_piston_speed(5800.0)
        assert math.isclose(sp, 11.7933, rel_tol=1e-3)

        vmax = geom.max_piston_speed(5800.0)
        assert vmax > sp


class TestThermodynamicTwin:
    """Tests for Thermodynamic Digital Twin physics derivations."""

    def test_known_value_pressure_relationships(self) -> None:
        geom = EngineGeometry()
        thermo = ThermodynamicTwin(geom)
        rec = _make_dummy_normalized_record(map_pa=150000.0, amb_press_pa=100000.0)
        res = thermo.compute(rec)

        assert res.boost_pressure_pa.valid
        assert math.isclose(res.boost_pressure_pa.value, 50000.0, rel_tol=1e-5)
        assert res.pressure_ratio.valid
        assert math.isclose(res.pressure_ratio.value, 1.5, rel_tol=1e-5)

    def test_air_density_and_afr(self) -> None:
        geom = EngineGeometry()
        thermo = ThermodynamicTwin(geom)
        rec = _make_dummy_normalized_record(map_pa=101325.0, amb_temp_k=288.15, lambda_val=1.0)
        res = thermo.compute(rec)

        assert res.air_density_kg_m3.valid
        # Ideal air density at 101325 Pa and 295 K: ~1.196 kg/m3
        assert 1.15 < res.air_density_kg_m3.value < 1.25
        assert res.afr.valid
        assert math.isclose(res.afr.value, 14.7, rel_tol=1e-3)

    def test_fuel_energy_rate(self) -> None:
        geom = EngineGeometry()
        thermo = ThermodynamicTwin(geom)
        rec = _make_dummy_normalized_record(fuel_kg_s=0.005)
        res = thermo.compute(rec)

        assert res.fuel_energy_rate_w.valid
        # 0.005 kg/s * 43.5e6 J/kg = 217,500 W
        assert math.isclose(res.fuel_energy_rate_w.value, 217500.0, rel_tol=1e-4)


class TestMechanicalTwin:
    """Tests for Mechanical Digital Twin derivations."""

    def test_crank_kinematics(self) -> None:
        geom = EngineGeometry()
        mech = MechanicalTwin(geom)
        rec = _make_dummy_normalized_record(rpm_val=6000.0)
        res = mech.compute(rec)

        assert res.crank_angular_velocity_rad_s.valid
        # 6000 RPM -> 2 * pi * 100 = 628.318 rad/s
        assert math.isclose(res.crank_angular_velocity_rad_s.value, 628.3185, rel_tol=1e-3)
        assert res.mean_piston_speed_m_s.valid

    def test_angular_acceleration_temporal(self) -> None:
        geom = EngineGeometry()
        mech = MechanicalTwin(geom)

        rec1 = _make_dummy_normalized_record(rpm_val=4000.0)
        res1 = mech.compute(rec1, prev_omega=None, dt_s=None)
        assert res1.crank_angular_acceleration_rad_s2.value == 0.0

        # Acceleration over 1 second from 4000 RPM (418.88 rad/s) to 5000 RPM (523.60 rad/s)
        omega1 = 4000.0 * 2.0 * math.pi / 60.0
        rec2 = _make_dummy_normalized_record(rpm_val=5000.0)
        res2 = mech.compute(rec2, prev_omega=omega1, dt_s=1.0)

        expected_alpha = (5000.0 * 2.0 * math.pi / 60.0 - omega1) / 1.0
        assert math.isclose(res2.crank_angular_acceleration_rad_s2.value, expected_alpha, rel_tol=1e-3)

    test_zero_dt_handling = lambda self: self._test_temporal_edge_case(0.0)
    test_negative_dt_handling = lambda self: self._test_temporal_edge_case(-1.0)

    def _test_temporal_edge_case(self, dt: float) -> None:
        geom = EngineGeometry()
        mech = MechanicalTwin(geom)
        rec = _make_dummy_normalized_record(rpm_val=4000.0)
        res = mech.compute(rec, prev_omega=300.0, dt_s=dt)
        assert res.crank_angular_acceleration_rad_s2.value == 0.0


class TestThermodynamicMechanicalTwin:
    """Integrated Digital Twin evaluation tests."""

    def test_evaluate_derived_engine_state(self) -> None:
        twin = ThermodynamicMechanicalTwin()
        rec = _make_dummy_normalized_record(rpm_val=5000.0, map_pa=130000.0, fuel_kg_s=0.006)
        state = twin.evaluate(rec)

        assert isinstance(state, DerivedEngineState)
        assert state.provenance == Provenance.DERIVED
        assert state.brake_power_kw.valid
        assert state.brake_torque_nm.valid
        assert state.bmep_pa.valid
        assert state.fmep_pa.valid
        assert state.eta_volumetric.valid
        assert state.mean_piston_speed_m_s.valid

        assert state.brake_power_kw.value > 0.0
        assert state.brake_torque_nm.value > 0.0

    def test_missing_rpm_isolation(self) -> None:
        """Verify invalid RPM invalidates speed-dependent quantities without raising errors."""
        twin = ThermodynamicMechanicalTwin()
        rec = _make_dummy_normalized_record(rpm_val=0.0, rpm_valid=False)
        state = twin.evaluate(rec)

        assert not state.mean_piston_speed_m_s.valid
        assert not state.brake_power_kw.valid
        assert not state.bmep_pa.valid
        # Independent AFR calculation remains valid
        assert state.afr.valid

    def test_missing_fuel_flow_isolation(self) -> None:
        """Verify invalid fuel flow falls back to speed-density power/torque estimate safely."""
        twin = ThermodynamicMechanicalTwin()
        rec = _make_dummy_normalized_record(fuel_valid=False)
        state = twin.evaluate(rec)

        # Power and torque are estimated via speed-density with quality=0.7
        assert state.brake_power_kw.valid
        assert state.brake_power_kw.quality == 0.7
        assert not state.eta_thermal.valid  # Thermal efficiency requires measured fuel flow

    def test_raw_record_end_to_end(self) -> None:
        """Verify module 5 -> module 6 end-to-end processing pipeline."""
        raw_record = RawSignalRecord(
            sequence_number=1,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0,
            egt_cyl2_hot_uv=30500.0,
            egt_cyl3_hot_uv=29800.0,
            egt_cyl4_hot_uv=30100.0,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0,
            oil_p_counts=2048,
            map_counts=2048,
            adc_vref_counts=4095,
            crank_period_us=15000.0,  # 4000 RPM
            fuel_pulse_hz=100.0,
            accel_counts_xyz=(10, 20, 980),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )

        norm_record = convert_raw_to_engineering_state(raw_record)
        state = evaluate_digital_twin(norm_record)

        assert state.provenance == Provenance.DERIVED
        assert state.mean_piston_speed_m_s.valid
        assert state.mean_piston_speed_m_s.value > 0.0
