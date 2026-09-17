"""
Unit Tests for Original Module 5 — Sensor Inverse Modelling and Engineering-Unit Conversion.

Validates all Module 5 requirements:
    - Explicit inverse conversion for all raw sensor groups
    - EGT & CHT thermocouple conversion with cold-junction compensation
    - Pt100 RTD oil temperature conversion
    - ADC-based pressure conversions (MAP & Oil pressure with ADC Vref)
    - Crank period to RPM conversion (with zero/negative safety checks)
    - Fuel pulse to mass flow rate conversion
    - Accelerometer raw counts to m/s² and RMS
    - Ambient condition conversion
    - Quality & failure isolation propagation
    - Provenance tagging (DERIVED)
    - Deterministic repeated conversions
"""

import math
from datetime import datetime, timezone
import pytest

from src.core.config import AppSettings, load_settings
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l2_digital_twin.sensor_inverse import SensorInverseModel, convert_raw_to_engineering_state


@pytest.fixture
def valid_raw_record() -> RawSignalRecord:
    rec = RawSignalRecord(
        sequence_number=10,
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
        crank_period_us=15000.0,
        fuel_pulse_hz=100.0,
        accel_counts_xyz=(30, 40, 0),
        ambient_temp_c=20.0,
        ambient_press_pa=101325.0,
        signal_quality=SignalQuality(score=1.0),
    )
    return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})


class TestSensorInverseModelling:
    """Complete unit tests for Module 5 sensor inverse models."""

    def test_crank_period_to_rpm(self, valid_raw_record: RawSignalRecord) -> None:
        """15,000 us crank period -> 4000 RPM."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        assert eng.rpm.valid is True
        assert eng.rpm.value == pytest.approx(4000.0, rel=1e-3)
        assert eng.rpm.provenance == Provenance.DERIVED

    def test_zero_crank_period_handled_safely(self, valid_raw_record: RawSignalRecord) -> None:
        """Zero crank period must not divide by zero and must flag channel invalid."""
        bad_raw = valid_raw_record.model_copy(update={"crank_period_us": 0.0})
        eng = convert_raw_to_engineering_state(bad_raw)

        assert eng.rpm.valid is False
        assert eng.rpm.value == 0.0
        assert eng.rpm.fault_flag is not None
        assert "Non-positive crank period" in eng.rpm.fault_flag

    def test_thermocouple_egt_with_cold_junction(self, valid_raw_record: RawSignalRecord) -> None:
        """30,000 uV at Type-K (41.27 uV/°C) with 25°C cold junction -> ~751.8°C -> 1025.0 K."""
        eng = convert_raw_to_engineering_state(valid_raw_record)

        # Cyl 1: 25.0 + (30000 / 41.27) = 751.84 °C -> 1024.99 K
        expected_egt1_k = 25.0 + (30000.0 / 41.27) + 273.15
        assert eng.egt_cyl_1.value == pytest.approx(expected_egt1_k, rel=1e-4)
        assert eng.egt_cyl_1.valid is True
        assert eng.egt_cyl_1.provenance == Provenance.DERIVED

        # Per-cylinder identity preserved
        assert eng.egt_cyl_2.value > eng.egt_cyl_1.value

    def test_thermocouple_cht(self, valid_raw_record: RawSignalRecord) -> None:
        """12,000 uV at Type-K with 25°C cold junction -> ~315.7°C -> 588.9 K."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        expected_cht_k = 25.0 + (12000.0 / 41.27) + 273.15
        assert eng.cht_cyl_1.value == pytest.approx(expected_cht_k, rel=1e-4)

    def test_oil_rtd_conversion(self, valid_raw_record: RawSignalRecord) -> None:
        """135.0 ohms on Pt100 -> T_C = (135 - 100)/(100 * 0.00385) = 90.9°C -> 364.06 K."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        expected_temp_c = (135.0 - 100.0) / (100.0 * 0.00385)
        expected_temp_k = expected_temp_c + 273.15
        assert eng.oil_temp.value == pytest.approx(expected_temp_k, rel=1e-3)
        assert eng.oil_temp.valid is True

    def test_invalid_rtd_ohms_handled_safely(self, valid_raw_record: RawSignalRecord) -> None:
        bad_raw = valid_raw_record.model_copy(update={"oil_rtd_ohms": -10.0})
        eng = convert_raw_to_engineering_state(bad_raw)

        assert eng.oil_temp.valid is False
        assert eng.oil_temp.fault_flag is not None

    def test_adc_pressures(self, valid_raw_record: RawSignalRecord) -> None:
        """2048 / 4095 ADC counts for MAP -> ~100 kPa (100,024 Pa)."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        expected_map = (2048 / 4095.0) * 200000.0
        assert eng.map_pressure.value == pytest.approx(expected_map, rel=1e-3)
        assert eng.map_pressure.valid is True

    def test_fuel_pulse_conversion(self, valid_raw_record: RawSignalRecord) -> None:
        """100 Hz pulse -> 0.01 kg/s."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        assert eng.fuel_flow.value == pytest.approx(0.01, rel=1e-3)

    def test_accelerometer_conversion(self, valid_raw_record: RawSignalRecord) -> None:
        """Counts (30, 40, 0) / 10 -> (3.0, 4.0, 0.0) m/s², RMS = 5.0 m/s²."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        assert eng.vibration_x.value == pytest.approx(3.0)
        assert eng.vibration_y.value == pytest.approx(4.0)
        assert eng.vibration_z.value == pytest.approx(0.0)
        assert eng.vibration_rms.value == pytest.approx(5.0)

    def test_ambient_sensor_conversion(self, valid_raw_record: RawSignalRecord) -> None:
        """20.0 °C -> 293.15 K, 101325 Pa -> 101325 Pa."""
        eng = convert_raw_to_engineering_state(valid_raw_record)
        assert eng.ambient_temp.value == pytest.approx(293.15)
        assert eng.ambient_pressure.value == pytest.approx(101325.0)

    def test_failure_isolation_and_quality_propagation(self, valid_raw_record: RawSignalRecord) -> None:
        """Invalid raw channel must propagate failure ONLY to corresponding derived channel."""
        # Mark egt_cyl1_hot_uv as invalid in RawSignalRecord signal_quality
        sq_bad = SignalQuality(score=0.8, valid=False, invalid_channels=["egt_cyl1_hot_uv"])
        bad_raw = valid_raw_record.model_copy(update={"signal_quality": sq_bad})

        eng = convert_raw_to_engineering_state(bad_raw)

        # egt_cyl_1 must be marked invalid
        assert eng.egt_cyl_1.valid is False
        assert eng.egt_cyl_1.quality == 0.0

        # Unaffected channels must remain valid!
        assert eng.egt_cyl_2.valid is True
        assert eng.rpm.valid is True
        assert eng.oil_temp.valid is True

    def test_deterministic_repeated_conversion(self, valid_raw_record: RawSignalRecord) -> None:
        model = SensorInverseModel()
        res1 = model.convert_raw_to_engineering(valid_raw_record)
        res2 = model.convert_raw_to_engineering(valid_raw_record)
        assert res1 == res2
