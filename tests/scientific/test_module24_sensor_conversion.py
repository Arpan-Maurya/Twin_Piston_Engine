"""
Scientific Acceptance Suite — Module 5: Sensor Inverse Modelling.
"""

import pytest
from src.core.config import get_settings
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l2_digital_twin.sensor_inverse import SensorInverseModel


def test_sensor_inverse_conversion_units_and_properties():
    settings = get_settings()
    model = SensorInverseModel(settings=settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    raw_rec = records[0]

    norm = model.convert_raw_to_engineering(raw_rec)

    # 1. Crank Period -> Engine Speed (RPM)
    expected_rpm = (60.0 * 1e6) / raw_rec.crank_period_us
    assert norm.rpm.valid is True
    assert norm.rpm.value == pytest.approx(expected_rpm, abs=1e-3)

    # 2. Thermocouple EGT (microvolts -> Kelvin)
    cal = settings.sensor_calibration
    expected_egt1_k = (raw_rec.egt_cold_c + (raw_rec.egt_cyl1_hot_uv / cal.type_k_uv_per_c)) + 273.15
    assert norm.egt_cyl_1.valid is True
    assert norm.egt_cyl_1.value == pytest.approx(expected_egt1_k, abs=1e-3)

    # 3. Ambient Temperature (C -> Kelvin)
    assert norm.ambient_temp.value == pytest.approx(raw_rec.ambient_temp_c + 273.15, abs=1e-3)

    # 4. Ambient Pressure (mbar -> Pa)
    assert norm.ambient_pressure.value == pytest.approx(raw_rec.ambient_press_pa, abs=1e-3)


def test_sensor_inverse_invalid_data_handling():
    settings = get_settings()
    model = SensorInverseModel(settings=settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=1.0, dt_s=1.0)
    raw_rec = records[0]

    # Create invalid record with zero crank period
    raw_invalid = raw_rec.model_copy(update={"crank_period_us": 0.0})
    norm = model.convert_raw_to_engineering(raw_invalid)

    assert norm.rpm.valid is False
    assert norm.rpm.quality == 0.0
    assert norm.rpm.fault_flag is not None
