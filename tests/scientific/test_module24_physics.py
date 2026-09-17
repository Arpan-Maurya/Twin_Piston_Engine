"""
Scientific Acceptance Suite — Module 6: Thermodynamic & Mechanical Twin.
"""

import math
import pytest
from src.core.config import get_settings
from src.l2_digital_twin.sensor_inverse import SensorInverseModel
from src.l2_digital_twin.thermo_mechanical_twin import EngineGeometry, ThermodynamicMechanicalTwin
from src.l1_data.simulator.forward_simulator import ScenarioRunner


def test_engine_geometry_rotax_915is():
    settings = get_settings()
    geom = EngineGeometry(settings)

    assert geom.bore_m == 0.084
    assert geom.stroke_m == 0.061
    assert geom.displacement_m3 == pytest.approx(0.001352, abs=1e-5)
    assert geom.num_cylinders == 4
    assert geom.compression_ratio == 10.5
    assert geom.lhv_j_kg == 43.5e6
    assert geom.stoichiometric_afr == 14.7

    # Mean piston speed formula check S_p = 2 * stroke * (RPM / 60)
    sp_5800 = geom.mean_piston_speed(5800)
    expected_sp = 2.0 * 0.061 * (5800 / 60.0)
    assert sp_5800 == pytest.approx(expected_sp, abs=1e-4)


def test_thermo_mechanical_twin_evaluation_bounds():
    settings = get_settings()
    inverse_model = SensorInverseModel(settings)
    twin = ThermodynamicMechanicalTwin(settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    raw_rec = records[0]

    norm = inverse_model.convert_raw_to_engineering(raw_rec)
    derived_state = twin.evaluate(norm)

    # 1. Brake Power & Torque sanity bounds
    assert derived_state.brake_power_kw.value >= 0.0
    assert derived_state.brake_power_kw.value <= 150.0  # Max rated 105 kW + boost margin
    assert derived_state.brake_torque_nm.value >= 0.0

    # 2. Mean Piston Speed
    assert derived_state.mean_piston_speed_m_s.value > 0.0
    assert derived_state.mean_piston_speed_m_s.value < 25.0  # Realistic piston speed cap
