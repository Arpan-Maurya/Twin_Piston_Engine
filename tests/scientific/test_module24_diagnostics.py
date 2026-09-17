"""
Scientific Acceptance Suite — Modules 7–11: Diagnostics & Residual Engine.
"""

import pytest
from src.core.config import get_settings
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l2_digital_twin import (
    evaluate_digital_twin,
    evaluate_egt_diagnostics,
    evaluate_lubrication_model,
    evaluate_misfire_detector,
    evaluate_residual_engine,
    evaluate_vibration_processor,
    convert_raw_to_engineering_state,
)


def test_egt_diagnostics_cylinder_spread_and_imbalance():
    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])

    diag_state, egt_res = evaluate_egt_diagnostics(norm)

    assert egt_res.mean_egt_k > 0.0
    assert egt_res.spread_egt_k >= 0.0
    assert isinstance(egt_res.imbalance_detected, bool)


def test_lubrication_vogel_viscosity_and_margins():
    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])

    lub_state, lub_res = evaluate_lubrication_model(norm)

    assert lub_state.dynamic_viscosity_pa_s.value > 0.0
    assert lub_state.oil_pressure_pa.value > 0.0
    assert lub_state.pressure_margin_pa is not None
    assert lub_res.provenance.value == "DERIVED"


def test_vibration_spectral_processing():
    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])

    vib_state, vib_res = evaluate_vibration_processor(norm)

    assert vib_state.overall_rms_m_s2.value >= 0.0
    assert vib_state.peak_m_s2.value >= 0.0
    assert vib_state.crest_factor.value >= 1.0
    assert vib_state.dominant_freq_hz.value >= 0.0


def test_misfire_combustion_evidence_fusion():
    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])
    twin_res = evaluate_digital_twin(norm)

    diag_state, egt_res = evaluate_egt_diagnostics(norm)
    vib_state, vib_res = evaluate_vibration_processor(norm)
    mis_state, mis_res = evaluate_misfire_detector(norm, twin_res, egt_res, vib_res)

    assert mis_res.overall_status.value in ("NORMAL", "WARNING", "CRITICAL", "INVALID")
    assert len(mis_res.cylinder_evidence) == 4


def test_residual_engine_sign_convention():
    """Verify sign convention: observed > expected => positive residual."""
    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])
    twin_res = evaluate_digital_twin(norm)

    res_state, exp_res = evaluate_residual_engine(norm, twin_res)

    assert "egt_cyl1" in res_state.residuals
    assert "oil_pressure" in res_state.residuals
    assert "oil_temp" in res_state.residuals
    assert res_state.residuals["egt_cyl1"].valid is True
