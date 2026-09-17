"""
Integration tests for Module 23 — Correctness Regression.

Verifies the foundational rule:
SAME INPUT + SAME CONFIG => SAME SCIENTIFIC OUTPUT
"""

import pytest
from src.core.config import get_settings
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter
from src.performance_resilience.cache import cache_deterministic


def test_scientific_correctness_invariance():
    settings = get_settings()

    # 1. Deterministic sequence generation
    runner = ScenarioRunner(seed=100)
    records, ground_truths, meta = runner.run_scenario(duration_s=10.0, dt_s=1.0)

    # 2. Run reference pipeline execution
    adapter_ref = PipelineReplayAdapter(settings)
    results_ref = adapter_ref.process_sequence(records)

    # 3. Run optimized pipeline execution
    adapter_opt = PipelineReplayAdapter(settings)
    results_opt = adapter_opt.process_sequence(records)

    assert len(results_ref) == len(results_opt)

    for ref, opt in zip(results_ref, results_opt):
        # Verify L2/L3 pipeline step output scientific fields match exactly
        assert ref.health_index == opt.health_index
        assert ref.anomaly_score == opt.anomaly_score
        assert ref.predicted_fault_class == opt.predicted_fault_class
        assert ref.rul_hours == opt.rul_hours
        assert ref.mission_risk_score == opt.mission_risk_score


def test_cached_math_exact_numerical_equality():
    @cache_deterministic(maxsize=16)
    def cached_power_calc(p_me: float, V_d: float, n: float) -> float:
        # P = (P_me * V_d * n) / 120
        return (p_me * V_d * n) / 120.0

    def raw_power_calc(p_me: float, V_d: float, n: float) -> float:
        return (p_me * V_d * n) / 120.0

    val_uncached = raw_power_calc(1.2e6, 0.002, 2500.0)
    val_cached = cached_power_calc(1.2e6, 0.002, 2500.0)
    val_cached_again = cached_power_calc(1.2e6, 0.002, 2500.0)

    assert val_uncached == val_cached == val_cached_again
