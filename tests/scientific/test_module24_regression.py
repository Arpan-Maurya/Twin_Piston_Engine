"""
Scientific Acceptance Suite — Modules 22–23: Validation Harness & Performance Invariance.
"""

import pytest
from src.core.config import get_settings
from src.dataset_validation import ValidationHarness, ClassificationMetrics, MetricsCalculator
from src.performance_resilience import BenchmarkHarness, FastLRUCache, cache_deterministic
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter


def test_dataset_validation_harness_integrity():
    calc = MetricsCalculator()
    y_true = [0] * 100
    y_pred = [0] * 100

    cm = calc.calculate_classification_metrics(y_true, y_pred)
    assert cm.accuracy == 1.0


def test_optimization_scientific_output_invariance():
    settings = get_settings()

    runner = ScenarioRunner(seed=55)
    records, _, _ = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    adapter_ref = PipelineReplayAdapter(settings)
    results_ref = adapter_ref.process_sequence(records)

    adapter_opt = PipelineReplayAdapter(settings)
    results_opt = adapter_opt.process_sequence(records)

    assert len(results_ref) == len(results_opt)
    for ref, opt in zip(results_ref, results_opt):
        assert ref.health_index == opt.health_index
        assert ref.anomaly_score == opt.anomaly_score
        assert ref.predicted_fault_class == opt.predicted_fault_class
        assert ref.rul_hours == opt.rul_hours
        assert ref.mission_risk_score == opt.mission_risk_score


def test_module23_benchmark_harness_measurement():
    harness = BenchmarkHarness()
    res = harness.run_benchmark(num_records=10)

    assert res.telemetry_ingestion_latency_ms >= 0.0
    assert res.digital_twin_latency_ms >= 0.0
    assert res.pipeline_total_latency_ms >= 0.0
    assert res.replay_throughput_records_per_sec > 0.0
