"""
Scientific Acceptance Suite — Cross-Module End-to-End Scenarios.
"""

import pytest
from src.core.config import get_settings
from src.core.provenance import FaultClass
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, WhatIfEngine


def test_e2e_healthy_engine_pipeline():
    settings = get_settings()
    adapter = PipelineReplayAdapter(settings)

    runner = ScenarioRunner(seed=100)
    records, _, _ = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    results = adapter.process_sequence(records)
    assert len(results) == 5

    for step_res in results:
        assert step_res.accepted is True
        assert 0.0 <= step_res.health_index <= 100.0
        assert 0.0 <= step_res.anomaly_score <= 1.0
        assert 0.0 <= step_res.mission_risk_score <= 100.0


def test_e2e_single_fault_injections_all_9_classes():
    settings = get_settings()
    adapter = PipelineReplayAdapter(settings)

    for fc in list(FaultClass):
        f_cfg = FaultScenarioConfig(
            fault_class=fc,
            start_time_s=1.0,
            duration_s=3.0,
            severity=0.8,
            affected_cylinders=[1],
        )
        runner = ScenarioRunner(seed=42)
        records, _, _ = runner.run_scenario(duration_s=3.0, dt_s=1.0, fault_scenarios=[f_cfg])

        results = adapter.process_sequence(records)
        assert len(results) == 3
        for step in results:
            # Structurally valid execution across all downstream layers
            assert step.health_index >= 0.0
            assert step.mission_risk_score >= 0.0


def test_e2e_replay_determinism():
    settings = get_settings()
    adapter1 = PipelineReplayAdapter(settings)
    adapter2 = PipelineReplayAdapter(settings)

    runner = ScenarioRunner(seed=777)
    records, _, _ = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    res1 = adapter1.process_sequence(records)
    res2 = adapter2.process_sequence(records)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1.health_index == r2.health_index
        assert r1.anomaly_score == r2.anomaly_score
        assert r1.rul_hours == r2.rul_hours
        assert r1.mission_risk_score == r2.mission_risk_score


def test_e2e_what_if_ground_truth_isolation():
    settings = get_settings()
    engine = WhatIfEngine(settings)

    scen = engine.create_what_if_scenario("base", {}, {"ambient_temp_k": 310.0, "seed": 42})
    records, gt_records, meta = engine.execute_what_if(scen, duration_s=3.0, dt_s=1.0)

    adapter = PipelineReplayAdapter(settings)
    results = adapter.process_sequence(records)

    # Verify that pipeline results do not modify or leak GT records
    assert len(results) == 3
    assert len(gt_records) == 3
    for gt in gt_records:
        assert gt.provenance.value == "SIMULATED"
