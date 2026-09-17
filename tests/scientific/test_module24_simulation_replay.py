"""
Scientific Acceptance Suite — Modules 17–18: Forward Physics Simulator & Replay/What-If.
"""

import pytest
from src.core.config import get_settings
from src.core.provenance import FaultClass
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, ReplayStatus, WhatIfEngine


def test_forward_simulator_reproducibility():
    runner1 = ScenarioRunner(seed=42)
    records1, ground_truth1, meta1 = runner1.run_scenario(duration_s=3.0, dt_s=1.0)

    runner2 = ScenarioRunner(seed=42)
    records2, ground_truth2, meta2 = runner2.run_scenario(duration_s=3.0, dt_s=1.0)

    assert len(records1) == len(records2)
    for r1, r2 in zip(records1, records2):
        assert r1.sequence_number == r2.sequence_number
        assert r1.crank_period_us == r2.crank_period_us
        assert r1.egt_cyl1_hot_uv == r2.egt_cyl1_hot_uv


def test_forward_simulator_fault_injection_all_9_classes():
    fault_classes = list(FaultClass)
    for fc in fault_classes:
        f_cfg = FaultScenarioConfig(
            fault_class=fc,
            start_time_s=1.0,
            duration_s=5.0,
            severity=0.5,
            affected_cylinders=[1],
        )
        runner = ScenarioRunner(seed=42)
        records, gt_records, meta = runner.run_scenario(duration_s=3.0, dt_s=1.0, fault_scenarios=[f_cfg])

        assert len(records) == 3
        assert len(gt_records) == 3
        assert gt_records[1].injected_fault_class == fc


def test_simulation_replay_engine_controls():
    runner = ScenarioRunner(seed=42)
    records, gt, meta = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    replay = ReplayEngine("test_scen")
    replay.load_scenario("test_scen", records)

    assert replay.is_loaded is True
    replay.play()
    assert replay.get_state().status == ReplayStatus.PLAYING

    replay.step()
    assert replay.get_state().replay_position == 1

    replay.pause()
    assert replay.get_state().status == ReplayStatus.PAUSED


def test_what_if_baseline_non_mutation():
    settings = get_settings()
    engine = WhatIfEngine(settings)

    baseline_config = {"seed": 42}
    modifications = {"ambient_temp_k": 320.0, "seed": 99}

    what_if_scen = engine.create_what_if_scenario("base1", baseline_config, modifications)

    # Baseline config must NOT be mutated
    assert baseline_config["seed"] == 42
    assert "ambient_temp_k" not in baseline_config
    assert what_if_scen.resulting_configuration["ambient_temp_k"] == 320.0
