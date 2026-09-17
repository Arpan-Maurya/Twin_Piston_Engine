"""
Unit tests for Original Module 18:
Simulation Replay and What-if Engine.
"""

from datetime import datetime, timezone
import pytest
import numpy as np

from src.core.provenance import FaultClass, Provenance
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ScenarioRunner,
)
from src.l1_data.simulator.replay_and_whatif import (
    MetricDelta,
    PipelineReplayAdapter,
    ReplayEngine,
    ReplayState,
    ReplayStatus,
    ScenarioComparator,
    WhatIfEngine,
    WhatIfScenario,
)


class TestReplayEngine:
    """Tests for ReplayEngine player logic."""

    @pytest.fixture
    def sample_scenario(self):
        runner = ScenarioRunner(seed=42)
        records, ground_truths, meta = runner.run_scenario(duration_s=3.0, dt_s=1.0)
        return records, ground_truths

    def test_replay_lifecycle(self, sample_scenario):
        records, ground_truths = sample_scenario
        engine = ReplayEngine(scenario_id="scen_test")
        engine.load_scenario("scen_test", records, ground_truths)

        assert engine.is_loaded is True
        state = engine.get_state()
        assert state.status == ReplayStatus.STOPPED
        assert state.replay_position == 0

        engine.play()
        assert engine.get_state().status == ReplayStatus.PLAYING

        r0 = engine.step()
        assert r0 is not None
        assert r0.sequence_number == records[0].sequence_number

        engine.pause()
        assert engine.get_state().status == ReplayStatus.PAUSED
        assert engine.step() is None  # Paused does not step

        engine.resume()
        assert engine.get_state().status == ReplayStatus.PLAYING

        engine.seek(2)
        assert engine.get_state().replay_position == 2

        engine.restart()
        assert engine.get_state().replay_position == 0

    def test_seek_out_of_bounds_raises(self, sample_scenario):
        records, _ = sample_scenario
        engine = ReplayEngine()
        engine.load_scenario("s1", records)

        with pytest.raises(IndexError):
            engine.seek(99)

    def test_empty_records_load_raises(self):
        engine = ReplayEngine()
        with pytest.raises(ValueError, match="Cannot load empty"):
            engine.load_scenario("empty", [])


class TestWhatIfEngine:
    """Tests for WhatIfEngine cloning, validation, and execution."""

    def test_valid_what_if_creation(self):
        engine = WhatIfEngine()
        baseline_cfg = {"seed": 42, "rpm_profile": [3000.0, 4000.0]}
        mods = {
            "rpm_profile": [3500.0, 4500.0],
            "fault_scenarios": [
                FaultScenarioConfig(
                    fault_class=FaultClass.MISFIRE,
                    affected_cylinder=1,
                    severity=0.7,
                    start_time_s=1.0,
                    duration_s=5.0,
                )
            ],
            "seed": 99,
        }

        what_if = engine.create_what_if_scenario("base_1", baseline_cfg, mods)

        assert isinstance(what_if, WhatIfScenario)
        assert what_if.parent_scenario_id == "base_1"
        assert what_if.seed == 99
        assert baseline_cfg["rpm_profile"] == [3000.0, 4000.0]  # Baseline unmutated!

    def test_unsupported_parameter_rejection(self):
        engine = WhatIfEngine()
        baseline_cfg = {"seed": 42}
        invalid_mods = {"fake_turbo_boost_psi": 999.0}

        with pytest.raises(ValueError, match="Unsupported what-if parameter"):
            engine.create_what_if_scenario("base_1", baseline_cfg, invalid_mods)

    def test_invalid_fault_severity_rejection(self):
        engine = WhatIfEngine()
        with pytest.raises(ValueError, match="Severity must be in range"):
            FaultScenarioConfig(fault_class=FaultClass.MISFIRE, severity=1.8)

    def test_what_if_execution_determinism(self):
        engine = WhatIfEngine()
        baseline_cfg = {"seed": 100}
        mods = {"seed": 100, "rpm_profile": [4000.0, 4000.0]}

        what_if = engine.create_what_if_scenario("base_1", baseline_cfg, mods)
        recs1, gts1, meta1 = engine.execute_what_if(what_if, duration_s=2.0, dt_s=1.0)
        recs2, gts2, meta2 = engine.execute_what_if(what_if, duration_s=2.0, dt_s=1.0)

        for r1, r2 in zip(recs1, recs2):
            assert r1.crank_period_us == r2.crank_period_us
            assert r1.egt_cyl1_hot_uv == r2.egt_cyl1_hot_uv


class TestPipelineAndComparison:
    """Tests for PipelineReplayAdapter and ScenarioComparator."""

    def test_pipeline_adapter_execution(self):
        runner = ScenarioRunner(seed=42)
        records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)

        adapter = PipelineReplayAdapter()
        step_results = adapter.process_sequence(records)

        assert len(step_results) == 2
        for res in step_results:
            assert res.accepted is True
            assert res.health_index >= 0.0
            assert isinstance(res.predicted_fault_class, FaultClass)

    def test_scenario_comparison(self):
        runner = ScenarioRunner(seed=42)
        base_recs, base_gts, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)

        # Create misfire what-if scenario
        fault_mods = [
            FaultScenarioConfig(
                fault_class=FaultClass.MISFIRE,
                affected_cylinder=1,
                severity=0.8,
                start_time_s=0.0,
                duration_s=10.0,
            )
        ]
        whatif_recs, whatif_gts, _ = runner.run_scenario(
            duration_s=2.0, dt_s=1.0, fault_scenarios=fault_mods, seed=42
        )

        adapter = PipelineReplayAdapter()
        base_pipe_res = adapter.process_sequence(base_recs)
        whatif_pipe_res = adapter.process_sequence(whatif_recs)

        comparator = ScenarioComparator()
        comp = comparator.compare_runs(
            baseline_results=base_pipe_res,
            what_if_results=whatif_pipe_res,
            baseline_id="nominal_base",
            what_if_id="misfire_whatif",
            ground_truths=whatif_gts,
        )

        assert comp.baseline_id == "nominal_base"
        assert comp.what_if_id == "misfire_whatif"
        assert "health_index" in comp.metric_deltas
        assert "anomaly_score" in comp.metric_deltas
        assert comp.ground_truth_validation is not None
        assert comp.ground_truth_validation["isolated_from_inference"] is True
