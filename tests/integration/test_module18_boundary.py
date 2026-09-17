"""
Integration and Architecture Boundary Tests for Original Module 18:
Simulation Replay and What-if Engine.
"""

from datetime import datetime, timezone
import pytest

from src.core.provenance import FaultClass, Provenance
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ScenarioRunner,
)
from src.l1_data.simulator.replay_and_whatif import (
    PipelineReplayAdapter,
    ReplayEngine,
    ScenarioComparator,
    WhatIfEngine,
)


class TestModule18ArchitectureBoundary:
    """Architectural boundary enforcement tests for Module 18."""

    def test_ground_truth_isolation_during_inference(self):
        """Verify ground truth is NOT passed to PipelineReplayAdapter during inference."""
        runner = ScenarioRunner(seed=42)
        records, ground_truths, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)

        adapter = PipelineReplayAdapter()

        # Input to pipeline is strictly list[RawSignalRecord]
        pipeline_results = adapter.process_sequence(records)

        for step_res in pipeline_results:
            rec_dict = step_res.record.model_dump()
            # Confirm no ground truth fields exist on input record or step result
            assert "true_power_kw" not in rec_dict
            assert "combustion_efficiency" not in rec_dict
            assert "active_fault" not in rec_dict

    def test_what_if_baseline_immutability(self):
        """Verify WhatIfEngine never mutates baseline scenario configuration."""
        baseline_cfg = {
            "seed": 42,
            "rpm_profile": [3000.0, 4000.0, 5000.0],
            "fault_scenarios": [],
        }

        engine = WhatIfEngine()

        modifications = {
            "rpm_profile": [3500.0, 4500.0, 5500.0],
            "fault_scenarios": [
                FaultScenarioConfig(
                    fault_class=FaultClass.OIL_DEGRADATION,
                    severity=0.5,
                )
            ],
        }

        what_if = engine.create_what_if_scenario("base_01", baseline_cfg, modifications)

        # Assert baseline_cfg retains original values exactly
        assert baseline_cfg["rpm_profile"] == [3000.0, 4000.0, 5000.0]
        assert len(baseline_cfg["fault_scenarios"]) == 0
        assert what_if.parent_scenario_id == "base_01"
        assert what_if.resulting_configuration["rpm_profile"] == [3500.0, 4500.0, 5500.0]

    def test_unsupported_parameter_validation_error(self):
        """Verify invalid or unsupported what-if parameters trigger explicit ValueError."""
        engine = WhatIfEngine()
        baseline_cfg = {"seed": 42}

        # Case 1: Unknown key
        with pytest.raises(ValueError, match="Unsupported what-if parameter"):
            engine.create_what_if_scenario("base", baseline_cfg, {"illegal_param": 123})

        # Case 2: Out of range severity
        with pytest.raises(ValueError, match="Severity must be in range"):
            FaultScenarioConfig(fault_class=FaultClass.MISFIRE, severity=2.5)
