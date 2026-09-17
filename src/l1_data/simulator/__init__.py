"""
L1 Data — Forward Physics Simulator and Replay/What-if package (Modules 17 & 18).

Module 17: Physics-based engine simulator and fault injector for Rotax 915 iS.
Module 18: Simulation replay engine, what-if scenario generator, pipeline replay adapter, and scenario comparator.

STRICT BOUNDARY CONSTRAINTS:
Ground truth physical state is isolated and NEVER accessible to L2 or L3 during inference.
Replayed data uses canonical RawSignalRecord (Provenance.SIMULATED).
"""

from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    ScenarioRunner,
    SensorForwardModel,
    SimulationGroundTruth,
    SimulationMetadata,
)
from src.l1_data.simulator.replay_and_whatif import (
    MetricDelta,
    PipelineReplayAdapter,
    PipelineStepResult,
    ReplayEngine,
    ReplayState,
    ReplayStatus,
    ScenarioComparator,
    ScenarioComparison,
    WhatIfEngine,
    WhatIfScenario,
)

__all__ = [
    # Module 17
    "FaultScenarioConfig",
    "ForwardPhysicsModel",
    "ScenarioRunner",
    "SensorForwardModel",
    "SimulationGroundTruth",
    "SimulationMetadata",
    # Module 18
    "MetricDelta",
    "PipelineReplayAdapter",
    "PipelineStepResult",
    "ReplayEngine",
    "ReplayState",
    "ReplayStatus",
    "ScenarioComparator",
    "ScenarioComparison",
    "WhatIfEngine",
    "WhatIfScenario",
]
