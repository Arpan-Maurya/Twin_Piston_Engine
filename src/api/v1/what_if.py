"""
What-If & Scenario Router — Module 18 What-If Engine Endpoints (Module 20).

Provides controlled scenario cloning, parameter modification, and scenario comparison.
Preserves baseline immutability; rejects unsupported parameters with validation errors.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_pipeline_adapter,
    get_replay_engine,
    get_scenario_comparator,
    get_what_if_engine,
)
from src.api.v1.schemas import (
    ScenarioCompareRequest,
    ScenarioCompareResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from src.core.provenance import Provenance
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import (
    PipelineReplayAdapter,
    ReplayEngine,
    ScenarioComparator,
    WhatIfEngine,
)

router = APIRouter(prefix="/scenarios", tags=["What-If Scenarios"])


@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create & Execute What-If Scenario",
    description="Clones a baseline scenario with validated parameter modifications (RPM, fault injection, seed). Does NOT mutate baseline.",
)
def create_what_if(
    request: WhatIfRequest,
    what_if_engine: WhatIfEngine = Depends(get_what_if_engine),
) -> WhatIfResponse:
    try:
        baseline_cfg = {"seed": 42}
        what_if_scen = what_if_engine.create_what_if_scenario(
            baseline_scenario_id=request.baseline_scenario_id,
            baseline_config=baseline_cfg,
            modifications=request.modifications,
        )

        records, ground_truths, meta = what_if_engine.execute_what_if(
            what_if_scen,
            duration_s=request.duration_s,
            dt_s=request.dt_s,
        )

        return WhatIfResponse(
            what_if_id=what_if_scen.what_if_id,
            parent_scenario_id=what_if_scen.parent_scenario_id,
            sample_count=len(records),
            seed=what_if_scen.seed,
            provenance=Provenance.SIMULATED,
            created_at=what_if_scen.created_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get(
    "/{scenario_id}",
    response_model=dict[str, Any],
    summary="Get Scenario Metadata",
    description="Returns scenario metadata and operating parameters.",
)
def get_scenario_by_id(
    scenario_id: str,
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> dict[str, Any]:
    state = replay_engine.get_state()
    if state.scenario_id != scenario_id and scenario_id != "default_scenario":
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")

    return {
        "scenario_id": scenario_id,
        "total_records": state.total_records,
        "provenance": Provenance.SIMULATED.value,
        "status": state.status.value if hasattr(state.status, "value") else str(state.status),
    }


@router.post(
    "/compare",
    response_model=ScenarioCompareResponse,
    summary="Compare Baseline vs What-If Scenario Pipeline Outputs",
    description="Executes baseline and what-if telemetry runs through L1->L2->L3 pipeline and returns metric deltas and state changes.",
)
def compare_scenarios(
    request: ScenarioCompareRequest,
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    adapter: PipelineReplayAdapter = Depends(get_pipeline_adapter),
    comparator: ScenarioComparator = Depends(get_scenario_comparator),
) -> ScenarioCompareResponse:
    base_records = replay_engine.get_all_records()
    if not base_records:
        raise HTTPException(status_code=400, detail="No baseline scenario available for comparison.")

    # Generate equivalent length what-if scenario
    runner = ScenarioRunner(seed=99)
    what_if_records, what_if_gts, _ = runner.run_scenario(
        duration_s=len(base_records) * 1.0,
        dt_s=1.0,
        seed=99,
    )

    base_results = adapter.process_sequence(base_records)
    what_if_results = adapter.process_sequence(what_if_records)

    comp = comparator.compare_runs(
        baseline_results=base_results,
        what_if_results=what_if_results,
        baseline_id=request.baseline_scenario_id,
        what_if_id=request.what_if_id,
        ground_truths=what_if_gts,
    )

    return ScenarioCompareResponse(
        baseline_id=comp.baseline_id,
        what_if_id=comp.what_if_id,
        metric_deltas={k: v.model_dump() if hasattr(v, "model_dump") else dataclasses.asdict(v) for k, v in comp.metric_deltas.items()},
        state_changes=comp.state_changes,
        timestamp_alignment_info=comp.timestamp_alignment_info,
        ground_truth_validation=comp.ground_truth_validation,
        provenance=Provenance.SIMULATED,
    )
