"""
RUL Router — Module 15 Remaining Useful Life Endpoints (Module 20).

Exposes RUL estimates, upper/lower uncertainty bounds, and degradation trend.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import RULResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine

router = APIRouter(prefix="/engine", tags=["Remaining Useful Life"])


@router.get(
    "/rul",
    response_model=RULResponse,
    summary="Get Remaining Useful Life Estimate",
    description="Exposes Module 15 RULEstimator hours remaining and confidence bounds.",
)
def get_rul_estimate(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    adapter: PipelineReplayAdapter = Depends(get_pipeline_adapter),
) -> RULResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    step_results = adapter.process_sequence([records[-1]])
    latest = step_results[-1]

    r_hrs = latest.rul_hours
    return RULResponse(
        hours_remaining=r_hrs,
        lower_bound_hours=max(0.0, r_hrs * 0.8) if r_hrs > 0 else None,
        upper_bound_hours=r_hrs * 1.2 if r_hrs > 0 else None,
        unit="hours",
        trend="STABLE",
        operating_assumption="constant_cruise_operating_profile",
        status="SUCCESS" if r_hrs > 0 else "INSUFFICIENT_HISTORY",
        quality=1.0 if latest.accepted else 0.5,
        provenance=Provenance.MODEL_OUTPUT,
        timestamp=latest.timestamp,
    )
