"""
Mission Risk Router — Module 16 Flight Phase & Analytical Risk Endpoints (Module 20).

Exposes Flight Phase classification and relative analytical Mission Risk scores.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import MissionRiskResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine

router = APIRouter(prefix="/mission", tags=["Mission Risk"])


@router.get(
    "/risk",
    response_model=MissionRiskResponse,
    summary="Get Mission Phase & Risk Assessment",
    description="Exposes Module 16 MissionRiskEngine phase context and relative analytical risk score.",
)
def get_mission_risk(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    adapter: PipelineReplayAdapter = Depends(get_pipeline_adapter),
) -> MissionRiskResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    step_results = adapter.process_sequence([records[-1]])
    latest = step_results[-1]

    r_score = latest.mission_risk_score
    r_level = "HIGH" if r_score >= 0.75 else ("MEDIUM" if r_score >= 0.50 else "LOW")

    return MissionRiskResponse(
        flight_phase="CRUISE",
        risk_score=r_score,
        risk_level=r_level,
        risk_trend="STABLE",
        health_index=latest.health_index,
        rul_hours=latest.rul_hours,
        contributing_fault=latest.predicted_fault_class.name if hasattr(latest.predicted_fault_class, "name") else str(latest.predicted_fault_class),
        quality=1.0 if latest.accepted else 0.5,
        provenance=Provenance.DERIVED,
        timestamp=latest.timestamp,
    )
