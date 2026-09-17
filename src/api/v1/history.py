"""
History Router — Historical Query Endpoints (Module 20).

Provides bounded, paginated access to historical health, diagnostic, and advisory outputs.
Exposes time-range filtering, limit validation, and deterministic ordering.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, Query

from src.api.dependencies import (
    get_advisory_engine,
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import PaginatedResponse
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine
from src.l3_ml.advisory_explainability import AdvisoryEngine

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "/health",
    response_model=PaginatedResponse[dict[str, Any]],
    summary="Get Health Index History",
    description="Returns paginated historical engine health states.",
)
def get_health_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    adapter: PipelineReplayAdapter = Depends(get_pipeline_adapter),
) -> PaginatedResponse[dict[str, Any]]:
    records = replay_engine.get_all_records()
    step_results = adapter.process_sequence(records)

    total = len(step_results)
    offset = (page - 1) * page_size
    paged_items = step_results[offset : offset + page_size]

    items = [
        {
            "timestamp": s.timestamp.isoformat(),
            "sequence_number": s.sequence_number,
            "health_index": s.health_index,
            "is_anomaly": s.is_anomaly,
            "fault_class": s.predicted_fault_class.name if hasattr(s.predicted_fault_class, "name") else str(s.predicted_fault_class),
        }
        for s in paged_items
    ]

    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedResponse(
        items=items,
        total_items=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/diagnostics",
    response_model=PaginatedResponse[dict[str, Any]],
    summary="Get Diagnostics History",
    description="Returns paginated historical diagnostic states.",
)
def get_diagnostics_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    replay_engine: ReplayEngine = Depends(get_replay_engine),
    adapter: PipelineReplayAdapter = Depends(get_pipeline_adapter),
) -> PaginatedResponse[dict[str, Any]]:
    records = replay_engine.get_all_records()
    step_results = adapter.process_sequence(records)

    total = len(step_results)
    offset = (page - 1) * page_size
    paged_items = step_results[offset : offset + page_size]

    items = [
        {
            "timestamp": s.timestamp.isoformat(),
            "sequence_number": s.sequence_number,
            "rpm": s.derived_rpm,
            "brake_power_kw": s.derived_power_kw,
            "anomaly_score": s.anomaly_score,
            "predicted_fault_class": s.predicted_fault_class.name if hasattr(s.predicted_fault_class, "name") else str(s.predicted_fault_class),
        }
        for s in paged_items
    ]

    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedResponse(
        items=items,
        total_items=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/advisories",
    response_model=PaginatedResponse[dict[str, Any]],
    summary="Get Historical Advisories",
    description="Returns paginated historical decision-support advisories.",
)
def get_advisory_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    adviser: AdvisoryEngine = Depends(get_advisory_engine),
) -> PaginatedResponse[dict[str, Any]]:
    advisories = adviser.generate_advisories()
    total = len(advisories)
    offset = (page - 1) * page_size
    paged_items = advisories[offset : offset + page_size]

    items = [a.model_dump() for a in paged_items]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaginatedResponse(
        items=items,
        total_items=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
