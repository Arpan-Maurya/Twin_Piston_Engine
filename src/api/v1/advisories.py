"""
Advisories & Explainability Router — Module 19 Decision-Support Endpoints (Module 20).

Exposes advisories, structured explanations, and operator diagnostic query Q&A.
Calls Module 19 AdvisoryEngine, ExplainabilityEngine, and DiagnosticQueryService.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_advisory_engine,
    get_diagnostic_query_service,
    get_explainability_engine,
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import (
    AdvisoryResponse,
    DiagnosticQueryRequest,
    DiagnosticQueryResponse,
    ExplanationResponse,
)
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine
from src.l3_ml.advisory_explainability import (
    AdvisoryEngine,
    DiagnosticQueryService,
    ExplainabilityEngine,
    QuestionType,
)

router = APIRouter(tags=["Advisories & Explainability"])


@router.get(
    "/advisories",
    response_model=list[AdvisoryResponse],
    summary="Get Active Decision-Support Advisories",
    description="Exposes Module 19 AdvisoryEngine decision-support advisories.",
)
def get_advisories(
    advisory_engine: AdvisoryEngine = Depends(get_advisory_engine),
) -> list[AdvisoryResponse]:
    advisories = advisory_engine.generate_advisories()
    return [
        AdvisoryResponse(
            advisory_id=adv.advisory_id,
            category=adv.category.value if hasattr(adv.category, "value") else str(adv.category),
            priority=adv.priority.value if hasattr(adv.priority, "value") else str(adv.priority),
            title=adv.title,
            message=adv.message,
            subsystem=adv.subsystem,
            confidence=adv.confidence,
            provenance=adv.provenance,
            timestamp=adv.timestamp,
            limitations=adv.limitations,
        )
        for adv in advisories
    ]


@router.get(
    "/advisories/{advisory_id}",
    response_model=AdvisoryResponse,
    summary="Get Advisory by ID",
    description="Returns specific advisory details by advisory_id.",
)
def get_advisory_by_id(
    advisory_id: str,
    advisory_engine: AdvisoryEngine = Depends(get_advisory_engine),
) -> AdvisoryResponse:
    advisories = advisory_engine.generate_advisories()
    for adv in advisories:
        if adv.advisory_id == advisory_id:
            return AdvisoryResponse(
                advisory_id=adv.advisory_id,
                category=adv.category.value if hasattr(adv.category, "value") else str(adv.category),
                priority=adv.priority.value if hasattr(adv.priority, "value") else str(adv.priority),
                title=adv.title,
                message=adv.message,
                subsystem=adv.subsystem,
                confidence=adv.confidence,
                provenance=adv.provenance,
                timestamp=adv.timestamp,
                limitations=adv.limitations,
            )
    raise HTTPException(status_code=404, detail=f"Advisory '{advisory_id}' not found.")


@router.get(
    "/explanations",
    response_model=ExplanationResponse,
    summary="Get Structured Diagnostic Explanation",
    description="Exposes Module 19 ExplainabilityEngine explanations.",
)
def get_explanation(
    explainer: ExplainabilityEngine = Depends(get_explainability_engine),
) -> ExplanationResponse:
    exp = explainer.explain_fault(None)
    return ExplanationResponse(
        explanation_id=exp.explanation_id,
        finding=exp.finding,
        observation=exp.observation,
        interpretation=exp.interpretation,
        evidence=[ev.model_dump() for ev in exp.evidence],
        uncertainty=exp.uncertainty,
        limitation=exp.limitation,
        provenance=exp.provenance,
        quality=exp.quality,
        timestamp=exp.timestamp,
    )


@router.post(
    "/diagnostic/query",
    response_model=DiagnosticQueryResponse,
    summary="Operator Diagnostic Q&A Query",
    description="Answers operator queries (e.g. HEALTH_STATUS, FAULT_STATUS, RUL_ESTIMATE, WHAT_CHANGED).",
)
def query_diagnostic(
    request: DiagnosticQueryRequest,
    query_service: DiagnosticQueryService = Depends(get_diagnostic_query_service),
) -> DiagnosticQueryResponse:
    ans = query_service.answer_query(request.question_type)
    return DiagnosticQueryResponse(
        question_type=ans.question_type,
        answer=ans.answer,
        evidence=[ev.model_dump() for ev in ans.evidence],
        limitations=ans.limitations,
        quality=ans.quality,
        provenance=ans.provenance,
        timestamp=ans.timestamp,
    )
