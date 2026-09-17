"""
Diagnostics Router — Modules 7–10 Physics Diagnostic Endpoints (Module 20).

Calls existing L2 Digital Twin diagnostic engines (EGT, Lubrication, Vibration, Misfire).
DOES NOT duplicate diagnostic calculations inside route functions.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import (
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import DiagnosticSummaryResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine

# L2 Digital Twin imports
from src.l2_digital_twin import (
    convert_raw_to_engineering_state,
    evaluate_digital_twin,
    evaluate_egt_diagnostics,
    evaluate_lubrication_model,
    evaluate_misfire_detector,
    evaluate_vibration_processor,
)

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


def _to_dict(obj: Any) -> Any:
    """Helper to convert Pydantic models, dataclasses, dicts, or tuples safely."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, tuple):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return str(obj) if not isinstance(obj, (int, float, bool, type(None))) else obj


@router.get(
    "",
    response_model=DiagnosticSummaryResponse,
    summary="Get Summary Physics Diagnostics",
    description="Exposes aggregated Module 6–10 Digital Twin physics states.",
)
def get_diagnostics_summary(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> DiagnosticSummaryResponse:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    rec = records[-1]
    norm_rec = convert_raw_to_engineering_state(rec)
    twin_res = evaluate_digital_twin(norm_rec)
    diag_state, egt_res = evaluate_egt_diagnostics(norm_rec)
    lub_state, lub_res = evaluate_lubrication_model(norm_rec)
    vib_state, vib_res = evaluate_vibration_processor(norm_rec)
    mis_out = evaluate_misfire_detector(norm_rec, egt_diag=egt_res, vib_diag=vib_res)

    mis_res = mis_out[1] if isinstance(mis_out, tuple) else mis_out
    mis_detected = getattr(mis_res, "misfire_detected", [False]*4)
    if isinstance(mis_detected, bool):
        mis_detected = [mis_detected]*4

    return DiagnosticSummaryResponse(
        rpm=norm_rec.rpm.value if (norm_rec.rpm.valid and norm_rec.rpm.value is not None) else 0.0,
        map_pa=norm_rec.map_pressure.value if (norm_rec.map_pressure.valid and norm_rec.map_pressure.value is not None) else 0.0,
        egt_mean_k=egt_res.mean_egt_k if hasattr(egt_res, "mean_egt_k") and egt_res.mean_egt_k is not None else (egt_res.egt_mean_k or 0.0 if hasattr(egt_res, "egt_mean_k") else 0.0),
        egt_spread_k=egt_res.spread_egt_k if hasattr(egt_res, "spread_egt_k") and egt_res.spread_egt_k is not None else (egt_res.egt_spread_k or 0.0 if hasattr(egt_res, "egt_spread_k") else 0.0),
        oil_temp_k=lub_state.oil_temperature_k.value if (lub_state.oil_temperature_k.valid and lub_state.oil_temperature_k.value is not None) else 0.0,
        oil_pressure_pa=lub_state.oil_pressure_pa.value if (lub_state.oil_pressure_pa.valid and lub_state.oil_pressure_pa.value is not None) else 0.0,
        vibration_rms_m_s2=vib_state.overall_rms_m_s2.value if (vib_state.overall_rms_m_s2.valid and vib_state.overall_rms_m_s2.value is not None) else 0.0,
        misfire_detected=list(mis_detected),
        overall_status=egt_res.overall_status.name if hasattr(egt_res.overall_status, "name") else str(getattr(egt_res, "overall_status", "NORMAL")),
        timestamp=rec.timestamp,
        provenance=Provenance.DERIVED,
    )


@router.get("/egt", response_model=dict[str, Any], summary="Get Module 7 Per-Cylinder EGT Diagnostics")
def get_egt_diagnostics(replay_engine: ReplayEngine = Depends(get_replay_engine)) -> dict[str, Any]:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    norm_rec = convert_raw_to_engineering_state(records[-1])
    diag_state, egt_res = evaluate_egt_diagnostics(norm_rec)
    return {"state": _to_dict(diag_state), "result": _to_dict(egt_res)}


@router.get("/lubrication", response_model=dict[str, Any], summary="Get Module 8 Lubrication Diagnostics")
def get_lubrication_diagnostics(replay_engine: ReplayEngine = Depends(get_replay_engine)) -> dict[str, Any]:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    norm_rec = convert_raw_to_engineering_state(records[-1])
    state, res = evaluate_lubrication_model(norm_rec)
    return {"state": _to_dict(state), "metrics": _to_dict(res)}


@router.get("/vibration", response_model=dict[str, Any], summary="Get Module 9 Vibration Features")
def get_vibration_diagnostics(replay_engine: ReplayEngine = Depends(get_replay_engine)) -> dict[str, Any]:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    norm_rec = convert_raw_to_engineering_state(records[-1])
    state, res = evaluate_vibration_processor(norm_rec)
    return {"state": _to_dict(state), "metrics": _to_dict(res)}


@router.get("/combustion", response_model=dict[str, Any], summary="Get Module 10 Misfire Diagnostics")
def get_combustion_diagnostics(replay_engine: ReplayEngine = Depends(get_replay_engine)) -> dict[str, Any]:
    records = replay_engine.get_all_records()
    if not records:
        raise HTTPException(status_code=404, detail="No telemetry available.")
    norm_rec = convert_raw_to_engineering_state(records[-1])
    diag_state, egt_res = evaluate_egt_diagnostics(norm_rec)
    vib_state, vib_res = evaluate_vibration_processor(norm_rec)
    mis_out = evaluate_misfire_detector(norm_rec, egt_diag=egt_res, vib_diag=vib_res)
    if isinstance(mis_out, tuple):
        return {"state": _to_dict(mis_out[0]), "result": _to_dict(mis_out[1])}
    return {"result": _to_dict(mis_out)}
