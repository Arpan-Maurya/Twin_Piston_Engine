"""
Replay Router — Module 18 Scenario Replay Endpoints (Module 20).

Exposes playback control over simulated telemetry scenario runs.
Delegates to Module 18 ReplayEngine without duplicating replay state logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_replay_engine
from src.api.v1.schemas import ReplayActionRequest, ReplayStateResponse
from src.core.provenance import Provenance
from src.l1_data.simulator.replay_and_whatif import ReplayEngine, ReplayStatus

router = APIRouter(prefix="/replay", tags=["Scenario Replay"])


def _to_response(engine: ReplayEngine) -> ReplayStateResponse:
    state = engine.get_state()
    return ReplayStateResponse(
        scenario_id=state.scenario_id,
        replay_position=state.replay_position,
        total_records=state.total_records,
        status=state.status.value if hasattr(state.status, "value") else str(state.status),
        current_timestamp=state.current_timestamp,
        start_time=state.start_time,
        end_time=state.end_time,
        sequence_number=state.sequence_number,
        speed_multiplier=state.speed_multiplier,
        provenance=Provenance.SIMULATED,
    )


@router.get(
    "/state",
    response_model=ReplayStateResponse,
    summary="Get Current Replay State",
    description="Returns current position, status, and metadata of scenario replay engine.",
)
def get_replay_state(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    return _to_response(replay_engine)


@router.post(
    "/start",
    response_model=ReplayStateResponse,
    summary="Start Scenario Replay",
    description="Starts playback of loaded scenario telemetry.",
)
def start_replay(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    if not replay_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No scenario loaded to replay.")
    replay_engine.play()
    return _to_response(replay_engine)


@router.post(
    "/pause",
    response_model=ReplayStateResponse,
    summary="Pause Scenario Replay",
    description="Pauses current scenario playback.",
)
def pause_replay(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    replay_engine.pause()
    return _to_response(replay_engine)


@router.post(
    "/resume",
    response_model=ReplayStateResponse,
    summary="Resume Scenario Replay",
    description="Resumes paused scenario playback.",
)
def resume_replay(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    replay_engine.resume()
    return _to_response(replay_engine)


@router.post(
    "/stop",
    response_model=ReplayStateResponse,
    summary="Stop Scenario Replay",
    description="Stops scenario playback and resets position to 0.",
)
def stop_replay(
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    replay_engine.stop()
    return _to_response(replay_engine)


@router.post(
    "/seek",
    response_model=ReplayStateResponse,
    summary="Seek Scenario Replay Position",
    description="Seeks scenario replay to specified sample index or timestamp.",
)
def seek_replay(
    request: ReplayActionRequest,
    replay_engine: ReplayEngine = Depends(get_replay_engine),
) -> ReplayStateResponse:
    if not replay_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No scenario loaded to seek.")

    try:
        if request.position is not None:
            replay_engine.seek(request.position)
        elif request.target_time is not None:
            replay_engine.seek_to_timestamp(request.target_time)
        else:
            raise HTTPException(status_code=400, detail="Must specify 'position' or 'target_time'.")
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return _to_response(replay_engine)
