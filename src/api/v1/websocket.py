"""
WebSocket Real-Time Router — /api/v1/ws/engine Stream Endpoint (Module 20 & 23).

Provides structured real-time diagnostic event streaming.
DISCONNECT-SAFE, EXCEPTION-ISOLATED, READ-ONLY DIAGNOSTIC STREAM.
RESILIENT: Bounded client capacity, slow-client isolation, non-blocking broadcast timeout.
DOES NOT LEAK RAW SIMULATOR GROUND TRUTH OR PRIVATE INTERNALS.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.dependencies import (
    get_advisory_engine,
    get_pipeline_adapter,
    get_replay_engine,
)
from src.api.v1.schemas import EventEnvelope
from src.core.logging import get_logger
from src.core.provenance import Provenance
from src.performance_resilience.tracker import tracker_instance

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["Real-Time Streaming"])


class ConnectionManager:
    """Resilient read-only WebSocket connection manager with slow-client isolation."""

    def __init__(self, max_clients: int = 100, send_timeout_s: float = 2.0) -> None:
        self.active_connections: list[WebSocket] = []
        self._max_clients = max_clients
        self._send_timeout_s = send_timeout_s
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self.active_connections) >= self._max_clients:
                logger.warning(f"WebSocket client connection rejected: Max capacity ({self._max_clients}) reached.")
                await websocket.close(code=1013, reason="Server busy / Max connections reached")
                return False

            await websocket.accept()
            self.active_connections.append(websocket)
            tracker_instance.set_active_ws_clients(len(self.active_connections))
            logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")
            return True

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            tracker_instance.set_active_ws_clients(len(self.active_connections))
            logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def send_envelope(self, websocket: WebSocket, envelope: EventEnvelope) -> None:
        """Send envelope to a single WebSocket client with timeout isolation."""
        try:
            payload_json = envelope.model_dump(mode="json")
            await asyncio.wait_for(
                websocket.send_json(payload_json),
                timeout=self._send_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSocket send timeout (slow client). Disconnecting client.")
            tracker_instance.increment_dropped_events()
            self.disconnect(websocket)
        except Exception as e:
            logger.warning(f"Failed to send envelope to WebSocket client: {e}")
            tracker_instance.increment_dropped_events()
            self.disconnect(websocket)


manager = ConnectionManager()


@router.websocket("/engine")
async def websocket_engine_endpoint(websocket: WebSocket) -> None:
    """Real-time engine diagnostic event streaming endpoint."""
    connected = await manager.connect(websocket)
    if not connected:
        return

    replay_engine = get_replay_engine()
    adapter = get_pipeline_adapter()

    try:
        # Emit initial system status event envelope
        init_envelope = EventEnvelope(
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            sequence_number=0,
            payload={"status": "CONNECTED", "stream": "realtime_engine_telemetry"},
            quality=1.0,
            provenance=Provenance.DERIVED,
        )
        await manager.send_envelope(websocket, init_envelope)

        records = replay_engine.get_all_records()
        if not records:
            close_envelope = EventEnvelope(
                event_type="system_status",
                timestamp=datetime.now(timezone.utc),
                sequence_number=0,
                payload={"status": "NO_TELEMETRY_AVAILABLE"},
                quality=0.0,
                provenance=Provenance.DERIVED,
            )
            await manager.send_envelope(websocket, close_envelope)
            await websocket.close()
            return

        step_results = adapter.process_sequence(records[:5])

        for step in step_results:
            if websocket not in manager.active_connections:
                break

            # 1. Telemetry event envelope
            t_env = EventEnvelope(
                event_type="telemetry_update",
                timestamp=step.timestamp,
                sequence_number=step.sequence_number,
                payload={
                    "rpm": step.derived_rpm,
                    "power_kw": step.derived_power_kw,
                },
                quality=1.0 if step.accepted else 0.5,
                provenance=Provenance.SIMULATED,
            )
            await manager.send_envelope(websocket, t_env)

            # 2. Health event envelope
            h_env = EventEnvelope(
                event_type="health_update",
                timestamp=step.timestamp,
                sequence_number=step.sequence_number,
                payload={
                    "health_index": step.health_index,
                    "is_anomaly": step.is_anomaly,
                    "anomaly_score": step.anomaly_score,
                    "predicted_fault_class": step.predicted_fault_class.name if hasattr(step.predicted_fault_class, "name") else str(step.predicted_fault_class),
                    "rul_hours": step.rul_hours,
                    "mission_risk_score": step.mission_risk_score,
                },
                quality=1.0 if step.accepted else 0.5,
                provenance=Provenance.DERIVED,
            )
            await manager.send_envelope(websocket, h_env)

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect event caught cleanly.")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket handler exception: {e}")
        manager.disconnect(websocket)
