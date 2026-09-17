"""
Integration tests for Original Module 20 WebSocket Real-Time Interface (/api/v1/ws/engine).
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture(scope="module")
def ws_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestWebSocketInterface:
    """Test suite for real-time engine status streaming over WebSockets."""

    def test_websocket_connection_and_initial_handshake(self, ws_client):
        with ws_client.websocket_connect("/api/v1/ws/engine") as websocket:
            data = websocket.receive_json()
            assert "event_type" in data
            assert data["event_type"] == "system_status"
            assert data["payload"]["status"] == "CONNECTED"
            assert "sequence_number" in data

    def test_websocket_event_streaming(self, ws_client):
        with ws_client.websocket_connect("/api/v1/ws/engine") as websocket:
            init_msg = websocket.receive_json()
            assert init_msg["event_type"] == "system_status"

            # Receive streaming telemetry_update and health_update envelopes
            event1 = websocket.receive_json()
            assert "event_type" in event1
            assert event1["event_type"] in ("telemetry_update", "health_update")
            assert "payload" in event1

            event2 = websocket.receive_json()
            assert "event_type" in event2
            assert event2["event_type"] in ("telemetry_update", "health_update")
            assert "payload" in event2
