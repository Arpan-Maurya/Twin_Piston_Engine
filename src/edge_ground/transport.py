"""
Communication Abstraction & Transport Layer — Module 21.

Defines the clean transport interface between Edge onboard nodes and Ground station receivers.
Supports:
- ITelemetryTransport Protocol / Base Class
- InMemoryTelemetryTransport (For simulation & fast deterministic testing)
- HTTPTelemetryTransport (For sending packets to Ground REST API)
- CommunicationState tracking (CONNECTED, DEGRADED, DISCONNECTED, BUFFERING, RECOVERING)
- Automatic retries, timeouts, and failure isolation
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable
from datetime import datetime, timezone

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import CommunicationState
from src.edge_ground.envelope import EdgeTelemetryEnvelope

logger = get_logger(__name__)


@runtime_checkable
class ITelemetryTransport(Protocol):
    """Protocol abstraction for Edge -> Ground communication transport."""

    def connect(self) -> bool:
        """Establish or verify transport connection to Ground station."""
        ...

    def disconnect(self) -> None:
        """Close transport link to Ground station."""
        ...

    def send_telemetry(self, envelope: EdgeTelemetryEnvelope) -> bool:
        """Transmit a single EdgeTelemetryEnvelope to Ground station."""
        ...

    def send_batch(self, envelopes: list[EdgeTelemetryEnvelope]) -> tuple[int, int]:
        """Transmit a batch of EdgeTelemetryEnvelopes. Returns (success_count, failed_count)."""
        ...

    def get_status(self) -> CommunicationState:
        """Get current communication link health status."""
        ...


class InMemoryTelemetryTransport:
    """In-Memory transport adapter for fast deterministic testing and simulation."""

    def __init__(self, simulate_network_failure: bool = False) -> None:
        self._connected: bool = True
        self._simulate_failure: bool = simulate_network_failure
        self._received_envelopes: list[EdgeTelemetryEnvelope] = []
        self._state: CommunicationState = CommunicationState.CONNECTED

    @property
    def received_envelopes(self) -> list[EdgeTelemetryEnvelope]:
        return list(self._received_envelopes)

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        self._state = CommunicationState.CONNECTED if connected else CommunicationState.DISCONNECTED

    def set_simulate_failure(self, fail: bool) -> None:
        self._simulate_failure = fail
        if fail:
            self._state = CommunicationState.DEGRADED

    def connect(self) -> bool:
        if self._simulate_failure:
            self._state = CommunicationState.DISCONNECTED
            return False
        self._connected = True
        self._state = CommunicationState.CONNECTED
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._state = CommunicationState.DISCONNECTED

    def send_telemetry(self, envelope: EdgeTelemetryEnvelope) -> bool:
        if not self._connected or self._simulate_failure:
            self._state = CommunicationState.DISCONNECTED
            return False
        self._received_envelopes.append(envelope)
        self._state = CommunicationState.CONNECTED
        return True

    def send_batch(self, envelopes: list[EdgeTelemetryEnvelope]) -> tuple[int, int]:
        if not self._connected or self._simulate_failure:
            self._state = CommunicationState.DISCONNECTED
            return 0, len(envelopes)

        success = 0
        failed = 0
        for env in envelopes:
            if self.send_telemetry(env):
                success += 1
            else:
                failed += 1
        return success, failed

    def get_status(self) -> CommunicationState:
        return self._state


class HTTPTelemetryTransport:
    """HTTP/REST client transport adapter connecting Edge to Ground API (/api/v1/telemetry)."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8000/api/v1/telemetry",
        timeout_s: float = 5.0,
        retry_attempts: int = 3,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_backoff_s = retry_backoff_s
        self._state: CommunicationState = CommunicationState.DISCONNECTED
        self._connected: bool = False

    def connect(self) -> bool:
        # Check endpoint reachability using httpx or requests
        try:
            import httpx
            health_url = self._endpoint_url.rsplit("/", 2)[0] + "/system/health"
            response = httpx.get(health_url, timeout=self._timeout_s)
            if response.status_code == 200:
                self._connected = True
                self._state = CommunicationState.CONNECTED
                return True
        except Exception:
            pass

        self._connected = False
        self._state = CommunicationState.DISCONNECTED
        return False

    def disconnect(self) -> None:
        self._connected = False
        self._state = CommunicationState.DISCONNECTED

    def send_telemetry(self, envelope: EdgeTelemetryEnvelope) -> bool:
        try:
            import httpx
            payload = {
                "sequence_number": envelope.sequence_number,
                "source_type": envelope.source_type.value if hasattr(envelope.source_type, "value") else str(envelope.source_type),
                "egt_cyl1_hot_uv": envelope.record.egt_cyl1_hot_uv,
                "egt_cyl2_hot_uv": envelope.record.egt_cyl2_hot_uv,
                "egt_cyl3_hot_uv": envelope.record.egt_cyl3_hot_uv,
                "egt_cyl4_hot_uv": envelope.record.egt_cyl4_hot_uv,
                "egt_cold_c": envelope.record.egt_cold_c,
                "cht_hot_uv": envelope.record.cht_hot_uv,
                "cht_cold_c": envelope.record.cht_cold_c,
                "oil_rtd_ohms": envelope.record.oil_rtd_ohms,
                "oil_p_counts": envelope.record.oil_p_counts,
                "map_counts": envelope.record.map_counts,
                "adc_vref_counts": envelope.record.adc_vref_counts,
                "crank_period_us": envelope.record.crank_period_us,
                "fuel_pulse_hz": envelope.record.fuel_pulse_hz,
                "accel_counts_xyz": list(envelope.record.accel_counts_xyz),
                "ambient_temp_c": envelope.record.ambient_temp_c,
                "ambient_press_pa": envelope.record.ambient_press_pa,
                "signature": envelope.signature,
            }

            for attempt in range(self._retry_attempts):
                try:
                    res = httpx.post(self._endpoint_url, json=payload, timeout=self._timeout_s)
                    if res.status_code in (200, 201):
                        self._connected = True
                        self._state = CommunicationState.CONNECTED
                        return True
                except Exception as ex:
                    logger.debug(f"HTTP transport send attempt {attempt + 1} failed: {ex}")
                    if attempt < self._retry_attempts - 1:
                        time.sleep(self._retry_backoff_s)

        except Exception as e:
            logger.warning(f"HTTP transport exception: {e}")

        self._connected = False
        self._state = CommunicationState.DISCONNECTED
        return False

    def send_batch(self, envelopes: list[EdgeTelemetryEnvelope]) -> tuple[int, int]:
        success = 0
        failed = 0
        for env in envelopes:
            if self.send_telemetry(env):
                success += 1
            else:
                failed += 1
        return success, failed

    def get_status(self) -> CommunicationState:
        return self._state
