"""
Edge Telemetry Onboard Node — Module 21.

Edge/Onboard deployment node implementation.
Responsibilities:
1. Receives raw telemetry packets from sensor hardware/source adapters.
2. Validates structural, range, staleness, and HMAC integrity checks (Module 3 TelemetryValidator).
3. Buffers valid packets in bounded store-and-forward FIFO buffer when link is down.
4. Transmits telemetry envelopes to Ground station via ITelemetryTransport abstraction.
5. Manages link state transitions (CONNECTED, DEGRADED, DISCONNECTED, BUFFERING, RECOVERING).
6. Exposes communication freshness and buffering health metadata.

STRICT BOUNDARY INVARIANTS:
- MUST NOT initialize ML models, Digital Twin physics, RUL, Mission Risk, Advisory Engine, or What-If engines.
- MUST NOT expose actuator or engine control commands.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.exceptions import RoleViolationError
from src.core.logging import get_logger
from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy
from src.edge_ground.buffer import EdgeStoreAndForwardBuffer
from src.edge_ground.envelope import EdgeTelemetryEnvelope
from src.edge_ground.transport import ITelemetryTransport, InMemoryTelemetryTransport
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator, ValidationResult

logger = get_logger(__name__)


class EdgeTelemetryNode:
    """Edge / Onboard Telemetry Acquisition, Validation, and Transport Node."""

    def __init__(
        self,
        node_id: str = "edge_node_01",
        settings: AppSettings | None = None,
        transport: ITelemetryTransport | None = None,
        validator: TelemetryValidator | None = None,
        signer: PacketSigner | None = None,
        buffer_capacity: int = 1000,
        overflow_policy: OverflowPolicy = OverflowPolicy.DISCARD_OLDEST,
    ) -> None:
        self._settings = settings or get_settings()

        # Role enforcement guard
        if self._settings.deployment.role == DeploymentRole.GROUND:
            raise RoleViolationError(
                "EdgeTelemetryNode initialized on a node configured with DeploymentRole.GROUND."
            )

        self._node_id = node_id
        self._transport = transport or InMemoryTelemetryTransport()
        self._validator = validator or TelemetryValidator(self._settings)
        self._signer = signer or PacketSigner(self._settings)
        self._buffer = EdgeStoreAndForwardBuffer(
            max_capacity=buffer_capacity or self._settings.deployment.buffer_capacity,
            overflow_policy=overflow_policy or self._settings.deployment.buffer_overflow_policy,
        )

        self._comm_state: CommunicationState = CommunicationState.DISCONNECTED
        self._last_successful_send: datetime | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def buffer(self) -> EdgeStoreAndForwardBuffer:
        return self._buffer

    @property
    def transport(self) -> ITelemetryTransport:
        return self._transport

    def get_communication_state(self) -> CommunicationState:
        """Return current edge node communication link health state."""
        transport_status = self._transport.get_status()
        if transport_status == CommunicationState.DISCONNECTED:
            if len(self._buffer) > 0:
                return CommunicationState.BUFFERING
            return CommunicationState.DISCONNECTED
        return transport_status

    def process_telemetry_packet(self, record: RawSignalRecord) -> ValidationResult:
        """Acquire, validate, buffer, and attempt transmission of a raw signal record.

        Pipeline order:
        1. Sign record if unsigned.
        2. Validate packet via Module 3 TelemetryValidator (integrity, sequence, stale, range).
        3. If rejected: drop record and return rejection result.
        4. If accepted: wrap into EdgeTelemetryEnvelope and enqueue into store-and-forward buffer.
        5. Attempt buffer flush over transport.
        """
        signature = self._signer.sign_record(record)
        val_result = self._validator.validate_packet(record, signature=signature)

        if not val_result.accepted or val_result.record is None:
            logger.warning(
                f"Edge packet rejected (seq={record.sequence_number}): {val_result.rejection_reason}"
            )
            return val_result

        # Create envelope and enqueue into store-and-forward buffer
        envelope = EdgeTelemetryEnvelope.from_record(
            record=val_result.record,
            signature=signature,
            node_id=self._node_id,
        )
        self._buffer.enqueue(envelope)

        # Attempt buffer flush
        self.flush_buffer()
        return val_result

    def flush_buffer(self) -> tuple[int, int]:
        """Attempt to flush buffered envelopes to Ground station in FIFO order.

        Returns (success_count, failed_count).
        """
        if len(self._buffer) == 0:
            return 0, 0

        # Attempt to send peeked batch
        batch = self._buffer.peek_batch(50)
        success_count = 0
        failed_count = 0

        for env in batch:
            sent = self._transport.send_telemetry(env)
            if sent:
                self._buffer.dequeue()  # Remove dequeued envelope
                success_count += 1
                self._last_successful_send = datetime.now(timezone.utc)
                self._buffer.record_successful_transmission(self._last_successful_send)
                self._comm_state = CommunicationState.CONNECTED
            else:
                failed_count += 1
                self._comm_state = CommunicationState.BUFFERING
                break  # Stop flushing on transmission failure to preserve FIFO ordering

        return success_count, failed_count

    def get_freshness_metadata(self) -> dict[str, Any]:
        """Return communication link health and telemetry freshness metadata."""
        freshness = self._buffer.get_freshness_metadata()
        freshness["node_id"] = self._node_id
        freshness["communication_state"] = self.get_communication_state().value
        freshness["last_successful_send"] = (
            self._last_successful_send.isoformat() if self._last_successful_send else None
        )
        return freshness
