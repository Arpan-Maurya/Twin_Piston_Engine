"""
Edge Store-and-Forward Bounded Buffer — Module 21.

Implements bounded FIFO telemetry buffering for Edge onboard nodes.
Buffers valid telemetry packets when communication link to Ground station is unavailable.
Flushes buffered telemetry in FIFO sequence order upon link restoration.
Prevents unbounded memory growth with explicit overflow drop policies.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import threading
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import OverflowPolicy
from src.edge_ground.envelope import EdgeTelemetryEnvelope

logger = get_logger(__name__)


class EdgeStoreAndForwardBuffer:
    """Thread-safe bounded FIFO queue for store-and-forward telemetry buffering."""

    def __init__(
        self,
        max_capacity: int = 1000,
        overflow_policy: OverflowPolicy = OverflowPolicy.DISCARD_OLDEST,
    ) -> None:
        if max_capacity <= 0:
            raise ValueError("Buffer max_capacity must be > 0")

        self._max_capacity = max_capacity
        self._overflow_policy = overflow_policy
        self._queue: deque[EdgeTelemetryEnvelope] = deque()
        self._lock = threading.Lock()

        # Metrics & telemetry freshness metadata
        self._dropped_count: int = 0
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._last_transmission_timestamp: datetime | None = None

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    @property
    def overflow_policy(self) -> OverflowPolicy:
        return self._overflow_policy

    def enqueue(self, envelope: EdgeTelemetryEnvelope) -> bool:
        """Enqueue an EdgeTelemetryEnvelope into the buffer.

        If buffer capacity is reached:
        - DISCARD_OLDEST: Pops head (oldest), increments dropped count, and appends incoming envelope.
        - DISCARD_NEWEST: Rejects incoming envelope and increments dropped count.

        Returns True if the envelope is stored, False if dropped.
        """
        with self._lock:
            if len(self._queue) >= self._max_capacity:
                self._dropped_count += 1
                if self._overflow_policy == OverflowPolicy.DISCARD_OLDEST:
                    dropped_env = self._queue.popleft()
                    logger.warning(
                        f"Buffer overflow ({self._max_capacity}): Discarded oldest packet (seq={dropped_env.sequence_number})."
                    )
                    self._queue.append(envelope)
                    self._total_enqueued += 1
                    return True
                else:
                    logger.warning(
                        f"Buffer overflow ({self._max_capacity}): Discarded incoming packet (seq={envelope.sequence_number})."
                    )
                    return False
            else:
                self._queue.append(envelope)
                self._total_enqueued += 1
                return True

    def dequeue(self) -> EdgeTelemetryEnvelope | None:
        """Pop and return the oldest (FIFO head) envelope from the buffer."""
        with self._lock:
            if not self._queue:
                return None
            env = self._queue.popleft()
            self._total_dequeued += 1
            return env

    def peek(self) -> EdgeTelemetryEnvelope | None:
        """View the oldest envelope in the buffer without removing it."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def peek_batch(self, batch_size: int) -> list[EdgeTelemetryEnvelope]:
        """View up to batch_size envelopes without dequeuing."""
        with self._lock:
            return list(self._queue)[:batch_size]

    def dequeue_batch(self, batch_size: int) -> list[EdgeTelemetryEnvelope]:
        """Pop and return up to batch_size envelopes in FIFO sequence."""
        with self._lock:
            batch = []
            for _ in range(min(batch_size, len(self._queue))):
                batch.append(self._queue.popleft())
                self._total_dequeued += 1
            return batch

    def clear(self) -> None:
        """Clear all buffered envelopes."""
        with self._lock:
            self._queue.clear()

    def record_successful_transmission(self, timestamp: datetime | None = None) -> None:
        """Record timestamp of last successful transmission to Ground."""
        with self._lock:
            self._last_transmission_timestamp = timestamp or datetime.now(timezone.utc)

    def get_freshness_metadata(self) -> dict[str, Any]:
        """Return telemetry freshness and buffer queue status metadata."""
        with self._lock:
            queue_depth = len(self._queue)
            oldest_ts = self._queue[0].timestamp.isoformat() if self._queue else None
            newest_ts = self._queue[-1].timestamp.isoformat() if self._queue else None
            last_sent_ts = (
                self._last_transmission_timestamp.isoformat()
                if self._last_transmission_timestamp
                else None
            )

            return {
                "queue_depth": queue_depth,
                "max_capacity": self._max_capacity,
                "overflow_policy": self._overflow_policy.value,
                "dropped_packet_count": self._dropped_count,
                "total_enqueued": self._total_enqueued,
                "total_dequeued": self._total_dequeued,
                "oldest_buffered_timestamp": oldest_ts,
                "newest_buffered_timestamp": newest_ts,
                "last_successful_transmission": last_sent_ts,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
