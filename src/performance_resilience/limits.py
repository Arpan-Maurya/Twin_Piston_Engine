"""
Resource Limits Configuration — Module 23.

Defines explicit, configurable maximum bounds for memory protection,
WebSocket client queues, history queries, and scenario execution to prevent unbounded resource growth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceLimitsConfig(BaseModel):
    """Resource limits and maximum capacity bounds."""

    max_telemetry_repository_records: int = Field(default=100000, gt=0)
    max_edge_buffer_capacity: int = Field(default=10000, gt=0)
    max_websocket_clients: int = Field(default=100, gt=0)
    max_websocket_client_queue_size: int = Field(default=100, gt=0)
    max_history_page_size: int = Field(default=1000, gt=0)
    max_replay_records_per_run: int = Field(default=50000, gt=0)
    max_what_if_duration_s: float = Field(default=600.0, gt=0.0)
    max_cache_entries: int = Field(default=1024, gt=0)
    request_timeout_s: float = Field(default=10.0, gt=0.0)

    @property
    def max_telemetry_buffer_size(self) -> int:
        return self.max_edge_buffer_capacity

    @property
    def max_ws_queue_per_client(self) -> int:
        return self.max_websocket_client_queue_size

