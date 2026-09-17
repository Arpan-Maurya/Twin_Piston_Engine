"""
Integration tests for Module 23 — Resource Limits & Capacity Bounding.
"""

import pytest
import asyncio
from src.performance_resilience.limits import ResourceLimitsConfig
from src.performance_resilience.cache import FastLRUCache
from src.api.v1.websocket import ConnectionManager


def test_resource_limits_config_bounds():
    limits = ResourceLimitsConfig(
        max_edge_buffer_capacity=50,
        max_websocket_clients=5,
        max_websocket_client_queue_size=10,
    )
    assert limits.max_telemetry_buffer_size == 50
    assert limits.max_websocket_clients == 5
    assert limits.max_ws_queue_per_client == 10


def test_lru_cache_bounded_memory():
    cache = FastLRUCache(maxsize=100)
    for i in range(500):
        cache.set(f"key_{i}", f"value_{i}")

    assert cache.size == 100
    assert cache.get("key_0") is None  # Evicted
    assert cache.get("key_499") == "value_499"  # Retained


@pytest.mark.asyncio
async def test_websocket_capacity_limit():
    manager = ConnectionManager(max_clients=2)

    class DummyWebSocket:
        def __init__(self, name: str):
            self.name = name
            self.sent = []

        async def accept(self):
            pass

        async def close(self, code: int = 1000, reason: str = ""):
            pass

        async def send_text(self, text: str):
            self.sent.append(text)

    ws1 = DummyWebSocket("ws1")
    ws2 = DummyWebSocket("ws2")
    ws3 = DummyWebSocket("ws3")

    assert await manager.connect(ws1) is True
    assert await manager.connect(ws2) is True
    # 3rd connection should be rejected due to capacity limit (max_clients=2)
    assert await manager.connect(ws3) is False

    assert len(manager.active_connections) == 2

    # Disconnect one
    manager.disconnect(ws1)
    assert len(manager.active_connections) == 1

    # Now ws3 can connect
    assert await manager.connect(ws3) is True
    assert len(manager.active_connections) == 2
