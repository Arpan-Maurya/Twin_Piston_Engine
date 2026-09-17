"""
Unit tests for Original Module 23 — Performance Optimization & Resilience.
"""

import pytest
import time
from src.performance_resilience.limits import ResourceLimitsConfig
from src.performance_resilience.tracker import PerformanceTracker
from src.performance_resilience.cache import FastLRUCache, cache_deterministic
from src.performance_resilience.resilience import retry_with_backoff, safe_subsystem_call
from src.performance_resilience.benchmark import BenchmarkHarness


def test_resource_limits_default_config():
    limits = ResourceLimitsConfig()
    assert limits.max_telemetry_buffer_size == 10000
    assert limits.max_websocket_clients == 100
    assert limits.max_ws_queue_per_client == 100
    assert limits.request_timeout_s == 10.0


def test_performance_tracker_metrics():
    tracker = PerformanceTracker()
    tracker.record_latency("ingestion", 15.5)
    tracker.record_latency("ingestion", 24.5)
    tracker.record_retry("db_query")
    tracker.record_error("ws_connect")

    snapshot = tracker.get_snapshot()
    assert snapshot.telemetry_ingestion_latency_ms == 20.0
    assert snapshot.retry_count == 1
    assert snapshot.error_count == 1


def test_fast_lru_cache():
    cache = FastLRUCache(maxsize=2)
    cache.set("a", 100)
    cache.set("b", 200)
    assert cache.get("a") == 100
    assert cache.get("b") == 200

    # Trigger eviction
    cache.set("c", 300)
    assert cache.get("a") is None  # 'a' was LRU
    assert cache.get("b") == 200
    assert cache.get("c") == 300
    assert cache.size == 2


def test_cache_deterministic_decorator():
    call_count = 0

    @cache_deterministic(maxsize=10)
    def calculate_static_twin_geometry(bore: float, stroke: float) -> float:
        nonlocal call_count
        call_count += 1
        return bore * stroke * 3.14159

    res1 = calculate_static_twin_geometry(85.0, 78.0)
    res2 = calculate_static_twin_geometry(85.0, 78.0)
    assert res1 == res2
    assert call_count == 1  # Second call served from cache


def test_retry_with_backoff_success():
    attempts = 0

    @retry_with_backoff(max_retries=3, initial_delay_s=0.01)
    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("Transient error")
        return "SUCCESS"

    result = flaky_func()
    assert result == "SUCCESS"
    assert attempts == 2


def test_retry_with_backoff_exhaustion():
    attempts = 0

    @retry_with_backoff(max_retries=2, initial_delay_s=0.01)
    def failing_func():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Persistent failure")

    with pytest.raises(RuntimeError, match="Persistent failure"):
        failing_func()
    assert attempts == 2


def test_safe_subsystem_call_success():
    val = safe_subsystem_call("ml_subsystem", lambda: 42, fallback=0)
    assert val == 42


def test_safe_subsystem_call_failure():
    def broken():
        raise FileNotFoundError("Model weights missing")

    val = safe_subsystem_call("ml_subsystem", broken, fallback=-1)
    assert val == -1


def test_benchmark_harness_run():
    harness = BenchmarkHarness()
    res = harness.run_benchmark(num_records=10)
    assert res.sample_records_count == 10
    assert res.telemetry_ingestion_latency_ms >= 0.0
    assert res.digital_twin_latency_ms >= 0.0
    assert res.pipeline_total_latency_ms >= 0.0
    assert res.replay_throughput_records_per_sec > 0.0
