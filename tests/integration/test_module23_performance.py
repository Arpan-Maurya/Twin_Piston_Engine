"""
Integration tests for Module 23 — Performance & Throughput.
"""

import pytest
import time
from src.performance_resilience.benchmark import BenchmarkHarness
from src.performance_resilience.cache import cache_deterministic


def test_pipeline_throughput_performance():
    harness = BenchmarkHarness()
    results = harness.run_benchmark(num_records=50)

    # Ingestion latency should be under 5ms per record
    assert results.telemetry_ingestion_latency_ms < 5.0
    # End-to-end pipeline latency should be under 50ms per record
    assert results.pipeline_total_latency_ms < 50.0
    # Replay throughput should exceed 1,000 records/sec
    assert results.replay_throughput_records_per_sec > 1000.0


def test_cached_computation_speedup():
    call_count = 0

    @cache_deterministic(maxsize=128)
    def compute_heavy_physics_lookup(param1: float, param2: float) -> float:
        nonlocal call_count
        call_count += 1
        time.sleep(0.005)  # Simulate expensive deterministic math calculation
        return param1 * 2.5 + param2 * 1.8

    # First call (uncached)
    t0 = time.perf_counter()
    r1 = compute_heavy_physics_lookup(12.0, 45.0)
    t1 = time.perf_counter()
    duration_uncached = t1 - t0

    # Second call (cached)
    t2 = time.perf_counter()
    r2 = compute_heavy_physics_lookup(12.0, 45.0)
    t3 = time.perf_counter()
    duration_cached = t3 - t2

    assert r1 == r2
    assert call_count == 1
    # Cached call should be significantly faster (at least 5x faster)
    assert duration_cached < (duration_uncached / 5.0)
