"""
Module 23 — Performance Optimization and Resilience Package.

Provides CPU caching utilities, memory management bounds, performance tracking,
resilience and failure isolation wrappers, retry/timeout helpers, and reproducible benchmarking.
"""

from src.performance_resilience.benchmark import BenchmarkHarness, BenchmarkResults
from src.performance_resilience.cache import FastLRUCache, cache_deterministic
from src.performance_resilience.limits import ResourceLimitsConfig
from src.performance_resilience.resilience import (
    retry_with_backoff,
    safe_subsystem_call,
)
from src.performance_resilience.tracker import (
    PerformanceMetricsSnapshot,
    PerformanceTracker,
    tracker_instance,
)

__all__ = [
    "ResourceLimitsConfig",
    "PerformanceTracker",
    "PerformanceMetricsSnapshot",
    "tracker_instance",
    "FastLRUCache",
    "cache_deterministic",
    "retry_with_backoff",
    "safe_subsystem_call",
    "BenchmarkHarness",
    "BenchmarkResults",
]
