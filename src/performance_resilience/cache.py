"""
Thread-Safe Calculation & Configuration Cache — Module 23.

Provides thread-safe, bounded caching for immutable configuration, engine geometry,
and static reference values.
STRICT RULE: Dynamic telemetry, live health states, RUL, and advisories are NEVER cached.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class FastLRUCache:
    """Thread-safe bounded LRU cache for deterministic functions."""

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._cache: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                val = self._cache.pop(key)
                self._cache[key] = val
                return val
            self._misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._maxsize:
                # Evict oldest (first inserted)
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def set(self, key: Any, value: Any) -> None:
        self.put(key, value)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hit_ratio(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0


def cache_deterministic(maxsize: int = 1024) -> Callable:
    """Decorator for caching pure, deterministic calculations."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        lru = FastLRUCache(maxsize=maxsize)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create hashable cache key
            key = (args, tuple(sorted(kwargs.items())))
            cached_val = lru.get(key)
            if cached_val is not None:
                return cached_val
            res = func(*args, **kwargs)
            lru.put(key, res)
            return res

        wrapper.cache_clear = lru.clear  # type: ignore
        return wrapper

    return decorator
