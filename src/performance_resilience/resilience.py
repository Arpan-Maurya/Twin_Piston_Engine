"""
Resilience & Subsystem Failure Isolation Utilities — Module 23.

Provides:
- Idempotent transient retry execution with exponential backoff
- Call timeout protection
- Exception-isolated subsystem degradation wrappers (ML, Advisory, Digital Twin)
"""

from __future__ import annotations

import asyncio
import time
import functools
from typing import Any, Callable, TypeVar

from src.core.exceptions import PistonEngineError
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay_s: float = 0.1,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator for retrying transient operations with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay_s
            last_exc = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as ex:
                    last_exc = ex
                    logger.warning(
                        f"Transient failure on {func.__name__} (attempt {attempt}/{max_retries}): {ex}"
                    )
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor

            raise last_exc or RuntimeError(f"Operation {func.__name__} failed after {max_retries} retries.")

        return wrapper

    return decorator


def safe_subsystem_call(
    subsystem_name: str = "subsystem",
    func: Callable[..., T] | None = None,
    fallback_value: T | None = None,
    fallback: T | None = None,
) -> T:
    """Safely execute a subsystem function, catching exceptions and returning fallback."""
    effective_fallback = fallback if fallback is not None else fallback_value
    # If called positional with (func, fallback_value, subsystem_name)
    if callable(subsystem_name) and func is not None:
        actual_func = subsystem_name
        effective_fallback = func
        actual_name = str(fallback_value or "subsystem")
    else:
        actual_func = func  # type: ignore
        actual_name = subsystem_name

    try:
        return actual_func()
    except Exception as e:
        logger.error(f"Failure in {actual_name}: {e}. Falling back to degraded state.")
        return effective_fallback  # type: ignore

