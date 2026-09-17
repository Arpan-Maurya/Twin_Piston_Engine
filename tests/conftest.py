"""
Shared test fixtures and configuration for pytest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest

from src.core.config import AppSettings, load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset_settings_singleton() -> Generator[None, None, None]:
    """Reset the global settings singleton before each test.

    This ensures tests don't leak configuration state to each other.
    """
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings() -> AppSettings:
    """Load application settings from the default config file."""
    return load_settings()


@pytest.fixture
def config_path() -> Path:
    """Path to the default configuration YAML."""
    return Path("config/default.yaml")


@pytest.fixture
def project_root() -> Path:
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def now_utc() -> datetime:
    """Current UTC timestamp."""
    return datetime.now(timezone.utc)
