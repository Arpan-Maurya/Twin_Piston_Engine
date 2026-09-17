"""
Source Taxonomy & Intended Use Enumerations — Module 22.

Defines formal dataset source categories, intended-use roles, and validation status levels.
Aligns with existing Provenance tags without duplicating concepts.
"""

from __future__ import annotations

from enum import Enum


class SourceType(str, Enum):
    """Categorical source type of a dataset."""

    SIMULATED = "SIMULATED"
    REPLAY = "REPLAY"
    LIVE = "LIVE"
    VALIDATION_GROUND_TRUTH = "VALIDATION_GROUND_TRUTH"


class IntendedUse(str, Enum):
    """Intended operational role of a dataset in the system."""

    DEVELOPMENT = "DEVELOPMENT"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    REPLAY = "REPLAY"
    SCIENTIFIC_VALIDATION = "SCIENTIFIC_VALIDATION"
    DEMONSTRATION = "DEMONSTRATION"


class ValidationStatus(str, Enum):
    """Outcome status of a validation harness evaluation."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
