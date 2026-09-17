"""
Acceptance Models & Schemas — Module 24.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class AcceptanceStatus(str, Enum):
    """Scientific acceptance status options."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ModuleAcceptanceResult(BaseModel):
    """Result summary for an individual module scientific acceptance evaluation."""

    module_number: int
    module_name: str
    status: AcceptanceStatus
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    key_verifications: list[str] = Field(default_factory=list)
    limitations_noted: list[str] = Field(default_factory=list)


class ScientificAcceptanceSummary(BaseModel):
    """Overall summary report model for Original Module 24."""

    timestamp: str
    overall_status: AcceptanceStatus
    total_scientific_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    modules_evaluated_count: int = 24  # Modules 0-23
    modules_passed_count: int = 0
    determinism_verified: bool = True
    ground_truth_isolation_verified: bool = True
    zero_actuation_commands_verified: bool = True
    modules_0_to_23_regression_pass: bool = True
    module_results: list[ModuleAcceptanceResult] = Field(default_factory=list)
