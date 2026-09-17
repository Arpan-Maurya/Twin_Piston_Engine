"""
Per-Cylinder EGT Diagnostics — Original Module 7.

Third processing stage of L2 Digital Twin.
Performs deterministic per-cylinder EGT health and balance diagnostics from
NormalizedSignalRecord produced by Module 5 (and optional DerivedEngineState from Module 6).

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (and optional DerivedEngineState)
    - Output: DiagnosticState (canonical domain schema) & EGTDiagnosticResult
    - Zero simulator internals or ground truth dependencies
    - Zero ML inference or advisory generation logic
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, EGTDiagnosticConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import DiagnosticState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)


def _severity_rank(status: DiagnosticStatus) -> int:
    """Helper for severity comparison."""
    ranks = {
        DiagnosticStatus.INVALID: 0,
        DiagnosticStatus.NORMAL: 1,
        DiagnosticStatus.WARNING: 2,
        DiagnosticStatus.CRITICAL: 3,
    }
    return ranks.get(status, 0)


def _max_status(s1: DiagnosticStatus, s2: DiagnosticStatus) -> DiagnosticStatus:
    """Return the status with higher severity."""
    if s1 == DiagnosticStatus.INVALID and s2 != DiagnosticStatus.INVALID:
        return s2
    if s2 == DiagnosticStatus.INVALID and s1 != DiagnosticStatus.INVALID:
        return s1
    return s1 if _severity_rank(s1) >= _severity_rank(s2) else s2


class CylinderEGTDiagnostic(BaseModel):
    """Detailed EGT diagnostic result for a single cylinder (1..4)."""

    cylinder_id: int = Field(ge=1, le=4)
    egt_k: float | None
    valid: bool
    status: DiagnosticStatus
    deviation_from_mean_k: float | None
    rate_of_change_k_s: float | None
    quality: float = Field(ge=0.0, le=1.0)
    reason: str | None = None

    model_config = ConfigDict(frozen=True)


class EGTDiagnosticResult(BaseModel):
    """Complete EGT diagnostic assessment across all cylinders."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    cylinder_diagnostics: list[CylinderEGTDiagnostic]
    mean_egt_k: float | None
    spread_egt_k: float | None
    max_egt_cylinder: int | None
    min_egt_cylinder: int | None
    overall_status: DiagnosticStatus
    imbalance_detected: bool
    affected_cylinders: list[int]

    model_config = ConfigDict(frozen=True)


class EGTDiagnosticsEngine:
    """Engine for evaluating per-cylinder EGT telemetry health & imbalance."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: EGTDiagnosticConfig = self._settings.egt_diagnostics
        self._prev_timestamp: datetime | None = None
        self._prev_egts: dict[int, float] = {}

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_egts.clear()

    def evaluate(self, record: NormalizedSignalRecord) -> tuple[DiagnosticState, EGTDiagnosticResult]:
        """Evaluate EGT diagnostics on a NormalizedSignalRecord.

        Returns both canonical DiagnosticState and detailed EGTDiagnosticResult.
        """
        raw_egt_channels: list[tuple[int, ChannelValue]] = [
            (1, record.egt_cyl_1),
            (2, record.egt_cyl_2),
            (3, record.egt_cyl_3),
            (4, record.egt_cyl_4),
        ]

        # Calculate temporal dt
        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        # Step 1: Collect valid cylinders
        valid_cylinders: list[tuple[int, float, float]] = []  # (cyl_id, egt_k, quality)
        for cyl_id, ch in raw_egt_channels:
            if ch.valid and ch.value > 0.0:
                valid_cylinders.append((cyl_id, ch.value, ch.quality))

        # Step 2: Mean, Max, Min, Spread
        if len(valid_cylinders) > 0:
            sum_egt = sum(v for _, v, _ in valid_cylinders)
            mean_egt: float | None = sum_egt / len(valid_cylinders)

            sorted_cyls = sorted(valid_cylinders, key=lambda x: x[1])
            min_cyl_id, min_egt_val, _ = sorted_cyls[0]
            max_cyl_id, max_egt_val, _ = sorted_cyls[-1]

            min_egt_cyl: int | None = min_cyl_id
            max_egt_cyl: int | None = max_cyl_id
            spread_egt: float | None = max_egt_val - min_egt_val
        else:
            mean_egt = None
            spread_egt = None
            min_egt_cyl = None
            max_egt_cyl = None

        # Step 3: Per-cylinder diagnostics
        cyl_diags: list[CylinderEGTDiagnostic] = []
        dev_list: list[float] = [0.0, 0.0, 0.0, 0.0]
        egt_k_list: list[float | None] = [None, None, None, None]
        status_list: list[DiagnosticStatus] = [DiagnosticStatus.INVALID] * 4
        rate_list: list[float | None] = [None] * 4
        affected_cylinders: list[int] = []

        overall_status = DiagnosticStatus.NORMAL if len(valid_cylinders) > 0 else DiagnosticStatus.INVALID

        new_prev_egts: dict[int, float] = {}

        for cyl_id, ch in raw_egt_channels:
            idx = cyl_id - 1
            if not ch.valid or ch.value <= 0.0:
                cyl_diags.append(
                    CylinderEGTDiagnostic(
                        cylinder_id=cyl_id,
                        egt_k=None,
                        valid=False,
                        status=DiagnosticStatus.INVALID,
                        deviation_from_mean_k=None,
                        rate_of_change_k_s=None,
                        quality=0.0,
                        reason=ch.fault_flag or f"Cylinder {cyl_id} EGT channel invalid or missing",
                    )
                )
                affected_cylinders.append(cyl_id)
                continue

            egt_val = ch.value
            new_prev_egts[cyl_id] = egt_val
            egt_k_list[idx] = egt_val

            # Deviation from mean
            dev = egt_val - mean_egt if mean_egt is not None else 0.0
            dev_list[idx] = dev

            # Rate of change
            rate_of_change: float | None = None
            if dt_s is not None and 0.0 < dt_s <= 10.0 and cyl_id in self._prev_egts:
                prev_val = self._prev_egts[cyl_id]
                rate_of_change = (egt_val - prev_val) / dt_s
            rate_list[idx] = rate_of_change

            # Threshold Checks
            status = DiagnosticStatus.NORMAL
            reasons: list[str] = []

            # 1. Absolute EGT temperature thresholds
            if egt_val >= self._cfg.egt_critical_temp_k:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Cylinder {cyl_id} EGT ({egt_val:.1f}K) exceeds critical threshold ({self._cfg.egt_critical_temp_k:.1f}K)")
            elif egt_val >= self._cfg.egt_warning_temp_k:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Cylinder {cyl_id} EGT ({egt_val:.1f}K) exceeds warning threshold ({self._cfg.egt_warning_temp_k:.1f}K)")

            # 2. Deviation from mean thresholds
            abs_dev = abs(dev)
            if abs_dev >= self._cfg.egt_dev_critical_k:
                status = _max_status(status, DiagnosticStatus.CRITICAL)
                reasons.append(f"Cylinder {cyl_id} EGT deviation ({dev:+.1f}K) exceeds critical threshold ({self._cfg.egt_dev_critical_k:.1f}K)")
            elif abs_dev >= self._cfg.egt_dev_warning_k:
                status = _max_status(status, DiagnosticStatus.WARNING)
                reasons.append(f"Cylinder {cyl_id} EGT deviation ({dev:+.1f}K) exceeds warning threshold ({self._cfg.egt_dev_warning_k:.1f}K)")

            # 3. Rate of change thresholds
            if rate_of_change is not None:
                abs_rate = abs(rate_of_change)
                if abs_rate >= self._cfg.egt_rate_critical_k_s:
                    status = _max_status(status, DiagnosticStatus.CRITICAL)
                    reasons.append(f"Cylinder {cyl_id} dEGT/dt ({rate_of_change:+.1f}K/s) exceeds critical rate")
                elif abs_rate >= self._cfg.egt_rate_warning_k_s:
                    status = _max_status(status, DiagnosticStatus.WARNING)
                    reasons.append(f"Cylinder {cyl_id} dEGT/dt ({rate_of_change:+.1f}K/s) exceeds warning rate")

            status_list[idx] = status
            if status != DiagnosticStatus.NORMAL:
                affected_cylinders.append(cyl_id)
                overall_status = _max_status(overall_status, status)

            cyl_diags.append(
                CylinderEGTDiagnostic(
                    cylinder_id=cyl_id,
                    egt_k=egt_val,
                    valid=True,
                    status=status,
                    deviation_from_mean_k=dev,
                    rate_of_change_k_s=rate_of_change,
                    quality=ch.quality,
                    reason="; ".join(reasons) if reasons else "Normal",
                )
            )

        # Step 4: Engine EGT Spread Imbalance
        imbalance_detected = False
        if spread_egt is not None:
            if spread_egt >= self._cfg.egt_spread_critical_k:
                overall_status = _max_status(overall_status, DiagnosticStatus.CRITICAL)
                imbalance_detected = True
            elif spread_egt >= self._cfg.egt_spread_warning_k:
                overall_status = _max_status(overall_status, DiagnosticStatus.WARNING)
                imbalance_detected = True

        # Update temporal state
        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        self._prev_egts = new_prev_egts

        # Step 5: Build result schemas
        egt_result = EGTDiagnosticResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            cylinder_diagnostics=cyl_diags,
            mean_egt_k=mean_egt,
            spread_egt_k=spread_egt,
            max_egt_cylinder=max_egt_cyl,
            min_egt_cylinder=min_egt_cyl,
            overall_status=overall_status,
            imbalance_detected=imbalance_detected,
            affected_cylinders=sorted(list(set(affected_cylinders))),
        )

        diagnostic_state = DiagnosticState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            per_cylinder_egt_dev_k=dev_list,
            per_cylinder_egt_k=egt_k_list,
            per_cylinder_egt_status=status_list,
            egt_mean_k=mean_egt,
            egt_spread_k=spread_egt,
            max_egt_cylinder=max_egt_cyl,
            min_egt_cylinder=min_egt_cyl,
            egt_rate_of_change_k_s=rate_list,
            egt_overall_status=overall_status,
        )

        return diagnostic_state, egt_result


def evaluate_egt_diagnostics(
    record: NormalizedSignalRecord,
    settings: AppSettings | None = None,
) -> tuple[DiagnosticState, EGTDiagnosticResult]:
    """Convenience function for evaluating Module 7 EGT diagnostics."""
    engine = EGTDiagnosticsEngine(settings)
    return engine.evaluate(record)
