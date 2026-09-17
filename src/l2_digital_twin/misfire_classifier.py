"""
Misfire and Combustion Stability Diagnostics — Original Module 10.

Sixth processing stage of L2 Digital Twin.
Performs deterministic multi-signal evidence fusion (EGT drop/deviation, crank speed stability,
and vibration spectral features) to characterize combustion stability and detect possible misfire
per cylinder.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (Module 5), DerivedEngineState (Module 6),
             DiagnosticState/EGTDiagnosticResult (Module 7), VibrationState/Result (Module 9)
    - Output: CombustionStabilityState & MisfireDiagnosticResult
    - Zero ML fault classifiers, anomaly detectors, RUL, or advisory recommendation logic
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, MisfireConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import (
    CombustionStabilityState,
    DerivedEngineState,
    DiagnosticState,
    VibrationState,
)
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l2_digital_twin.vibration_processor import VibrationSignalProcessingResult

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


class CylinderCombustionEvidence(BaseModel):
    """Multi-signal combustion evidence breakdown for a single cylinder (1..4)."""

    cylinder_id: int = Field(ge=1, le=4)
    status: DiagnosticStatus
    possible_misfire: bool
    evidence_score: float = Field(ge=0.0, le=1.0)
    egt_evidence: float = Field(ge=0.0, le=1.0)
    crank_evidence: float = Field(ge=0.0, le=1.0)
    vibration_evidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class MisfireDiagnosticResult(BaseModel):
    """Complete engine combustion stability & misfire diagnostic result container."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    cylinder_evidence: list[CylinderCombustionEvidence]
    misfire_detected: list[bool]
    overall_status: DiagnosticStatus
    affected_cylinders: list[int]

    model_config = ConfigDict(frozen=True)


class MisfireDetector:
    """Deterministic multi-signal combustion stability and misfire analyzer."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: MisfireConfig = self._settings.misfire
        self._prev_timestamp: datetime | None = None
        self._prev_egts: dict[int, float] = {}
        self._prev_rpm: float | None = None

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_egts.clear()
        self._prev_rpm = None

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
        vib_diag: VibrationSignalProcessingResult | VibrationState | None = None,
    ) -> tuple[CombustionStabilityState, MisfireDiagnosticResult]:
        """Evaluate multi-signal combustion stability and misfire indicators."""
        raw_egts: list[tuple[int, ChannelValue]] = [
            (1, record.egt_cyl_1),
            (2, record.egt_cyl_2),
            (3, record.egt_cyl_3),
            (4, record.egt_cyl_4),
        ]

        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        # Step 1: Crank Speed Irregularity Evidence
        rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value > 0) else None
        crank_evidence_val = 0.0
        crank_reasons: list[str] = []

        if rpm_val is not None and self._prev_rpm is not None and dt_s is not None and 0.0 < dt_s <= 10.0:
            delta_rpm = abs(rpm_val - self._prev_rpm)
            rate_rpm_s = delta_rpm / dt_s

            if rate_rpm_s >= self._cfg.rpm_std_threshold:
                crank_evidence_val = min(1.0, rate_rpm_s / (self._cfg.rpm_std_threshold * 2.0))
                crank_reasons.append(f"Crank speed fluctuation ({rate_rpm_s:.1f} RPM/s) detected")
        elif rpm_val is None:
            crank_reasons.append("RPM telemetry missing/invalid")

        # Step 2: Vibration Evidence
        vibration_evidence_val = 0.0
        vibration_reasons: list[str] = []

        overall_rms = 0.0
        if isinstance(vib_diag, VibrationSignalProcessingResult):
            overall_rms = vib_diag.overall_rms_m_s2
        elif isinstance(vib_diag, VibrationState):
            overall_rms = vib_diag.overall_rms_m_s2.value if vib_diag.overall_rms_m_s2.valid else 0.0
        else:
            vib_rms_ch = record.vibration_rms
            if vib_rms_ch.valid:
                overall_rms = vib_rms_ch.value

        if overall_rms >= self._cfg.vibration_rms_threshold_m_s2:
            vibration_evidence_val = min(1.0, overall_rms / (self._cfg.vibration_rms_threshold_m_s2 * 1.5))
            vibration_reasons.append(f"Elevated vibration RMS ({overall_rms:.1f} m/s²) detected")

        # Step 3: Extract EGT mean / deviations
        valid_egt_vals = [ch.value for _, ch in raw_egts if ch.valid and ch.value > 0.0]
        mean_egt = (sum(valid_egt_vals) / len(valid_egt_vals)) if valid_egt_vals else None

        # Step 4: Per-Cylinder Evidence Fusion
        cyl_evidences: list[CylinderCombustionEvidence] = []
        misfire_flags: list[bool] = [False, False, False, False]
        status_list: list[DiagnosticStatus] = [DiagnosticStatus.NORMAL] * 4
        score_list: list[float] = [0.0] * 4
        affected_cyls: list[int] = []

        overall_status = DiagnosticStatus.NORMAL if valid_egt_vals else DiagnosticStatus.INVALID
        new_prev_egts: dict[int, float] = {}

        for cyl_id, ch in raw_egts:
            idx = cyl_id - 1
            if not ch.valid or ch.value <= 0.0:
                cyl_evidences.append(
                    CylinderCombustionEvidence(
                        cylinder_id=cyl_id,
                        status=DiagnosticStatus.INVALID,
                        possible_misfire=False,
                        evidence_score=0.0,
                        egt_evidence=0.0,
                        crank_evidence=0.0,
                        vibration_evidence=0.0,
                        reasons=[ch.fault_flag or f"Cylinder {cyl_id} EGT channel invalid"],
                    )
                )
                status_list[idx] = DiagnosticStatus.INVALID
                continue

            egt_val = ch.value
            new_prev_egts[cyl_id] = egt_val

            egt_evidence_val = 0.0
            reasons: list[str] = []

            # 1. EGT drop / deviation indicators
            if mean_egt is not None:
                dev = egt_val - mean_egt
                if dev <= -self._cfg.egt_dev_threshold_k:
                    egt_evidence_val = min(1.0, abs(dev) / (self._cfg.egt_dev_threshold_k * 1.5))
                    reasons.append(f"Cylinder {cyl_id} EGT deviation ({dev:+.1f} K) below mean")

            # 2. Sudden temporal EGT drop
            if dt_s is not None and 0.0 < dt_s <= 10.0 and cyl_id in self._prev_egts:
                prev_val = self._prev_egts[cyl_id]
                egt_drop_rate = (prev_val - egt_val) / dt_s
                if egt_drop_rate >= self._cfg.egt_rate_drop_threshold_k_s:
                    egt_ev_drop = min(1.0, egt_drop_rate / (self._cfg.egt_rate_drop_threshold_k_s * 1.5))
                    egt_evidence_val = max(egt_evidence_val, egt_ev_drop)
                    reasons.append(f"Rapid EGT drop rate (-{egt_drop_rate:.1f} K/s)")

            # Multi-Signal Evidence Fusion
            total_score = (
                (self._cfg.egt_evidence_weight * egt_evidence_val)
                + (self._cfg.crank_evidence_weight * crank_evidence_val)
                + (self._cfg.vibration_evidence_weight * vibration_evidence_val)
            )
            total_score = float(min(1.0, max(0.0, total_score)))
            score_list[idx] = total_score

            status = DiagnosticStatus.NORMAL
            possible_misfire = False

            if total_score >= self._cfg.misfire_confidence_threshold or (egt_evidence_val >= 0.8 and crank_evidence_val >= 0.5):
                status = DiagnosticStatus.CRITICAL
                possible_misfire = True
                misfire_flags[idx] = True
                affected_cyls.append(cyl_id)
                reasons.append(f"High multi-signal evidence ({total_score:.2f}) indicates possible misfire")
            elif total_score >= self._cfg.unstable_confidence_threshold or egt_evidence_val >= 0.5:
                status = DiagnosticStatus.WARNING
                affected_cyls.append(cyl_id)
                reasons.append(f"Moderate evidence ({total_score:.2f}) indicates combustion instability")

            status_list[idx] = status
            overall_status = _max_status(overall_status, status)

            all_reasons = reasons + crank_reasons + vibration_reasons
            cyl_evidences.append(
                CylinderCombustionEvidence(
                    cylinder_id=cyl_id,
                    status=status,
                    possible_misfire=possible_misfire,
                    evidence_score=total_score,
                    egt_evidence=egt_evidence_val,
                    crank_evidence=crank_evidence_val,
                    vibration_evidence=vibration_evidence_val,
                    reasons=all_reasons if all_reasons else ["Normal combustion"],
                )
            )

        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        self._prev_egts = new_prev_egts
        if rpm_val is not None:
            self._prev_rpm = rpm_val

        misfire_result = MisfireDiagnosticResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            cylinder_evidence=cyl_evidences,
            misfire_detected=misfire_flags,
            overall_status=overall_status,
            affected_cylinders=sorted(list(set(affected_cyls))),
        )

        stability_state = CombustionStabilityState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            misfire_detected=misfire_flags,
            misfire_status=status_list,
            evidence_score=score_list,
            overall_combustion_status=overall_status,
        )

        return stability_state, misfire_result


def evaluate_misfire_detector(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
    vib_diag: VibrationSignalProcessingResult | VibrationState | None = None,
    settings: AppSettings | None = None,
) -> tuple[CombustionStabilityState, MisfireDiagnosticResult]:
    """Convenience function for Module 10 misfire & combustion stability evaluation."""
    detector = MisfireDetector(settings)
    return detector.evaluate(record, derived_state, egt_diag, vib_diag)
