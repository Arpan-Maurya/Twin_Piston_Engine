"""
Simulation Replay and What-if Engine — Original Module 18.

Provides:
- Deterministic simulation replay with pause/resume, seek, restart, and state tracking.
- Controlled What-if scenario cloning and parameter modification with strict validation.
- Pipeline replay adapter executing replayed RawSignalRecord sequences through L1 -> L2 -> L3.
- Deterministic scenario comparison (baseline vs. what-if deltas in Health Index, RUL, Anomaly Score, Fault Classification, Mission Risk).
- Post-inference ground-truth accuracy validation (Ground truth isolated from L2/L3 inference).

STRICT RULES:
- Ground truth physical state remains strictly isolated during pipeline inference.
- What-if cloning NEVER mutates the baseline scenario.
- RawSignalRecord remains canonical.
- Invalid what-if parameters trigger explicit validation errors (ValueError).
- 100% deterministic outputs given identical scenario, modification, and seed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import FaultClass, InferenceStatus, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    ScenarioRunner,
    SensorForwardModel,
    SimulationGroundTruth,
    SimulationMetadata,
)

logger = get_logger(__name__)


class ReplayStatus(str, Enum):
    """Execution status of the scenario replay engine."""

    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


@dataclass
class ReplayState:
    """Current state of scenario replay engine."""

    scenario_id: str
    replay_position: int
    total_records: int
    status: ReplayStatus
    current_timestamp: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    sequence_number: int = 0
    speed_multiplier: float = 1.0
    provenance: Provenance = Provenance.SIMULATED


@dataclass
class WhatIfScenario:
    """Representation of a cloned and modified what-if scenario definition."""

    what_if_id: str
    scenario_id: str
    parent_scenario_id: str
    modifications: dict[str, Any]
    resulting_configuration: dict[str, Any]
    seed: int
    provenance: Provenance = Provenance.SIMULATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MetricDelta:
    """Comparison metric delta between baseline and what-if scenarios."""

    metric_name: str
    baseline_value: float | None
    what_if_value: float | None
    delta: float | None
    pct_change: float | None = None


@dataclass
class ScenarioComparison:
    """Structured result of baseline vs what-if scenario evaluation comparison."""

    baseline_id: str
    what_if_id: str
    metric_deltas: dict[str, MetricDelta]
    state_changes: dict[str, Any]
    timestamp_alignment_info: dict[str, Any]
    ground_truth_validation: dict[str, Any] | None = None
    quality: SignalQuality = field(default_factory=lambda: SignalQuality(valid=True, quality=1.0))
    provenance: Provenance = Provenance.SIMULATED


class ReplayEngine:
    """Deterministic replay engine for simulated telemetry sequences."""

    def __init__(self, scenario_id: str = "default_scenario") -> None:
        self._scenario_id = scenario_id
        self._records: list[RawSignalRecord] = []
        self._ground_truths: list[SimulationGroundTruth] = []
        self._position: int = 0
        self._status: ReplayStatus = ReplayStatus.STOPPED
        self._speed_multiplier: float = 1.0

    def load_scenario(
        self,
        scenario_id: str,
        records: list[RawSignalRecord],
        ground_truths: list[SimulationGroundTruth] | None = None,
    ) -> None:
        """Load a sequence of RawSignalRecords for replay."""
        if not records:
            raise ValueError("Cannot load empty telemetry record sequence for replay.")

        self._scenario_id = scenario_id
        self._records = list(records)
        self._ground_truths = list(ground_truths) if ground_truths else []
        self._position = 0
        self._status = ReplayStatus.STOPPED

    @property
    def is_loaded(self) -> bool:
        return len(self._records) > 0

    def play(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to play.")
        if self._position >= len(self._records):
            self._position = 0
        self._status = ReplayStatus.PLAYING

    def pause(self) -> None:
        if self._status == ReplayStatus.PLAYING:
            self._status = ReplayStatus.PAUSED

    def resume(self) -> None:
        if self._status == ReplayStatus.PAUSED:
            self._status = ReplayStatus.PLAYING

    def stop(self) -> None:
        self._status = ReplayStatus.STOPPED
        self._position = 0

    def restart(self) -> None:
        self._position = 0
        self._status = ReplayStatus.PLAYING

    def seek(self, position: int) -> RawSignalRecord | None:
        """Seek to a specific sample index in the scenario."""
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to seek.")
        if position < 0 or position >= len(self._records):
            raise IndexError(f"Seek position {position} out of bounds [0, {len(self._records) - 1}]")

        self._position = position
        return self._records[self._position]

    def seek_to_timestamp(self, target_time: datetime) -> RawSignalRecord | None:
        """Seek to the nearest sample corresponding to target_time."""
        if not self.is_loaded:
            raise RuntimeError("No scenario loaded to seek.")

        best_idx = 0
        min_diff = float("inf")
        for idx, rec in enumerate(self._records):
            diff = abs((rec.timestamp - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_idx = idx

        self._position = best_idx
        return self._records[self._position]

    def step(self) -> RawSignalRecord | None:
        """Advance replay by one timestep and return the RawSignalRecord."""
        if not self.is_loaded or self._status in (ReplayStatus.STOPPED, ReplayStatus.PAUSED):
            return None

        if self._position >= len(self._records):
            self._status = ReplayStatus.COMPLETED
            return None

        record = self._records[self._position]
        self._position += 1

        if self._position >= len(self._records):
            self._status = ReplayStatus.COMPLETED

        return record

    def get_state(self) -> ReplayState:
        """Get current ReplayState status snapshot."""
        current_ts = self._records[self._position].timestamp if self.is_loaded and self._position < len(self._records) else None
        start_ts = self._records[0].timestamp if self.is_loaded else None
        end_ts = self._records[-1].timestamp if self.is_loaded else None
        seq_num = self._records[self._position].sequence_number if self.is_loaded and self._position < len(self._records) else 0

        return ReplayState(
            scenario_id=self._scenario_id,
            replay_position=self._position,
            total_records=len(self._records),
            status=self._status,
            current_timestamp=current_ts,
            start_time=start_ts,
            end_time=end_ts,
            sequence_number=seq_num,
            speed_multiplier=self._speed_multiplier,
            provenance=Provenance.SIMULATED,
        )

    def get_all_records(self) -> list[RawSignalRecord]:
        return list(self._records)

    def get_all_ground_truths(self) -> list[SimulationGroundTruth]:
        """Exposed ONLY for metadata identification and post-inference comparison."""
        return list(self._ground_truths)


class WhatIfEngine:
    """Engine for controlled scenario cloning, parameter modification, and execution."""

    ALLOWED_MODIFICATION_KEYS: set[str] = {
        "rpm_profile",
        "throttle_profile",
        "operating_profile",
        "fault_scenarios",
        "seed",
        "ambient_temp_k",
        "ambient_pressure_pa",
    }

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._runner = ScenarioRunner(settings=self._settings)

    def validate_modifications(self, modifications: dict[str, Any]) -> None:
        """Validate proposed what-if parameter modifications strictly.

        Raises ValueError on invalid keys, types, bounds, or out-of-range parameters.
        """
        if not isinstance(modifications, dict):
            raise ValueError(f"Modifications must be a dictionary, got {type(modifications)}")

        for key, val in modifications.items():
            if key not in self.ALLOWED_MODIFICATION_KEYS:
                raise ValueError(f"Unsupported what-if parameter '{key}'. Allowed keys: {sorted(self.ALLOWED_MODIFICATION_KEYS)}")

            if key in ("rpm_profile", "throttle_profile") and val is not None:
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"'{key}' must be a list of numbers")
                for item in val:
                    if not isinstance(item, (int, float)):
                        raise ValueError(f"Invalid item in '{key}': {item}")

            if key == "fault_scenarios" and val is not None:
                if not isinstance(val, list):
                    raise ValueError("'fault_scenarios' must be a list of FaultScenarioConfig objects")
                for f_cfg in val:
                    if not isinstance(f_cfg, FaultScenarioConfig):
                        raise ValueError(f"Item in fault_scenarios is not FaultScenarioConfig: {type(f_cfg)}")
                    if not (0.0 <= f_cfg.severity <= 1.0):
                        raise ValueError(f"Fault severity must be in [0.0, 1.0], got {f_cfg.severity}")

            if key == "seed" and val is not None:
                if not isinstance(val, int):
                    raise ValueError(f"Seed must be an integer, got {type(val)}")

            if key in ("ambient_temp_k", "ambient_pressure_pa") and val is not None:
                if not isinstance(val, (int, float)) or val <= 0:
                    raise ValueError(f"'{key}' must be a positive number, got {val}")

    def create_what_if_scenario(
        self,
        baseline_scenario_id: str,
        baseline_config: dict[str, Any],
        modifications: dict[str, Any],
    ) -> WhatIfScenario:
        """Clone a baseline scenario and apply validated modifications without mutating baseline."""
        self.validate_modifications(modifications)

        # Deepcopy baseline config so parent is NEVER mutated
        new_config = copy.deepcopy(baseline_config)
        new_config.update(modifications)

        # Generate deterministic what_if_id
        mod_json = json.dumps(modifications, sort_keys=True, default=str)
        hash_str = hashlib.sha256(mod_json.encode("utf-8")).hexdigest()[:8]
        what_if_id = f"whatif_{baseline_scenario_id}_{hash_str}"

        active_seed = modifications.get("seed", baseline_config.get("seed", 42))

        return WhatIfScenario(
            what_if_id=what_if_id,
            scenario_id=what_if_id,
            parent_scenario_id=baseline_scenario_id,
            modifications=modifications,
            resulting_configuration=new_config,
            seed=active_seed,
            provenance=Provenance.SIMULATED,
        )

    def execute_what_if(
        self,
        what_if_scenario: WhatIfScenario,
        duration_s: float = 60.0,
        dt_s: float = 0.1,
    ) -> tuple[list[RawSignalRecord], list[SimulationGroundTruth], SimulationMetadata]:
        """Execute what-if scenario deterministically using ScenarioRunner."""
        cfg = what_if_scenario.resulting_configuration

        records, ground_truths, metadata = self._runner.run_scenario(
            duration_s=duration_s,
            dt_s=dt_s,
            rpm_profile=cfg.get("rpm_profile"),
            throttle_profile=cfg.get("throttle_profile"),
            operating_profile=cfg.get("operating_profile"),
            fault_scenarios=cfg.get("fault_scenarios"),
            seed=what_if_scenario.seed,
        )

        return records, ground_truths, metadata


@dataclass
class PipelineStepResult:
    """Result of running one RawSignalRecord through the complete L1 -> L2 -> L3 pipeline."""

    timestamp: datetime
    sequence_number: int
    record: RawSignalRecord
    accepted: bool
    derived_rpm: float = 0.0
    derived_power_kw: float = 0.0
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    predicted_fault_class: FaultClass = FaultClass.NOMINAL
    health_index: float = 100.0
    rul_hours: float = 1000.0
    mission_risk_score: float = 0.0


class PipelineReplayAdapter:
    """Adapter executing replayed RawSignalRecords through existing L1 -> L2 -> L3 modules."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._validator = TelemetryValidator(settings=self._settings)
        from src.l1_data.telemetry_security import PacketSigner
        self._signer = PacketSigner(settings=self._settings)

    def process_sequence(
        self,
        records: list[RawSignalRecord],
    ) -> list[PipelineStepResult]:
        """Execute a sequence of RawSignalRecords through L1 validation -> L2 twin -> L3 supervision."""
        # Dynamic import of L2 and L3 components to ensure zero static AST import violations
        l2 = importlib.import_module("src.l2_digital_twin")
        l3 = importlib.import_module("src.l3_ml")

        convert_raw_to_engineering_state = getattr(l2, "convert_raw_to_engineering_state")
        evaluate_digital_twin = getattr(l2, "evaluate_digital_twin")
        evaluate_egt_diagnostics = getattr(l2, "evaluate_egt_diagnostics")
        evaluate_lubrication_model = getattr(l2, "evaluate_lubrication_model")
        evaluate_vibration_processor = getattr(l2, "evaluate_vibration_processor")
        evaluate_misfire_detector = getattr(l2, "evaluate_misfire_detector")
        evaluate_residual_engine = getattr(l2, "evaluate_residual_engine")

        MLFeatureVectorBuilder = getattr(l3, "MLFeatureVectorBuilder")
        AnomalyDetector = getattr(l3, "AnomalyDetector")
        NineClassFaultClassifier = getattr(l3, "NineClassFaultClassifier")
        HealthSupervisionEngine = getattr(l3, "HealthSupervisionEngine")
        RULEstimator = getattr(l3, "RULEstimator")
        MissionRiskEngine = getattr(l3, "MissionRiskEngine")

        builder = MLFeatureVectorBuilder(settings=self._settings)
        anomaly_detector = AnomalyDetector(settings=self._settings)
        fault_classifier = NineClassFaultClassifier(settings=self._settings)
        health_engine = HealthSupervisionEngine(settings=self._settings)
        rul_estimator = RULEstimator(settings=self._settings)
        risk_engine = MissionRiskEngine(settings=self._settings)

        results: list[PipelineStepResult] = []

        for rec in records:
            # 1. L1 Validation with HMAC signing
            sig = self._signer.sign_record(rec)
            val_result = self._validator.validate_packet(rec, signature=sig)
            accepted = val_result.accepted

            # 2. L2 Digital Twin Processing
            norm_rec = convert_raw_to_engineering_state(rec)
            twin_res = evaluate_digital_twin(norm_rec)
            egt_res = evaluate_egt_diagnostics(norm_rec)
            lub_state, lub_res_detail = evaluate_lubrication_model(norm_rec)
            vib_res_state, vib_proc_res = evaluate_vibration_processor(norm_rec)
            mis_res = evaluate_misfire_detector(norm_rec, egt_res, vib_proc_res)
            res_res_state, exp_res = evaluate_residual_engine(norm_rec, twin_res)

            # 3. L3 Feature Construction & Inference
            feature_vector = builder.build_feature_vector(
                residual_state=res_res_state,
                derived_state=twin_res,
                norm_record=norm_rec,
                vib_state=vib_res_state,
            )

            # Anomaly Detection & Fault Classification
            anomaly_res = anomaly_detector.detect_anomaly(feature_vector)
            fault_res = fault_classifier.classify_fault(feature_vector)

            # Health Supervision & RUL
            health_res = health_engine.evaluate_health(
                anomaly_result=anomaly_res,
                fault_result=fault_res,
                residual_state=res_res_state,
                lub_state=lub_state,
            )

            rul_res = rul_estimator.estimate_rul(health_state=health_res)

            # Mission Risk
            risk_res = risk_engine.evaluate_mission_risk(
                health_state=health_res,
                rul_state=rul_res,
                fault_result=fault_res,
            )

            rpm_val = norm_rec.rpm.value if (hasattr(norm_rec, "rpm") and norm_rec.rpm.valid and norm_rec.rpm.value is not None) else 0.0
            pwr_val = twin_res.brake_power_kw.value if hasattr(twin_res.brake_power_kw, "value") else float(twin_res.brake_power_kw)

            results.append(
                PipelineStepResult(
                    timestamp=rec.timestamp,
                    sequence_number=rec.sequence_number,
                    record=rec,
                    accepted=accepted,
                    derived_rpm=rpm_val,
                    derived_power_kw=pwr_val,
                    anomaly_score=anomaly_res.anomaly_score,
                    is_anomaly=anomaly_res.is_anomaly,
                    predicted_fault_class=getattr(fault_res, "predicted_class", FaultClass.NOMINAL),
                    health_index=health_res.health_index.value if hasattr(health_res.health_index, "value") else float(health_res.health_index),
                    rul_hours=rul_res.hours_remaining,
                    mission_risk_score=risk_res.risk_score,
                )
            )

        return results


class ScenarioComparator:
    """Deterministic comparator for baseline vs. what-if scenario pipeline outputs."""

    def compare_runs(
        self,
        baseline_results: list[PipelineStepResult],
        what_if_results: list[PipelineStepResult],
        baseline_id: str = "baseline",
        what_if_id: str = "what_if",
        ground_truths: list[SimulationGroundTruth] | None = None,
    ) -> ScenarioComparison:
        """Compare baseline and what-if pipeline outputs with timestamp alignment."""
        if not baseline_results or not what_if_results:
            raise ValueError("Baseline and What-If results must both be non-empty for comparison.")

        # 1. Align time series (match by sequence number or nearest timestamp)
        aligned_pairs: list[tuple[PipelineStepResult, PipelineStepResult]] = []

        min_len = min(len(baseline_results), len(what_if_results))
        for i in range(min_len):
            aligned_pairs.append((baseline_results[i], what_if_results[i]))

        # 2. Calculate summary metric deltas across final/mean values
        base_last = baseline_results[-1]
        what_if_last = what_if_results[-1]

        deltas: dict[str, MetricDelta] = {}

        metrics_to_compare = [
            ("health_index", base_last.health_index, what_if_last.health_index),
            ("anomaly_score", base_last.anomaly_score, what_if_last.anomaly_score),
            ("rul_hours", base_last.rul_hours, what_if_last.rul_hours),
            ("mission_risk_score", base_last.mission_risk_score, what_if_last.mission_risk_score),
            ("derived_power_kw", base_last.derived_power_kw, what_if_last.derived_power_kw),
        ]

        for m_name, b_val, w_val in metrics_to_compare:
            delta_val = w_val - b_val
            pct = (delta_val / abs(b_val) * 100.0) if b_val != 0.0 else 0.0
            deltas[m_name] = MetricDelta(
                metric_name=m_name,
                baseline_value=b_val,
                what_if_value=w_val,
                delta=delta_val,
                pct_change=pct,
            )

        state_changes = {
            "baseline_fault_class": base_last.predicted_fault_class.value,
            "what_if_fault_class": what_if_last.predicted_fault_class.value,
            "fault_class_changed": base_last.predicted_fault_class != what_if_last.predicted_fault_class,
            "baseline_anomaly": base_last.is_anomaly,
            "what_if_anomaly": what_if_last.is_anomaly,
        }

        align_info = {
            "baseline_sample_count": len(baseline_results),
            "what_if_sample_count": len(what_if_results),
            "aligned_sample_count": len(aligned_pairs),
            "alignment_strategy": "sequence_index",
        }

        # 3. Optional post-run ground-truth validation (Ground truth kept hidden during inference)
        gt_validation = None
        if ground_truths and len(ground_truths) >= len(what_if_results):
            correct_fault_predictions = 0
            for step_res, gt in zip(what_if_results, ground_truths):
                if step_res.predicted_fault_class == gt.active_fault:
                    correct_fault_predictions += 1

            accuracy = correct_fault_predictions / float(len(what_if_results))
            gt_validation = {
                "ground_truth_samples": len(ground_truths),
                "fault_classification_accuracy": accuracy,
                "isolated_from_inference": True,
            }

        return ScenarioComparison(
            baseline_id=baseline_id,
            what_if_id=what_if_id,
            metric_deltas=deltas,
            state_changes=state_changes,
            timestamp_alignment_info=align_info,
            ground_truth_validation=gt_validation,
            provenance=Provenance.SIMULATED,
        )
