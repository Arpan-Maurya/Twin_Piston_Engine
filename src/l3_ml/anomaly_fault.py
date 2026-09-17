"""
Anomaly Detection and Nine-Class Fault Classifier — Original Module 13.

Second stage of L3 ML & Supervision Layer.
Provides generic anomaly detection and nine-class fault classification services,
integrating directly with Original Module 12 ML Inference Infrastructure.

STRICT BOUNDARY CONSTRAINTS:
    - ML MUST NOT consume RawSignalRecord directly. Features MUST come from Modules 5-11 / Module 12 feature vectors.
    - Preserves exact project nine-class fault taxonomy (FaultClass 0-8).
    - Reuses Module 12 model registry/infrastructure; does NOT implement a duplicate model loader.
    - Zero fabricated model predictions, zero fake accuracy claims.
    - Reports MODEL_UNAVAILABLE when trained model artifacts are missing.
    - Rule-based diagnostic fallbacks are explicitly flagged with is_ml=False and Provenance.DERIVED.
    - Zero advisory recommendations or flight control logic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    CombustionStabilityState,
    DerivedEngineState,
    DiagnosticState,
    FaultClassificationResult,
    LubricationState,
    ModelMetadata,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
)
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.ml_infrastructure import (
    MLFeatureVector,
    MLInferenceService,
)

logger = get_logger(__name__)

# Authoritative 9-Class Taxonomy Map
FAULT_CLASS_MAP: dict[int, FaultClass] = {fc.value: fc for fc in FaultClass}


class AnomalyDetector:
    """Anomaly detection service operating over approved Module 12 feature vectors."""

    def __init__(
        self,
        inference_service: MLInferenceService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._inference_service = inference_service or MLInferenceService(settings=settings)
        self._settings = settings or get_settings()

    def detect_anomaly(
        self,
        feature_vector: MLFeatureVector,
        model_name: str = "anomaly_detector",
    ) -> AnomalyResult:
        """Evaluate anomaly detection model via Module 12 inference infrastructure.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        if isinstance(feature_vector, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: AnomalyDetector MUST NOT consume RawSignalRecord directly.")

        ts = feature_vector.timestamp

        # Check input validity
        if not feature_vector.valid:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INVALID_INPUT,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=0.5,
                evidence={"reason": "Invalid or non-finite feature vector"},
                quality=0.0,
                is_ml=True,
            )

        inf_result = self._inference_service.predict(model_name, feature_vector)

        if inf_result.status != InferenceStatus.SUCCESS:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=inf_result.status,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=0.5,
                evidence={"error": inf_result.error_message or str(inf_result.status)},
                quality=inf_result.quality,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        # Interpret output prediction from model
        pred = inf_result.prediction
        is_anomaly = False
        score = 0.0
        threshold = 0.5
        evidence: dict[str, Any] = {}

        if isinstance(pred, dict):
            score = float(pred.get("score", 0.0))
            threshold = float(pred.get("threshold", 0.5))
            is_anomaly = bool(pred.get("is_anomaly", score >= threshold))
            evidence = pred.get("evidence", {})
        elif isinstance(pred, (int, float)):
            score = float(pred)
            threshold = 0.5
            is_anomaly = score >= threshold
            evidence = {"raw_score": score}
        elif isinstance(pred, (list, tuple)) and len(pred) > 0:
            score = float(pred[0])
            threshold = 0.5
            is_anomaly = score >= threshold
            evidence = {"raw_score": score}

        # Finite check on score
        if math.isnan(score) or math.isinf(score):
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INFERENCE_ERROR,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=threshold,
                evidence={"reason": "Non-finite anomaly score returned by model"},
                quality=0.0,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        return AnomalyResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=InferenceStatus.SUCCESS,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            threshold=threshold,
            evidence=evidence,
            quality=inf_result.quality,
            model_metadata=inf_result.model_metadata,
            is_ml=True,
        )

    def detect_anomaly_rule_fallback(
        self,
        residual_state: ResidualState,
        threshold: float = 3.0,
    ) -> AnomalyResult:
        """Deterministic rule/threshold-based anomaly detector.

        Used when ML model is unavailable or for deterministic validation.
        Clearly tags output as is_ml=False and Provenance.DERIVED.
        """
        if isinstance(residual_state, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: AnomalyDetector MUST NOT consume RawSignalRecord directly.")

        ts = residual_state.timestamp
        max_residual = 0.0
        evidence: dict[str, Any] = {}
        invalid_count = 0

        for key, ptv in residual_state.residuals.items():
            if not ptv.valid or ptv.value is None or math.isnan(ptv.value) or math.isinf(ptv.value):
                invalid_count += 1
                continue
            abs_val = abs(ptv.value)
            evidence[key] = abs_val
            if abs_val > max_residual:
                max_residual = abs_val

        if invalid_count == len(residual_state.residuals) and len(residual_state.residuals) > 0:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.DERIVED,
                status=InferenceStatus.INVALID_INPUT,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=threshold,
                evidence={"reason": "All input residuals are invalid"},
                quality=0.0,
                is_ml=False,
            )

        is_anomaly = max_residual >= threshold
        # Normalize score relative to threshold (cap at 10.0 for safety)
        norm_score = min(max_residual / max(threshold, 1e-6), 10.0)

        return AnomalyResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            status=InferenceStatus.SUCCESS,
            is_anomaly=is_anomaly,
            anomaly_score=norm_score,
            threshold=threshold,
            evidence=evidence,
            quality=1.0 if invalid_count == 0 else max(0.0, 1.0 - (invalid_count / len(residual_state.residuals))),
            is_ml=False,
        )


class NineClassFaultClassifier:
    """Nine-class fault classification service integrating with Module 12 ML infrastructure."""

    def __init__(
        self,
        inference_service: MLInferenceService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._inference_service = inference_service or MLInferenceService(settings=settings)
        self._settings = settings or get_settings()

    def classify_fault(
        self,
        feature_vector: MLFeatureVector,
        model_name: str = "fault_classifier",
    ) -> FaultClassificationResult:
        """Classify fault into exact nine-class taxonomy using Module 12 ML infrastructure.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        if isinstance(feature_vector, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: FaultClassifier MUST NOT consume RawSignalRecord directly.")

        ts = feature_vector.timestamp

        # Validate feature vector input
        if not feature_vector.valid:
            return FaultClassificationResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INVALID_INPUT,
                predicted_class=FaultClass.NOMINAL,
                class_id=FaultClass.NOMINAL.value,
                class_name=FaultClass.NOMINAL.name,
                confidence=None,
                probabilities=None,
                quality=0.0,
                is_ml=True,
            )

        inf_result = self._inference_service.predict(model_name, feature_vector)

        if inf_result.status != InferenceStatus.SUCCESS:
            return FaultClassificationResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=inf_result.status,
                predicted_class=FaultClass.NOMINAL,
                class_id=FaultClass.NOMINAL.value,
                class_name=FaultClass.NOMINAL.name,
                confidence=None,
                probabilities=None,
                quality=inf_result.quality,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        pred = inf_result.prediction
        predicted_class = FaultClass.NOMINAL
        confidence: float | None = None
        probabilities: dict[str, float] | None = None
        evidence: dict[str, Any] | None = None

        # Format A: Dictionary of class probabilities or dict with "predicted_class" / "probabilities"
        if isinstance(pred, dict):
            if "probabilities" in pred and isinstance(pred["probabilities"], (dict, list)):
                raw_probs = pred["probabilities"]
                prob_dict, conf, p_class = self._parse_probabilities(raw_probs)
                if prob_dict is not None:
                    probabilities = prob_dict
                    confidence = conf
                    predicted_class = p_class
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
            elif "class_id" in pred:
                cid = int(pred["class_id"])
                if cid in FAULT_CLASS_MAP:
                    predicted_class = FAULT_CLASS_MAP[cid]
                    confidence = float(pred.get("confidence")) if pred.get("confidence") is not None else None
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
            else:
                # Direct dict of class_name -> float
                prob_dict, conf, p_class = self._parse_probabilities(pred)
                if prob_dict is not None:
                    probabilities = prob_dict
                    confidence = conf
                    predicted_class = p_class
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        # Format B: List/Sequence of 9 probabilities
        elif isinstance(pred, (list, tuple)):
            prob_dict, conf, p_class = self._parse_probabilities(pred)
            if prob_dict is not None:
                probabilities = prob_dict
                confidence = conf
                predicted_class = p_class
            else:
                return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        # Format C: Scalar integer class_id
        elif isinstance(pred, int):
            if pred in FAULT_CLASS_MAP:
                predicted_class = FAULT_CLASS_MAP[pred]
                confidence = None
                probabilities = None
            else:
                return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
        else:
            return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=InferenceStatus.SUCCESS,
            predicted_class=predicted_class,
            class_id=predicted_class.value,
            class_name=predicted_class.name,
            confidence=confidence,
            probabilities=probabilities,
            feature_schema_version=inf_result.feature_schema_version,
            quality=inf_result.quality,
            evidence=evidence,
            model_metadata=inf_result.model_metadata,
            is_ml=True,
        )

    def classify_fault_rule_fallback(
        self,
        residual_state: ResidualState | None = None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
        vib_state: VibrationState | None = None,
        comb_state: CombustionStabilityState | None = None,
        lub_state: LubricationState | None = None,
    ) -> FaultClassificationResult:
        """Deterministic physics/diagnostic rule-based fault classifier fallback.

        Used for offline/diagnostic testing when ML model artifact is unavailable.
        Evaluates Module 7-10 outputs to classify into exact FaultClass (0-8).
        Clearly tags result as is_ml=False and Provenance.DERIVED.
        """
        ts = datetime.now(timezone.utc)
        evidence: dict[str, Any] = {}
        predicted_class = FaultClass.NOMINAL

        # 1. Misfire check (Module 10)
        if comb_state and any(comb_state.misfire_detected):
            predicted_class = FaultClass.MISFIRE
            evidence["misfire_cylinders"] = [i + 1 for i, m in enumerate(comb_state.misfire_detected) if m]

        # 2. Exhaust Valve Leak check (Module 7 EGT diagnostic)
        elif egt_diag and hasattr(egt_diag, "overall_status") and egt_diag.overall_status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            predicted_class = FaultClass.EXHAUST_VALVE_LEAK
            if isinstance(egt_diag, EGTDiagnosticResult):
                evidence["egt_spread_k"] = egt_diag.spread_egt_k
            elif hasattr(egt_diag, "max_spread_c"):
                evidence["egt_spread_c"] = getattr(egt_diag, "max_spread_c").value

        # 3. Bearing Wear / Mechanical Vibration check (Module 9)
        elif vib_state and vib_state.overall_rms_m_s2.valid and vib_state.overall_rms_m_s2.value > 15.0:
            predicted_class = FaultClass.BEARING_WEAR
            evidence["vibration_rms"] = vib_state.overall_rms_m_s2.value

        # 4. Oil Degradation check (Module 8)
        elif lub_state and lub_state.oil_pressure_bar.valid and lub_state.oil_pressure_bar.value < 1.5:
            predicted_class = FaultClass.OIL_DEGRADATION
            evidence["oil_pressure_bar"] = lub_state.oil_pressure_bar.value

        # 5. Intake Boost Leak check (Module 11 residuals)
        elif residual_state and "map_pressure" in residual_state.residuals:
            map_res = residual_state.residuals["map_pressure"]
            if map_res.valid and map_res.value < -10000.0:
                predicted_class = FaultClass.INTAKE_BOOST_LEAK
                evidence["map_residual_pa"] = map_res.value

        # 6. Cooling Fault check (Module 11 residuals)
        elif residual_state and "oil_temp" in residual_state.residuals:
            ot_res = residual_state.residuals["oil_temp"]
            if ot_res.valid and ot_res.value > 15.0:
                predicted_class = FaultClass.COOLING_FAULT
                evidence["oil_temp_residual_c"] = ot_res.value

        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            status=InferenceStatus.SUCCESS,
            predicted_class=predicted_class,
            class_id=predicted_class.value,
            class_name=predicted_class.name,
            confidence=1.0 if predicted_class != FaultClass.NOMINAL else 0.9,
            probabilities=None,  # Rule engine does not compute ML soft-max probabilities
            quality=1.0,
            evidence=evidence,
            is_ml=False,
        )

    def _parse_probabilities(
        self,
        probs: Any,
    ) -> tuple[dict[str, float] | None, float | None, FaultClass]:
        """Validate and format 9-class probability distribution."""
        prob_dict: dict[str, float] = {}

        if isinstance(probs, dict):
            # Map by class name or integer string
            for k, v in probs.items():
                val = float(v)
                if math.isnan(val) or math.isinf(val) or val < 0.0:
                    return None, None, FaultClass.NOMINAL
                if k in FAULT_CLASS_MAP:
                    fc = FAULT_CLASS_MAP[int(k)]
                    prob_dict[fc.name] = val
                elif hasattr(FaultClass, str(k)):
                    prob_dict[str(k)] = val
                else:
                    return None, None, FaultClass.NOMINAL

            if len(prob_dict) != 9:
                return None, None, FaultClass.NOMINAL

        elif isinstance(probs, (list, tuple)):
            if len(probs) != 9:
                return None, None, FaultClass.NOMINAL
            for i, p in enumerate(probs):
                val = float(p)
                if math.isnan(val) or math.isinf(val) or val < 0.0:
                    return None, None, FaultClass.NOMINAL
                fc = FAULT_CLASS_MAP[i]
                prob_dict[fc.name] = val
        else:
            return None, None, FaultClass.NOMINAL

        # Find max probability class
        best_class_name = max(prob_dict, key=lambda k: prob_dict[k])
        best_prob = prob_dict[best_class_name]
        best_fc = getattr(FaultClass, best_class_name)

        return prob_dict, best_prob, best_fc

    def _error_result(
        self,
        ts: datetime,
        status: InferenceStatus,
        metadata: ModelMetadata | None,
    ) -> FaultClassificationResult:
        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=status,
            predicted_class=FaultClass.NOMINAL,
            class_id=FaultClass.NOMINAL.value,
            class_name=FaultClass.NOMINAL.name,
            confidence=None,
            probabilities=None,
            quality=0.0,
            model_metadata=metadata,
            is_ml=True,
        )
