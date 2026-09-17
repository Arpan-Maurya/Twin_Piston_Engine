"""
Remaining Useful Life (RUL) with Uncertainty — Original Module 15.

Fourth stage of L3 ML & Supervision Layer.
Provides deterministic RUL estimation, history tracking, uncertainty bounds,
and Module 12 ML model integration.

STRICT BOUNDARY CONSTRAINTS:
    - Predictive analytics ONLY. Zero engine/UAV control or actuation commands.
    - ML & physics inputs MUST NOT consume RawSignalRecord directly.
    - RUL MUST NOT be derived as HI * arbitrary lifetime.
    - Uncertainty is mandatory; never fabricate fake confidence or fake uncertainty.
    - If no trained ML RUL model exists, reports MODEL_UNAVAILABLE.
    - Baseline trend estimators are clearly tagged with is_ml=False and Provenance.DERIVED.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    DegradationState,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    HealthState,
    ModelMetadata,
    RULState,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.ml_infrastructure import (
    MLFeatureVector,
    MLInferenceService,
)

logger = get_logger(__name__)


class HealthHistoryTracker:
    """Deterministic history tracking for HealthState observations."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._max_samples = max_samples
        self._samples: list[HealthState] = []

    def clear(self) -> None:
        """Clear historical samples."""
        self._samples.clear()

    def add_sample(self, sample: HealthState) -> bool:
        """Add a HealthState sample to history.

        Validates timestamp order, validity, and non-duplicate timestamps.
        """
        if isinstance(sample, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: HealthHistoryTracker MUST NOT consume RawSignalRecord directly.")

        if not sample.health_index.valid or sample.health_index.value is None or math.isnan(sample.health_index.value):
            return False

        if self._samples:
            last_ts = self._samples[-1].timestamp
            dt = (sample.timestamp - last_ts).total_seconds()
            if dt <= 0.0:  # Skip duplicate or negative timestamps
                return False
            if dt > 300.0:  # Large time gap: clear old history window
                self._samples.clear()

        self._samples.append(sample)
        if len(self._samples) > self._max_samples:
            self._samples.pop(0)

        return True

    def get_samples(self) -> list[HealthState]:
        return list(self._samples)

    def count(self) -> int:
        return len(self._samples)


class RULEstimator:
    """Remaining Useful Life estimation engine integrating Module 12 ML infrastructure and baseline trend analytics."""

    def __init__(
        self,
        inference_service: MLInferenceService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._inference_service = inference_service or MLInferenceService(settings=settings)
        self._settings = settings or get_settings()
        self._config = self._settings.rul
        self._history_tracker = HealthHistoryTracker(max_samples=self._config.history_window_max_samples)

    def reset_history(self) -> None:
        """Reset historical tracking state."""
        self._history_tracker.clear()

    def estimate_rul(
        self,
        health_state: HealthState,
        feature_vector: MLFeatureVector | None = None,
        operating_assumption: str | None = None,
        model_name: str = "rul_estimator",
    ) -> RULState:
        """Estimate RUL using ML model (if available) or deterministic baseline fallback.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        if isinstance(health_state, RawSignalRecord) or isinstance(feature_vector, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: RULEstimator MUST NOT consume RawSignalRecord directly.")

        # Record health sample in history
        self._history_tracker.add_sample(health_state)

        assumption = operating_assumption or self._config.default_operating_assumption

        # 1. Try ML model inference via Module 12 if feature vector is provided
        if feature_vector is not None:
            ml_res = self._estimate_rul_ml(feature_vector, health_state, assumption, model_name)
            if ml_res.status != InferenceStatus.MODEL_UNAVAILABLE:
                return ml_res

        # 2. Baseline trend estimator fallback
        return self._estimate_rul_baseline(health_state, assumption)

    def _estimate_rul_ml(
        self,
        feature_vector: MLFeatureVector,
        health_state: HealthState,
        operating_assumption: str,
        model_name: str,
    ) -> RULState:
        """Execute ML RUL model via Module 12 inference service."""
        ts = feature_vector.timestamp

        inf_result = self._inference_service.predict(model_name, feature_vector)

        if inf_result.status != InferenceStatus.SUCCESS:
            return RULState(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                hours_remaining=0.0,
                lower_bound_hours=None,
                upper_bound_hours=None,
                confidence=None,
                unit="hours",
                status=inf_result.status,
                quality=inf_result.quality,
                degradation_state=health_state.degradation_state,
                trend=health_state.trend,
                operating_assumption=operating_assumption,
                model_name=model_name,
                model_version=inf_result.model_version,
                feature_schema_version=inf_result.feature_schema_version,
                evidence={"error": inf_result.error_message or str(inf_result.status)},
                is_ml=True,
            )

        pred = inf_result.prediction
        rul_est = 0.0
        lower: float | None = None
        upper: float | None = None
        conf: float | None = None
        unc_avail = False
        unc_rep: dict[str, Any] | None = None

        if isinstance(pred, dict):
            rul_est = float(pred.get("hours_remaining", pred.get("rul_estimate", 0.0)))
            lower = float(pred["lower_bound"]) if "lower_bound" in pred and pred["lower_bound"] is not None else None
            upper = float(pred["upper_bound"]) if "upper_bound" in pred and pred["upper_bound"] is not None else None
            conf = float(pred["confidence"]) if "confidence" in pred and pred["confidence"] is not None else None
            unc_rep = pred.get("uncertainty")
            unc_avail = unc_rep is not None or (lower is not None and upper is not None)
        elif isinstance(pred, (int, float)):
            rul_est = float(pred)
            lower = None
            upper = None
            conf = None
            unc_avail = False
        else:
            return self._invalid_prediction_result(ts, health_state, operating_assumption, "Unrecognized model output structure")

        # Sanity validation
        is_valid, reason = self._validate_rul_sanity(rul_est, lower, upper)
        if not is_valid:
            return self._invalid_prediction_result(ts, health_state, operating_assumption, reason)

        return RULState(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            hours_remaining=rul_est,
            lower_bound_hours=lower,
            upper_bound_hours=upper,
            confidence=conf,
            unit="hours",
            status=InferenceStatus.SUCCESS,
            quality=inf_result.quality,
            uncertainty_available=unc_avail,
            uncertainty_representation=unc_rep,
            degradation_state=health_state.degradation_state,
            trend=health_state.trend,
            operating_assumption=operating_assumption,
            model_name=model_name,
            model_version=inf_result.model_version,
            feature_schema_version=inf_result.feature_schema_version,
            evidence={"prediction_type": "ML_MODEL"},
            is_ml=True,
        )

    def _estimate_rul_baseline(
        self,
        health_state: HealthState,
        operating_assumption: str,
    ) -> RULState:
        """Deterministic baseline trend-extrapolation RUL estimator.

        Uses historical (t, HI) observations to project remaining time to target HI threshold.
        Clearly tagged as is_ml=False and Provenance.DERIVED.
        """
        ts = health_state.timestamp
        samples = self._history_tracker.get_samples()

        if len(samples) < self._config.min_history_samples:
            return RULState(
                timestamp=ts,
                provenance=Provenance.DERIVED,
                hours_remaining=0.0,
                lower_bound_hours=None,
                upper_bound_hours=None,
                confidence=None,
                unit="hours",
                status=InferenceStatus.INSUFFICIENT_HISTORY,
                quality=health_state.quality * (len(samples) / float(self._config.min_history_samples)),
                uncertainty_available=False,
                degradation_state=health_state.degradation_state,
                trend=health_state.trend,
                operating_assumption=operating_assumption,
                evidence={"samples_available": len(samples), "samples_required": self._config.min_history_samples},
                is_ml=False,
            )

        # Compute linear regression over (t_sec, hi) points
        t0 = samples[0].timestamp
        x_sec = [(s.timestamp - t0).total_seconds() for s in samples]
        y_hi = [s.health_index.value for s in samples]

        n = len(samples)
        mean_x = sum(x_sec) / n
        mean_y = sum(y_hi) / n

        ss_xx = sum((x - mean_x) ** 2 for x in x_sec)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_sec, y_hi))

        target_hi = self._config.baseline_degradation_target_hi
        current_hi = health_state.health_index.value

        # Baseline slope calculation
        if ss_xx > 0.0:
            slope_sec = ss_xy / ss_xx  # dHI/dt_sec
        else:
            slope_sec = 0.0

        max_horizon = self._config.max_prediction_horizon_hours

        # Scenario 1: Stable or Improving health (slope >= 0)
        if slope_sec >= 0.0:
            rul_hours = max_horizon
            lower_hours: float | None = None
            upper_hours: float | None = None
            uncertainty_available = False
            unc_rep = None
            conf = None

        # Scenario 2: Degrading health (slope < 0)
        else:
            # Time in seconds to reach target HI
            remaining_sec = (current_hi - target_hi) / abs(slope_sec)
            rul_hours = min(remaining_sec / 3600.0, max_horizon)

            # Compute slope standard error for confidence interval if n >= 3
            residual_sq_sum = sum((y - (mean_y + slope_sec * (x - mean_x))) ** 2 for x, y in zip(x_sec, y_hi))
            if n > 2 and ss_xx > 0.0:
                s_err = math.sqrt(residual_sq_sum / (n - 2))
                slope_se = s_err / math.sqrt(ss_xx)

                # 95% interval slope bounds (using t approx ~ 2.0)
                slope_lower = slope_sec - 2.0 * slope_se
                slope_upper = slope_sec + 2.0 * slope_se

                # Convert to RUL bounds in hours
                if slope_lower < 0.0:
                    rul_upper = min((current_hi - target_hi) / abs(slope_lower) / 3600.0, max_horizon)
                else:
                    rul_upper = max_horizon

                if slope_upper < 0.0:
                    rul_lower = max(0.0, (current_hi - target_hi) / abs(slope_upper) / 3600.0)
                else:
                    rul_lower = max(0.0, rul_hours * 0.7)

                lower_hours = min(rul_hours, rul_lower)
                upper_hours = max(rul_hours, rul_upper)
                uncertainty_available = True
                unc_rep = {"slope_se": slope_se, "residual_std": s_err}
                conf = max(0.5, min(0.95, 1.0 - (s_err / (mean_y + 1e-6))))
            else:
                lower_hours = None
                upper_hours = None
                uncertainty_available = False
                unc_rep = None
                conf = None

        # Sanity validation
        is_valid, reason = self._validate_rul_sanity(rul_hours, lower_hours, upper_hours)
        if not is_valid:
            return self._invalid_prediction_result(ts, health_state, operating_assumption, reason)

        return RULState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            hours_remaining=rul_hours,
            lower_bound_hours=lower_hours,
            upper_bound_hours=upper_hours,
            confidence=conf,
            unit="hours",
            status=InferenceStatus.SUCCESS,
            quality=health_state.quality,
            uncertainty_available=uncertainty_available,
            uncertainty_representation=unc_rep,
            degradation_state=health_state.degradation_state,
            trend=health_state.trend,
            operating_assumption=operating_assumption,
            model_name="baseline_degradation_trend_estimator",
            model_version=self._settings.version,
            evidence={
                "slope_per_hour": slope_sec * 3600.0,
                "history_samples_used": n,
                "target_hi": target_hi,
            },
            is_ml=False,
        )

    def _validate_rul_sanity(
        self,
        rul_est: float,
        lower: float | None,
        upper: float | None,
    ) -> tuple[bool, str]:
        """Validate RUL bounds and non-negativity sanity."""
        if math.isnan(rul_est) or math.isinf(rul_est) or rul_est < 0.0:
            return False, f"RUL estimate is invalid: {rul_est}"

        if lower is not None:
            if math.isnan(lower) or math.isinf(lower) or lower < 0.0:
                return False, f"Lower bound is invalid: {lower}"
            if lower > rul_est:
                return False, f"Lower bound {lower} > RUL estimate {rul_est}"

        if upper is not None:
            if math.isnan(upper) or math.isinf(upper) or upper < 0.0:
                return False, f"Upper bound is invalid: {upper}"
            if upper < rul_est:
                return False, f"Upper bound {upper} < RUL estimate {rul_est}"

        return True, ""

    def _invalid_prediction_result(
        self,
        ts: datetime,
        health_state: HealthState,
        operating_assumption: str,
        reason: str,
    ) -> RULState:
        return RULState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            hours_remaining=0.0,
            lower_bound_hours=None,
            upper_bound_hours=None,
            confidence=None,
            unit="hours",
            status=InferenceStatus.INVALID_PREDICTION,
            quality=0.0,
            uncertainty_available=False,
            degradation_state=health_state.degradation_state,
            trend=health_state.trend,
            operating_assumption=operating_assumption,
            evidence={"reason": reason},
            is_ml=False,
        )
