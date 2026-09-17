"""
ML Inference Infrastructure — Original Module 12.

First stage of L3 ML & Supervision Layer.
Provides production-grade infrastructure for feature vector construction, schema validation,
model loading, lifecycle management, deterministic inference execution, and latency monitoring.

STRICT BOUNDARY CONSTRAINTS:
    - ML MUST NOT consume RawSignalRecord directly. Features MUST come from Modules 5-11.
    - Infrastructure MUST report MODEL_UNAVAILABLE when model artifacts are missing.
    - Zero fake model predictions, zero fake ML training, zero fake accuracy claims.
    - Zero Module 13+ fault classification or anomaly detection logic.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    InferenceStatus,
    ModelLifecycleStatus,
    Provenance,
)
from src.core.schemas import (
    DerivedEngineState,
    ModelMetadata,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import NormalizedSignalRecord

logger = get_logger(__name__)

STANDARD_FEATURE_ORDER = [
    "egt_cyl1_residual",
    "egt_cyl2_residual",
    "egt_cyl3_residual",
    "egt_cyl4_residual",
    "oil_pressure_residual",
    "oil_temp_residual",
    "vibration_rms_residual",
    "brake_power_kw_residual",
    "bmep_pa",
    "eta_volumetric",
    "eta_thermal",
    "mean_piston_speed_m_s",
]


class MLFeature(BaseModel):
    """Container for a single ML input feature with lineage metadata."""

    name: str
    value: float
    unit_or_semantic: str = ""
    provenance: Provenance = Field(default=Provenance.DERIVED)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    valid: bool = True

    model_config = ConfigDict(frozen=True)


class MLFeatureVector(BaseModel):
    """Deterministic, validated feature vector for ML inference."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: Provenance = Field(default=Provenance.DERIVED)
    feature_names: list[str]
    values: list[float]
    features: dict[str, MLFeature]
    schema_version: str = "1.0.0"
    valid: bool = True
    quality: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class MLFeatureVectorBuilder:
    """Deterministic builder for ML input feature vectors."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def build_feature_vector(
        self,
        residual_state: ResidualState | None = None,
        derived_state: DerivedEngineState | None = None,
        norm_record: NormalizedSignalRecord | None = None,
        vib_state: VibrationState | None = None,
        schema_version: str = "1.0.0",
        **kwargs: Any,
    ) -> MLFeatureVector:
        """Construct a deterministic feature vector from approved L2 physics outputs.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        # Strict boundary check against raw telemetry
        for arg in list(kwargs.values()) + [residual_state, derived_state, norm_record, vib_state]:
            if isinstance(arg, RawSignalRecord):
                raise TypeError(
                    "STRICT BOUNDARY VIOLATION: ML features MUST NOT consume RawSignalRecord directly. "
                    "Use approved Module 11 residuals or Module 6 physics-derived features."
                )

        ts = (
            (residual_state.timestamp if residual_state else None)
            or (derived_state.timestamp if derived_state else None)
            or (norm_record.timestamp if norm_record else None)
            or datetime.now(timezone.utc)
        )

        extracted_features: dict[str, MLFeature] = {}
        values_list: list[float] = []
        overall_valid = True
        qualities: list[float] = []

        def _get_val(feat_name: str) -> MLFeature:
            clean_name = feat_name.removesuffix("_residual")
            if residual_state:
                if feat_name in residual_state.residuals:
                    ptv = residual_state.residuals[feat_name]
                    is_finite = ptv.value is not None and not math.isnan(ptv.value) and not math.isinf(ptv.value)
                    val = ptv.value if (ptv.valid and is_finite) else 0.0
                    return MLFeature(
                        name=feat_name,
                        value=val,
                        provenance=ptv.provenance,
                        quality=ptv.quality if (ptv.valid and is_finite) else 0.0,
                        valid=ptv.valid and is_finite,
                    )
                if clean_name in residual_state.residuals:
                    ptv = residual_state.residuals[clean_name]
                    is_finite = ptv.value is not None and not math.isnan(ptv.value) and not math.isinf(ptv.value)
                    val = ptv.value if (ptv.valid and is_finite) else 0.0
                    return MLFeature(
                        name=feat_name,
                        value=val,
                        provenance=ptv.provenance,
                        quality=ptv.quality if (ptv.valid and is_finite) else 0.0,
                        valid=ptv.valid and is_finite,
                    )

            if derived_state:
                target_attr = feat_name if hasattr(derived_state, feat_name) else clean_name
                if hasattr(derived_state, target_attr):
                    attr = getattr(derived_state, target_attr)
                    if isinstance(attr, ProvenanceTaggedValue):
                        is_finite = attr.value is not None and not math.isnan(attr.value) and not math.isinf(attr.value)
                        val = attr.value if (attr.valid and is_finite) else 0.0
                        return MLFeature(
                            name=feat_name,
                            value=val,
                            provenance=attr.provenance,
                            quality=attr.quality if (attr.valid and is_finite) else 0.0,
                            valid=attr.valid and is_finite,
                        )

            if vib_state:
                target_attr = feat_name if hasattr(vib_state, feat_name) else clean_name
                if hasattr(vib_state, target_attr):
                    attr = getattr(vib_state, target_attr)
                    if isinstance(attr, ProvenanceTaggedValue):
                        is_finite = attr.value is not None and not math.isnan(attr.value) and not math.isinf(attr.value)
                        val = attr.value if (attr.valid and is_finite) else 0.0
                        return MLFeature(
                            name=feat_name,
                            value=val,
                            provenance=attr.provenance,
                            quality=attr.quality if (attr.valid and is_finite) else 0.0,
                            valid=attr.valid and is_finite,
                        )

            # Missing feature fallback (reproducible zero-fill)
            return MLFeature(
                name=feat_name,
                value=0.0,
                provenance=Provenance.DERIVED,
                quality=0.5,
                valid=True,
            )

        for fname in STANDARD_FEATURE_ORDER:
            feat = _get_val(fname)
            if not feat.valid or math.isnan(feat.value) or math.isinf(feat.value):
                feat = MLFeature(name=fname, value=0.0, provenance=feat.provenance, quality=0.0, valid=False)
                overall_valid = False

            extracted_features[fname] = feat
            values_list.append(feat.value)
            qualities.append(feat.quality)

        avg_quality = (sum(qualities) / len(qualities)) if qualities else 1.0

        return MLFeatureVector(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            feature_names=STANDARD_FEATURE_ORDER,
            values=values_list,
            features=extracted_features,
            schema_version=schema_version,
            valid=overall_valid,
            quality=avg_quality,
        )


class InferenceResult(BaseModel):
    """Container for ML model prediction output and execution metadata."""

    model_name: str
    model_version: str
    feature_schema_version: str = "1.0.0"
    prediction: Any = None
    status: InferenceStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    error_message: str | None = None
    model_metadata: ModelMetadata | None = None

    model_config = ConfigDict(frozen=True)


class BaseMLModel:
    """Base class / abstract interface for ML models in the infrastructure."""

    def __init__(self, metadata: ModelMetadata, feature_schema_version: str = "1.0.0") -> None:
        self.metadata = metadata
        self.feature_schema_version = feature_schema_version
        self.status = ModelLifecycleStatus.READY

    def validate_schema(self, feature_vector: MLFeatureVector) -> bool:
        """Validate feature vector schema compatibility."""
        if feature_vector.schema_version != self.feature_schema_version:
            return False
        if len(self.metadata.input_shape) > 0 and len(feature_vector.values) != self.metadata.input_shape[0]:
            return False
        return True

    def predict(self, feature_vector: MLFeatureVector) -> Any:
        """Execute model inference. Must be overridden by model implementations."""
        raise NotImplementedError("Subclasses must implement predict()")


class MLModelLoader:
    """Safely loads ML model artifacts from disk with security path checks."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def load_model(
        self,
        model_path: str | Path,
        metadata: ModelMetadata,
        feature_schema_version: str = "1.0.0",
    ) -> BaseMLModel | None:
        """Attempt to load a model artifact from disk. Returns None if artifact missing.

        SEC-001: Validates that model_path resolves within the configured model_dir
        to prevent path-traversal attacks loading arbitrary pickle objects.
        """
        path = Path(model_path).resolve()
        allowed_dir = Path(self._settings.model_dir).resolve()

        # Path confinement check — prevent traversal outside model directory
        try:
            path.relative_to(allowed_dir)
        except ValueError:
            logger.warning(
                "SEC-001: Model path '%s' is outside allowed model directory '%s'. Rejecting load.",
                path, allowed_dir,
            )
            return None

        if not path.exists() or not path.is_file():
            logger.info(f"ML Model artifact not found at {path}. System will report MODEL_UNAVAILABLE.")
            return None

        # Safe loading wrapper
        try:
            import joblib
            model_obj = joblib.load(path)
            
            class LoadedJoblibModel(BaseMLModel):
                def __init__(self, meta: ModelMetadata, schema_ver: str, obj: Any) -> None:
                    super().__init__(meta, schema_ver)
                    self._obj = obj

                def predict(self, feature_vector: MLFeatureVector) -> Any:
                    feats = np.array([feature_vector.values])
                    return self._obj.predict(feats).tolist()

            return LoadedJoblibModel(metadata, feature_schema_version, model_obj)
        except Exception as e:
            logger.error(f"Failed to load ML model artifact from {path}: {e}")
            return None


class MLInferenceService:
    """Service managing ML model lifecycle and executing inference."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._models: dict[str, BaseMLModel] = {}
        self.loader = MLModelLoader(self._settings)
        self.feature_builder = MLFeatureVectorBuilder(self._settings)

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded and ready."""
        return model_name in self._models and self._models[model_name].status == ModelLifecycleStatus.READY

    def register_model_instance(self, model: BaseMLModel) -> None:
        """Register a valid model instance directly into service (e.g. for testing)."""
        self._models[model.metadata.name] = model

    def load_model_artifact(
        self,
        model_name: str,
        model_path: str | Path,
        metadata: ModelMetadata,
        feature_schema_version: str = "1.0.0",
    ) -> ModelLifecycleStatus:
        """Load a model from disk into the service."""
        model = self.loader.load_model(model_path, metadata, feature_schema_version)
        if model is None:
            return ModelLifecycleStatus.UNAVAILABLE

        self._models[model_name] = model
        return ModelLifecycleStatus.READY

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        if model_name in self._models:
            del self._models[model_name]

    def reload_model(
        self,
        model_name: str,
        new_path: str | Path,
        metadata: ModelMetadata,
        feature_schema_version: str = "1.0.0",
    ) -> ModelLifecycleStatus:
        """Safely reload a model. Preserves previous valid model if reload fails."""
        new_model = self.loader.load_model(new_path, metadata, feature_schema_version)
        if new_model is None:
            logger.warning(f"Reload failed for {model_name}. Preserving previously loaded model.")
            return ModelLifecycleStatus.ERROR if model_name not in self._models else ModelLifecycleStatus.READY

        self._models[model_name] = new_model
        return ModelLifecycleStatus.READY

    def predict(
        self,
        model_name: str,
        feature_vector: MLFeatureVector,
    ) -> InferenceResult:
        """Execute ML model inference on a feature vector."""
        ts = feature_vector.timestamp

        # 1. Model Availability Check
        if not self.is_model_loaded(model_name):
            return InferenceResult(
                model_name=model_name,
                model_version="unknown",
                feature_schema_version=feature_vector.schema_version,
                prediction=None,
                status=InferenceStatus.MODEL_UNAVAILABLE,
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                error_message=f"Model '{model_name}' is not loaded or artifact is unavailable.",
            )

        model = self._models[model_name]

        # 2. Input Validity Check
        if not feature_vector.valid or not feature_vector.values or any(math.isnan(v) or math.isinf(v) for v in feature_vector.values):
            return InferenceResult(
                model_name=model_name,
                model_version=model.metadata.version,
                feature_schema_version=feature_vector.schema_version,
                prediction=None,
                status=InferenceStatus.INVALID_INPUT,
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                model_metadata=model.metadata,
                error_message="Feature vector contains invalid or non-finite values.",
            )

        # 3. Schema Compatibility Check
        if not model.validate_schema(feature_vector):
            return InferenceResult(
                model_name=model_name,
                model_version=model.metadata.version,
                feature_schema_version=feature_vector.schema_version,
                prediction=None,
                status=InferenceStatus.SCHEMA_MISMATCH,
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                model_metadata=model.metadata,
                error_message=f"Feature schema mismatch. Model expects {model.metadata.input_shape} inputs at schema version {model.feature_schema_version}.",
            )

        # 4. Predict Execution
        t0 = time.perf_counter()
        try:
            pred = model.predict(feature_vector)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            if isinstance(pred, InferenceResult):
                return pred.model_copy(
                    update={
                        "latency_ms": latency_ms if pred.latency_ms == 0.0 else pred.latency_ms,
                        "quality": pred.quality if hasattr(pred, "quality") else feature_vector.quality,
                        "model_metadata": pred.model_metadata or model.metadata,
                    }
                )

            return InferenceResult(
                model_name=model_name,
                model_version=model.metadata.version,
                feature_schema_version=feature_vector.schema_version,
                prediction=pred,
                status=InferenceStatus.SUCCESS,
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                quality=feature_vector.quality,
                latency_ms=latency_ms,
                model_metadata=model.metadata,
            )
        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            logger.error(f"Inference exception in model {model_name}: {e}")
            return InferenceResult(
                model_name=model_name,
                model_version=model.metadata.version,
                feature_schema_version=feature_vector.schema_version,
                prediction=None,
                status=InferenceStatus.INFERENCE_ERROR,
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                latency_ms=latency_ms,
                model_metadata=model.metadata,
                error_message=str(e),
            )
