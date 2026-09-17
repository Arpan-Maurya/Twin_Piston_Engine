"""
L3 Machine Learning & Supervision Layer.

Module 12: ML Inference Infrastructure.
Module 13: Anomaly Detection and Nine-Class Fault Classifier.
Module 14: Health Index and Degradation Supervision.
Module 15: Remaining Useful Life with Uncertainty.
Module 16: Mission Phase and Mission Risk.
Module 19: Advisory and Explainability Engine.
"""

from src.l3_ml.advisory_explainability import (
    Advisory,
    AdvisoryCategory,
    AdvisoryEngine,
    AdvisoryPriority,
    DiagnosticAnswer,
    DiagnosticQueryService,
    EvidenceItem,
    ExplainabilityEngine,
    Explanation,
    QuestionType,
)
from src.l3_ml.anomaly_fault import (
    AnomalyDetector,
    NineClassFaultClassifier,
)
from src.l3_ml.health_supervision import HealthSupervisionEngine
from src.l3_ml.mission_risk import (
    MissionPhaseClassifier,
    MissionRiskEngine,
)
from src.l3_ml.ml_infrastructure import (
    BaseMLModel,
    InferenceResult,
    MLFeature,
    MLFeatureVector,
    MLFeatureVectorBuilder,
    MLInferenceService,
    MLModelLoader,
)
from src.l3_ml.rul_estimation import (
    HealthHistoryTracker,
    RULEstimator,
)

__all__ = [
    # Module 12
    "BaseMLModel",
    "InferenceResult",
    "MLFeature",
    "MLFeatureVector",
    "MLFeatureVectorBuilder",
    "MLInferenceService",
    "MLModelLoader",
    # Module 13
    "AnomalyDetector",
    "NineClassFaultClassifier",
    # Module 14
    "HealthSupervisionEngine",
    # Module 15
    "HealthHistoryTracker",
    "RULEstimator",
    # Module 16
    "MissionPhaseClassifier",
    "MissionRiskEngine",
    # Module 19
    "Advisory",
    "AdvisoryCategory",
    "AdvisoryEngine",
    "AdvisoryPriority",
    "DiagnosticAnswer",
    "DiagnosticQueryService",
    "EvidenceItem",
    "ExplainabilityEngine",
    "Explanation",
    "QuestionType",
]
