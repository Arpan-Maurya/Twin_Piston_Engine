"""
Validation Metrics Evaluator — Module 22.

Calculates regression, classification (including Module 13 Nine-Class taxonomy),
anomaly detection, RUL prediction, and Health Index validation metrics.
Calculates metrics ONLY when valid reference labels/ground truth exist.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
from pydantic import BaseModel, Field

from src.core.provenance import FaultClass


class RegressionMetrics(BaseModel):
    """Continuous value regression metrics."""

    mae: float
    rmse: float
    mbe: float  # Mean Bias Error (predicted - expected)
    sample_count: int


class ClassificationMetrics(BaseModel):
    """Categorical classification evaluation metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    sample_count: int


class AnomalyMetrics(BaseModel):
    """Anomaly detection evaluation metrics."""

    detection_rate: float  # True positive rate (TP / (TP + FN))
    false_positive_rate: float  # (FP / (FP + TN))
    false_negative_rate: float  # (FN / (TP + FN))
    sample_count: int


class RULMetrics(BaseModel):
    """Remaining Useful Life estimation metrics."""

    mae_hours: float
    rmse_hours: float
    bounds_coverage_pct: float  # % of true RUL values falling inside [lower, upper]
    sample_count: int


class MetricsCalculator:
    """Calculates scientific validation metrics for backend outputs."""

    @staticmethod
    def calculate_regression_metrics(
        y_true: Sequence[float], y_pred: Sequence[float]
    ) -> RegressionMetrics:
        """Calculate MAE, RMSE, MBE for continuous physical parameters."""
        if not y_true or len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must be non-empty and equal length")

        n = len(y_true)
        errors = [p - t for t, p in zip(y_true, y_pred)]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e ** 2 for e in errors]

        mae = sum(abs_errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        mbe = sum(errors) / n

        return RegressionMetrics(mae=mae, rmse=rmse, mbe=mbe, sample_count=n)

    @staticmethod
    def calculate_classification_metrics(
        y_true: Sequence[FaultClass | int], y_pred: Sequence[FaultClass | int]
    ) -> ClassificationMetrics:
        """Calculate 9-class fault classification metrics preserving Module 13 FaultClass taxonomy (0..8)."""
        if not y_true or len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must be non-empty and equal length")

        n = len(y_true)
        # Authoritative Module 13 FaultClass Taxonomy: 0..8
        num_classes = 9
        cm = [[0] * num_classes for _ in range(num_classes)]

        correct = 0
        for t, p in zip(y_true, y_pred):
            t_id = int(t.value) if isinstance(t, FaultClass) else int(t)
            p_id = int(p.value) if isinstance(p, FaultClass) else int(p)
            if 0 <= t_id < num_classes and 0 <= p_id < num_classes:
                cm[t_id][p_id] += 1
                if t_id == p_id:
                    correct += 1

        accuracy = correct / n

        per_class_f1 = {}
        precisions = []
        recalls = []

        for c in range(num_classes):
            class_name = FaultClass(c).name
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(num_classes) if r != c)
            fn = sum(cm[c][col] for col in range(num_classes) if col != c)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_f1[class_name] = f1
            precisions.append(prec)
            recalls.append(rec)

        macro_prec = sum(precisions) / num_classes
        macro_rec = sum(recalls) / num_classes
        macro_f1 = sum(per_class_f1.values()) / num_classes

        return ClassificationMetrics(
            accuracy=accuracy,
            precision_macro=macro_prec,
            recall_macro=macro_rec,
            f1_macro=macro_f1,
            per_class_f1=per_class_f1,
            confusion_matrix=cm,
            sample_count=n,
        )

    @staticmethod
    def calculate_anomaly_metrics(
        y_true_bool: Sequence[bool], y_pred_bool: Sequence[bool]
    ) -> AnomalyMetrics:
        """Calculate anomaly detection rate, false positive rate, and false negative rate."""
        if not y_true_bool or len(y_true_bool) != len(y_pred_bool):
            raise ValueError("y_true_bool and y_pred_bool must be non-empty and equal length")

        n = len(y_true_bool)
        tp = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if t and p)
        fp = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if not t and p)
        fn = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if t and not p)
        tn = sum(1 for t, p in zip(y_true_bool, y_pred_bool) if not t and not p)

        det_rate = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        return AnomalyMetrics(
            detection_rate=det_rate,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            sample_count=n,
        )

    @staticmethod
    def calculate_rul_metrics(
        y_true_rul: Sequence[float],
        y_pred_rul: Sequence[float],
        bounds: Sequence[tuple[float, float]] | None = None,
    ) -> RULMetrics:
        """Calculate RUL prediction error and confidence bounds coverage percentage."""
        if not y_true_rul or len(y_true_rul) != len(y_pred_rul):
            raise ValueError("y_true_rul and y_pred_rul must be non-empty and equal length")

        n = len(y_true_rul)
        errors = [p - t for t, p in zip(y_true_rul, y_pred_rul)]
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

        coverage_pct = 1.0
        if bounds and len(bounds) == n:
            inside = 0
            for t, (lb, ub) in zip(y_true_rul, bounds):
                if lb <= t <= ub:
                    inside += 1
            coverage_pct = inside / n

        return RULMetrics(
            mae_hours=mae,
            rmse_hours=rmse,
            bounds_coverage_pct=coverage_pct,
            sample_count=n,
        )
