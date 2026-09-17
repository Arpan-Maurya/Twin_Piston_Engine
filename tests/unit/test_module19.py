"""
Unit tests for Original Module 19:
Advisory and Explainability Engine.
"""

from datetime import datetime, timezone
import pytest

from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    DegradationState,
    FaultClassificationResult,
    HealthState,
    MissionState,
    ProvenanceTaggedValue,
    RULState,
    SignalQuality,
)
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


class TestExplainabilityEngine:
    """Tests for ExplainabilityEngine explanation and evidence logic."""

    @pytest.fixture
    def engine(self):
        return ExplainabilityEngine()

    def test_explain_nominal_fault(self, engine):
        f_res = FaultClassificationResult(
            predicted_class=FaultClass.NOMINAL,
            class_name="NOMINAL",
            confidence=0.99,
        )
        exp = engine.explain_fault(f_res)
        assert isinstance(exp, Explanation)
        assert "NOMINAL" in exp.finding
        assert exp.quality == 1.0

    def test_explain_detected_fault(self, engine):
        f_res = FaultClassificationResult(
            predicted_class=FaultClass.MISFIRE,
            class_id=1,
            class_name="MISFIRE",
            confidence=0.88,
        )
        exp = engine.explain_fault(f_res)
        assert "MISFIRE" in exp.finding
        assert "DETECTED" in exp.finding

    def test_explain_health(self, engine):
        h_state = HealthState(
            health_index=ProvenanceTaggedValue(value=0.75, provenance=Provenance.DERIVED, valid=True),
            degradation_state=DegradationState.WARNING,
            trend="DOWNWARD",
            component_health={"lubrication": 0.6},
        )
        exp = engine.explain_health(h_state)
        assert "0.75" in exp.finding
        assert "WARNING" in exp.finding

    def test_explain_rul(self, engine):
        r_state = RULState(
            hours_remaining=120.0,
            lower_bound_hours=90.0,
            upper_bound_hours=150.0,
            status=InferenceStatus.SUCCESS,
        )
        exp = engine.explain_rul(r_state)
        assert "120.0 hours" in exp.finding

    def test_explain_mission_risk(self, engine):
        m_state = MissionState(
            risk_score=0.45,
            risk_level="MEDIUM",
        )
        exp = engine.explain_mission_risk(m_state)
        assert "MEDIUM" in exp.finding
        assert "0.450" in exp.finding


class TestAdvisoryEngine:
    """Tests for AdvisoryEngine decision-support guidance generation."""

    @pytest.fixture
    def engine(self):
        return AdvisoryEngine()

    def test_nominal_advisory(self, engine):
        advisories = engine.generate_advisories()
        assert len(advisories) == 1
        assert advisories[0].category == AdvisoryCategory.CONTINUE_MONITORING
        assert advisories[0].priority == AdvisoryPriority.INFORMATION

    def test_fault_advisory(self, engine):
        f_res = FaultClassificationResult(
            predicted_class=FaultClass.MISFIRE,
            class_name="MISFIRE",
            confidence=0.92,
        )
        advisories = engine.generate_advisories(fault_result=f_res)
        assert len(advisories) >= 1
        assert advisories[0].priority == AdvisoryPriority.CRITICAL
        assert "Inspect" in advisories[0].title


class TestDiagnosticQueryService:
    """Tests for DiagnosticQueryService operator question answering."""

    @pytest.fixture
    def service(self):
        return DiagnosticQueryService()

    def test_health_status_query(self, service):
        h_state = HealthState(
            health_index=ProvenanceTaggedValue(value=0.92, provenance=Provenance.DERIVED, valid=True),
            degradation_state=DegradationState.HEALTHY,
        )
        ans = service.answer_query(QuestionType.HEALTH_STATUS, health_state=h_state)
        assert isinstance(ans, DiagnosticAnswer)
        assert "0.92" in ans.answer

    def test_what_changed_query(self, service):
        h_curr = HealthState(
            health_index=ProvenanceTaggedValue(value=0.70, provenance=Provenance.DERIVED, valid=True),
            degradation_state=DegradationState.WARNING,
        )
        h_prev = HealthState(
            health_index=ProvenanceTaggedValue(value=0.95, provenance=Provenance.DERIVED, valid=True),
            degradation_state=DegradationState.HEALTHY,
        )
        ans = service.answer_query(QuestionType.WHAT_CHANGED, health_state=h_curr, previous_health_state=h_prev)
        assert "0.95" in ans.answer
        assert "0.70" in ans.answer

    def test_insufficient_history_query(self, service):
        ans = service.answer_query(QuestionType.WHAT_CHANGED)
        assert "INSUFFICIENT_HISTORY" in ans.answer
