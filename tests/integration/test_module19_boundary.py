"""
Integration and Architecture Boundary Tests for Original Module 19:
Advisory and Explainability Engine.
"""

from datetime import datetime, timezone
import pytest

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.advisory_explainability import (
    AdvisoryEngine,
    DiagnosticQueryService,
    ExplainabilityEngine,
    QuestionType,
)


class TestModule19ArchitectureBoundary:
    """Architectural boundary enforcement tests for Module 19."""

    def test_raw_signal_record_direct_input_raises_type_error(self):
        """Verify Module 19 services reject RawSignalRecord with explicit TypeError."""
        raw_rec = RawSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            source_type=Provenance.SIMULATED,
            egt_cyl1_hot_uv=20000.0,
            egt_cyl2_hot_uv=20000.0,
            egt_cyl3_hot_uv=20000.0,
            egt_cyl4_hot_uv=20000.0,
            egt_cold_c=25.0,
            cht_hot_uv=15000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=130.0,
            oil_p_counts=2000,
            map_counts=2000,
            adc_vref_counts=4095,
            crank_period_us=12000.0,
            fuel_pulse_hz=50.0,
            accel_counts_xyz=(10, 10, 10),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(valid=True, quality=1.0),
        )

        explainer = ExplainabilityEngine()
        adviser = AdvisoryEngine()
        query_service = DiagnosticQueryService()

        # 1. ExplainabilityEngine direct input rejection
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            explainer.explain_health(raw_rec)  # type: ignore

        # 2. AdvisoryEngine direct input rejection
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            adviser.generate_advisories(health_state=raw_rec)  # type: ignore

        # 3. DiagnosticQueryService direct input rejection
        with pytest.raises(TypeError, match="STRICT BOUNDARY VIOLATION"):
            query_service.answer_query(QuestionType.HEALTH_STATUS, health_state=raw_rec)  # type: ignore

    def test_zero_autonomous_control_commands(self):
        """Verify Module 19 output schemas contain zero actuation/control fields."""
        adviser = AdvisoryEngine()
        advisories = adviser.generate_advisories()

        for adv in advisories:
            adv_dict = adv.model_dump()
            assert "actuator_command" not in adv_dict
            assert "throttle_command" not in adv_dict
            assert "abort_mission_command" not in adv_dict
            assert "flight_control" not in adv_dict
