"""
Unit Tests for Original Module 10 — Misfire and Combustion Stability Diagnostics.

Tests normal operation, EGT drop/deviation indicators, crank speed fluctuation,
vibration evidence, multi-signal evidence fusion, failure isolation, and robustness.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import CombustionStabilityState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.misfire_classifier import (
    MisfireDiagnosticResult,
    MisfireDetector,
    evaluate_misfire_detector,
)


def _make_record_combustion(
    egt1: float = 950.0,
    egt2: float = 950.0,
    egt3: float = 950.0,
    egt4: float = 950.0,
    rpm_val: float = 4000.0,
    vib_rms: float = 5.0,
    valid1: bool = True,
    timestamp: datetime | None = None,
) -> NormalizedSignalRecord:
    """Helper to build a NormalizedSignalRecord for combustion stability testing."""
    ts = timestamp or datetime.now(timezone.utc)

    def _ch(val: float, valid: bool = True) -> ChannelValue:
        return ChannelValue(
            value=val,
            provenance=Provenance.DERIVED,
            valid=valid,
            validity_reason=ChannelValidity.VALID if valid else ChannelValidity.INVALID_RANGE,
            quality=1.0 if valid else 0.0,
        )

    return NormalizedSignalRecord(
        timestamp=ts,
        sequence_number=1,
        source_id="test_source",
        provenance=Provenance.DERIVED,
        integrity_hash="test_hash",
        rpm=_ch(rpm_val),
        map_pressure=_ch(120000.0),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(egt1, valid=valid1),
        egt_cyl_2=_ch(egt2),
        egt_cyl_3=_ch(egt3),
        egt_cyl_4=_ch(egt4),
        cht_cyl_1=_ch(380.0),
        cht_cyl_2=_ch(380.0),
        cht_cyl_3=_ch(380.0),
        cht_cyl_4=_ch(380.0),
        oil_temp=_ch(360.0),
        oil_pressure=_ch(400000.0),
        coolant_temp=_ch(360.0),
        fuel_flow=_ch(0.005),
        fuel_pressure=_ch(350000.0),
        intake_air_temp=_ch(295.0),
        ambient_pressure=_ch(101325.0),
        ambient_temp=_ch(288.15),
        voltage=_ch(13.8),
        current=_ch(15.0),
        vibration_x=_ch(vib_rms / math.sqrt(3)),
        vibration_y=_ch(vib_rms / math.sqrt(3)),
        vibration_z=_ch(vib_rms / math.sqrt(3)),
        vibration_rms=_ch(vib_rms),
        propeller_speed=_ch(1656.0),
        boost_pressure=_ch(18675.0),
        wastegate_duty=_ch(40.0),
        lambda_sensor=_ch(1.0),
        ignition_timing_cyl_1=_ch(25.0),
        ignition_timing_cyl_2=_ch(25.0),
        ignition_timing_cyl_3=_ch(25.0),
        ignition_timing_cyl_4=_ch(25.0),
        altitude=_ch(1000.0),
        engine_hours=_ch(10.0),
    )


class TestMisfireDetector:
    """Unit test suite for Original Module 10 Misfire & Combustion Stability."""

    def test_normal_stable_combustion(self) -> None:
        rec = _make_record_combustion()
        state, res = evaluate_misfire_detector(rec)

        assert isinstance(state, CombustionStabilityState)
        assert isinstance(res, MisfireDiagnosticResult)
        assert state.provenance == Provenance.DERIVED
        assert state.overall_combustion_status == DiagnosticStatus.NORMAL
        assert state.misfire_detected == [False, False, False, False]
        assert len(res.affected_cylinders) == 0

    def test_sustained_egt_deviation_below_mean(self) -> None:
        # Cyl 1 EGT is 880 K vs mean ~945 K (dev = -65 K < -40 K threshold)
        rec = _make_record_combustion(egt1=880.0, egt2=960.0, egt3=970.0, egt4=970.0)
        state, res = evaluate_misfire_detector(rec)

        assert res.cylinder_evidence[0].egt_evidence > 0.0
        assert res.cylinder_evidence[0].status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)

    def test_sudden_egt_drop_temporal(self) -> None:
        detector = MisfireDetector()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec1 = _make_record_combustion(egt1=950.0, timestamp=t0)
        detector.evaluate(rec1)

        # 1 second later: Cyl 1 EGT drops by 40 K -> -40 K/s drop rate
        t1 = t0 + timedelta(seconds=1)
        rec2 = _make_record_combustion(egt1=910.0, timestamp=t1)
        state2, res2 = detector.evaluate(rec2)

        assert res2.cylinder_evidence[0].egt_evidence > 0.0

    def test_crank_speed_fluctuation(self) -> None:
        detector = MisfireDetector()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec1 = _make_record_combustion(rpm_val=4000.0, timestamp=t0)
        detector.evaluate(rec1)

        # 1 second later: RPM drops by 100 RPM -> rate 100 RPM/s (> 50 threshold)
        t1 = t0 + timedelta(seconds=1)
        rec2 = _make_record_combustion(rpm_val=3900.0, timestamp=t1)
        _, res2 = detector.evaluate(rec2)

        assert res2.cylinder_evidence[0].crank_evidence > 0.0

    def test_elevated_vibration_evidence(self) -> None:
        rec = _make_record_combustion(vib_rms=25.0)  # Elevated vibration (> 15 m/s2 threshold)
        _, res = evaluate_misfire_detector(rec)

        assert res.cylinder_evidence[0].vibration_evidence > 0.0

    def test_multi_signal_combined_possible_misfire(self) -> None:
        detector = MisfireDetector()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec1 = _make_record_combustion(egt1=950.0, rpm_val=4000.0, vib_rms=5.0, timestamp=t0)
        detector.evaluate(rec1)

        # Combined evidence: Cyl 1 EGT drop + RPM fluctuation + high vibration
        t1 = t0 + timedelta(seconds=1)
        rec2 = _make_record_combustion(egt1=860.0, egt2=960.0, egt3=960.0, egt4=960.0, rpm_val=3880.0, vib_rms=22.0, timestamp=t1)
        state2, res2 = detector.evaluate(rec2)

        cyl1 = res2.cylinder_evidence[0]
        assert cyl1.possible_misfire
        assert cyl1.status == DiagnosticStatus.CRITICAL
        assert res2.misfire_detected[0]
        assert 1 in res2.affected_cylinders

    def test_cylinder_failure_isolation(self) -> None:
        """Cylinder 1 invalid: Cyl 1 flagged INVALID, valid cylinders evaluated normally."""
        rec = _make_record_combustion(valid1=False)
        state, res = evaluate_misfire_detector(rec)

        assert res.cylinder_evidence[0].status == DiagnosticStatus.INVALID
        assert not res.misfire_detected[0]
        assert res.cylinder_evidence[1].status == DiagnosticStatus.NORMAL

    def test_temporal_first_sample(self) -> None:
        detector = MisfireDetector()
        rec = _make_record_combustion()
        state, res = detector.evaluate(rec)

        assert res.cylinder_evidence[0].crank_evidence == 0.0
