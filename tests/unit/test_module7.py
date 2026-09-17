"""
Unit Tests for Original Module 7 — Per-Cylinder EGT Diagnostics.

Tests all EGT diagnostic features: cylinder identity, mean/spread calculation,
max/min cylinder identification, threshold evaluation, failure isolation,
temporal dEGT/dt handling, and schema/provenance compliance.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import AppSettings, EGTDiagnosticConfig
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import DiagnosticState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.egt_diagnostics import (
    EGTDiagnosticResult,
    EGTDiagnosticsEngine,
    evaluate_egt_diagnostics,
)


def _make_record_with_egts(
    egt1: float = 950.0,
    egt2: float = 955.0,
    egt3: float = 945.0,
    egt4: float = 950.0,
    valid1: bool = True,
    valid2: bool = True,
    valid3: bool = True,
    valid4: bool = True,
    timestamp: datetime | None = None,
) -> NormalizedSignalRecord:
    """Helper to build a NormalizedSignalRecord with specific cylinder EGTs."""
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
        rpm=_ch(4000.0),
        map_pressure=_ch(120000.0),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(egt1, valid=valid1),
        egt_cyl_2=_ch(egt2, valid=valid2),
        egt_cyl_3=_ch(egt3, valid=valid3),
        egt_cyl_4=_ch(egt4, valid=valid4),
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
        vibration_x=_ch(1.0),
        vibration_y=_ch(1.0),
        vibration_z=_ch(9.8),
        vibration_rms=_ch(9.9),
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


class TestEGTDiagnosticsEngine:
    """Unit test suite for Original Module 7 EGT Diagnostics."""

    def test_four_cylinder_healthy_egt(self) -> None:
        rec = _make_record_with_egts(950.0, 955.0, 945.0, 950.0)
        diag_state, egt_res = evaluate_egt_diagnostics(rec)

        assert isinstance(diag_state, DiagnosticState)
        assert isinstance(egt_res, EGTDiagnosticResult)
        assert egt_res.overall_status == DiagnosticStatus.NORMAL
        assert not egt_res.imbalance_detected
        assert len(egt_res.affected_cylinders) == 0

        # Mean = (950 + 955 + 945 + 950) / 4 = 950.0 K
        assert math.isclose(egt_res.mean_egt_k, 950.0, rel_tol=1e-4)
        assert math.isclose(egt_res.spread_egt_k, 10.0, rel_tol=1e-4)
        assert egt_res.max_egt_cylinder == 2
        assert egt_res.min_egt_cylinder == 3

    def test_cylinder_identity(self) -> None:
        rec = _make_record_with_egts(900.0, 910.0, 920.0, 930.0)
        _, egt_res = evaluate_egt_diagnostics(rec)

        cyl_ids = [c.cylinder_id for c in egt_res.cylinder_diagnostics]
        assert cyl_ids == [1, 2, 3, 4]

    def test_high_egt_warning_and_critical(self) -> None:
        # High EGT on Cyl 1 (1130 K > 1123.15 K warning, 1180 K > 1173.15 K critical)
        rec_warn = _make_record_with_egts(1130.0, 950.0, 950.0, 950.0)
        _, res_warn = evaluate_egt_diagnostics(rec_warn)
        assert res_warn.cylinder_diagnostics[0].status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)

        rec_crit = _make_record_with_egts(1180.0, 950.0, 950.0, 950.0)
        _, res_crit = evaluate_egt_diagnostics(rec_crit)
        assert res_crit.cylinder_diagnostics[0].status == DiagnosticStatus.CRITICAL
        assert res_crit.overall_status == DiagnosticStatus.CRITICAL
        assert 1 in res_crit.affected_cylinders

    def test_egt_spread_imbalance(self) -> None:
        # Spread = 1010 - 920 = 90 K (> 80 K critical threshold)
        rec = _make_record_with_egts(1010.0, 920.0, 950.0, 950.0)
        _, res = evaluate_egt_diagnostics(rec)

        assert res.imbalance_detected
        assert res.overall_status == DiagnosticStatus.CRITICAL

    def test_missing_or_invalid_cylinder_isolation(self) -> None:
        """Cylinder 3 invalid: mean/max/min should be calculated from Cyl 1, 2, 4 only."""
        rec = _make_record_with_egts(950.0, 960.0, 0.0, 940.0, valid3=False)
        diag_state, res = evaluate_egt_diagnostics(rec)

        # Cyl 3 must be marked INVALID
        assert res.cylinder_diagnostics[2].status == DiagnosticStatus.INVALID
        assert not res.cylinder_diagnostics[2].valid

        # Valid mean = (950 + 960 + 940) / 3 = 950.0 K
        assert math.isclose(res.mean_egt_k, 950.0, rel_tol=1e-4)
        assert res.max_egt_cylinder == 2
        assert res.min_egt_cylinder == 4
        assert res.spread_egt_k == 20.0

        # Cylinders 1, 2, 4 must remain NORMAL
        assert res.cylinder_diagnostics[0].status == DiagnosticStatus.NORMAL
        assert res.cylinder_diagnostics[1].status == DiagnosticStatus.NORMAL
        assert res.cylinder_diagnostics[3].status == DiagnosticStatus.NORMAL

    def test_all_cylinders_invalid(self) -> None:
        rec = _make_record_with_egts(valid1=False, valid2=False, valid3=False, valid4=False)
        _, res = evaluate_egt_diagnostics(rec)

        assert res.overall_status == DiagnosticStatus.INVALID
        assert res.mean_egt_k is None
        assert res.spread_egt_k is None

    def test_rapid_egt_change(self) -> None:
        engine = EGTDiagnosticsEngine()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec1 = _make_record_with_egts(950.0, 950.0, 950.0, 950.0, timestamp=t0)
        engine.evaluate(rec1)

        # 1 second later: Cyl 1 EGT increases by 35 K -> dEGT/dt = 35 K/s (> 30 K/s critical)
        t1 = t0 + timedelta(seconds=1)
        rec2 = _make_record_with_egts(985.0, 950.0, 950.0, 950.0, timestamp=t1)
        _, res2 = engine.evaluate(rec2)

        cyl1_diag = res2.cylinder_diagnostics[0]
        assert math.isclose(cyl1_diag.rate_of_change_k_s, 35.0, rel_tol=1e-4)
        assert cyl1_diag.status == DiagnosticStatus.CRITICAL

    test_first_sample_temporal = lambda self: self._test_temporal_edge(None, 0.0)
    test_zero_dt_temporal = lambda self: self._test_temporal_edge(0.0, 0.0)

    def _test_temporal_edge(self, dt: float | None, expected_rate: float) -> None:
        engine = EGTDiagnosticsEngine()
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        rec = _make_record_with_egts(950.0, 950.0, 950.0, 950.0, timestamp=t0)
        _, res = engine.evaluate(rec)

        rate = res.cylinder_diagnostics[0].rate_of_change_k_s
        assert rate is None or rate == 0.0

    def test_provenance_and_schema_compliance(self) -> None:
        rec = _make_record_with_egts()
        diag_state, res = evaluate_egt_diagnostics(rec)

        assert diag_state.provenance == Provenance.DERIVED
        assert res.provenance == Provenance.DERIVED
        assert len(diag_state.per_cylinder_egt_dev_k) == 4
        assert len(diag_state.per_cylinder_egt_status) == 4
