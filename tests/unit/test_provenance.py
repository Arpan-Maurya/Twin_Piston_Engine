"""
Unit tests for provenance enums, base schemas, and tagged values.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.provenance import (
    AlertLevel,
    ChannelValidity,
    DegradationState,
    FaultClass,
    FlightPhase,
    Provenance,
)
from src.core.schemas import (
    ChannelRange,
    OperatingPoint,
    ProvenanceTaggedValue,
    TimestampedRecord,
    make_tagged,
)


class TestProvenance:
    """Test Provenance enum."""

    def test_all_five_values_exist(self) -> None:
        assert len(Provenance) == 5

    def test_values(self) -> None:
        assert Provenance.REAL == "REAL"
        assert Provenance.SIMULATED == "SIMULATED"
        assert Provenance.CSV_REPLAY == "CSV_REPLAY"
        assert Provenance.DERIVED == "DERIVED"
        assert Provenance.MODEL_OUTPUT == "MODEL_OUTPUT"

    def test_string_serialization(self) -> None:
        assert str(Provenance.REAL) == "Provenance.REAL"
        assert Provenance.REAL.value == "REAL"


class TestFaultClass:
    """Test FaultClass enum."""

    def test_nine_classes(self) -> None:
        assert len(FaultClass) == 9

    def test_nominal_is_zero(self) -> None:
        assert FaultClass.NOMINAL == 0

    def test_all_unique_values(self) -> None:
        values = [fc.value for fc in FaultClass]
        assert len(values) == len(set(values))

    def test_class_names(self) -> None:
        expected = {
            "NOMINAL", "MISFIRE", "DETONATION_KNOCK", "EXHAUST_VALVE_LEAK",
            "INTAKE_BOOST_LEAK", "OIL_DEGRADATION", "COOLING_FAULT",
            "BEARING_WEAR", "SENSOR_FAULT",
        }
        actual = {fc.name for fc in FaultClass}
        assert actual == expected


class TestProvenanceTaggedValue:
    """Test ProvenanceTaggedValue schema."""

    def test_create_valid(self) -> None:
        tv = ProvenanceTaggedValue(
            value=42.0,
            provenance=Provenance.DERIVED,
        )
        assert tv.value == 42.0
        assert tv.provenance == Provenance.DERIVED
        assert tv.valid is True
        assert tv.quality == 1.0
        assert tv.fault_flag is None

    def test_frozen_immutability(self) -> None:
        tv = ProvenanceTaggedValue(value=1.0, provenance=Provenance.REAL)
        with pytest.raises(Exception):  # ValidationError for frozen model
            tv.value = 2.0  # type: ignore[misc]

    def test_as_invalid(self) -> None:
        tv = ProvenanceTaggedValue(value=100.0, provenance=Provenance.REAL)
        invalid = tv.as_invalid(ChannelValidity.INVALID_RANGE, "Value exceeds max")
        assert invalid.valid is False
        assert invalid.validity_reason == ChannelValidity.INVALID_RANGE
        assert invalid.quality == 0.0
        assert invalid.fault_flag == "Value exceeds max"
        # Original is unchanged (immutable)
        assert tv.valid is True

    def test_quality_bounds(self) -> None:
        with pytest.raises(Exception):
            ProvenanceTaggedValue(value=1.0, provenance=Provenance.REAL, quality=1.5)
        with pytest.raises(Exception):
            ProvenanceTaggedValue(value=1.0, provenance=Provenance.REAL, quality=-0.1)


class TestMakeTagged:
    """Test the make_tagged convenience factory."""

    def test_simple(self) -> None:
        tv = make_tagged(99.0, Provenance.SIMULATED)
        assert tv.value == 99.0
        assert tv.provenance == Provenance.SIMULATED
        assert tv.valid is True

    def test_invalid(self) -> None:
        tv = make_tagged(0.0, Provenance.CSV_REPLAY, valid=False, quality=0.0)
        assert tv.valid is False
        assert tv.quality == 0.0


class TestOperatingPoint:
    """Test OperatingPoint schema."""

    def test_valid_operating_point(self) -> None:
        op = OperatingPoint(
            rpm=4000,
            map_pressure_pa=120000,
            altitude_m=1500,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325,
            throttle_pct=60.0,
        )
        assert op.rpm == 4000

    def test_rpm_out_of_range(self) -> None:
        with pytest.raises(Exception):
            OperatingPoint(
                rpm=7000,  # > max 6500
                map_pressure_pa=101325,
                altitude_m=0,
                ambient_temp_k=288,
                ambient_pressure_pa=101325,
                throttle_pct=50,
            )

    def test_frozen(self) -> None:
        op = OperatingPoint(
            rpm=3000, map_pressure_pa=100000, altitude_m=0,
            ambient_temp_k=288, ambient_pressure_pa=101325, throttle_pct=50,
        )
        with pytest.raises(Exception):
            op.rpm = 4000  # type: ignore[misc]


class TestChannelRange:
    """Test ChannelRange schema."""

    def test_contains(self) -> None:
        cr = ChannelRange(min=0, max=6500)
        assert cr.contains(3000)
        assert cr.contains(0)
        assert cr.contains(6500)
        assert not cr.contains(-1)
        assert not cr.contains(6501)


class TestTimestampedRecord:
    """Test TimestampedRecord base class."""

    def test_has_timestamp(self) -> None:
        record = TimestampedRecord()
        assert isinstance(record.timestamp, datetime)
        assert record.timestamp.tzinfo is not None


class TestFlightPhase:
    """Test FlightPhase enum."""

    def test_all_phases(self) -> None:
        phases = {"GROUND", "TAKEOFF", "CLIMB", "CRUISE", "DESCENT", "LANDING"}
        assert {fp.value for fp in FlightPhase} == phases


class TestAlertLevel:
    """Test AlertLevel enum."""

    def test_all_levels(self) -> None:
        assert len(AlertLevel) == 5


class TestDegradationState:
    """Test DegradationState enum."""

    def test_all_states(self) -> None:
        assert len(DegradationState) == 5
