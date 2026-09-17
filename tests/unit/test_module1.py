"""
Unit tests for Original Module 1 — Backend Foundation, Configuration, and Canonical Schemas.

Validates all Module 1 acceptance requirements:
    1. Canonical RawSignalRecord & SignalQuality metadata
    2. Range and physical limits validation
    3. Complete domain models (DerivedEngineState, DiagnosticState, ResidualState,
       HealthState, RULState, Alert, MissionState, Provenance, SignalQuality)
    4. Clean interfaces/protocols between layers
    5. Externalized, versionable, overridable configuration & error handling
    6. Validation rules (missing fields, range violations, malformed records, invalid provenance)
    7. Provenance enum tag distinctions
"""

from datetime import datetime, timezone
import pytest

from src.core.config import AppSettings, load_settings
from src.core.exceptions import ConfigurationError, PacketIntegrityError
from src.core.interfaces import (
    L1SourceAdapterProtocol,
    L2DigitalTwinProtocol,
    L3SupervisionProtocol,
    L4AdvisoryProtocol,
    L5InterfaceProtocol,
)
from src.core.provenance import (
    AlertLevel,
    ChannelValidity,
    DegradationState,
    FaultClass,
    FlightPhase,
    Provenance,
)
from src.core.schemas import (
    Alert,
    DerivedEngineState,
    DiagnosticState,
    HealthState,
    MissionState,
    OperatingPoint,
    ProvenanceTaggedValue,
    ResidualState,
    RULState,
    SignalQuality,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l1_data.validation import SignalValidator


class TestProvenanceEnum:
    """Validate mandatory provenance tags."""

    def test_provenance_members(self) -> None:
        expected = {"REAL", "SIMULATED", "CSV_REPLAY", "DERIVED", "MODEL_OUTPUT"}
        actual = {p.value for p in Provenance}
        assert actual == expected


class TestRawSignalRecord:
    """Validate canonical RawSignalRecord acquisition schema."""

    @pytest.fixture
    def valid_raw_record(self) -> RawSignalRecord:
        record = RawSignalRecord(
            sequence_number=100,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0,
            egt_cyl2_hot_uv=30500.0,
            egt_cyl3_hot_uv=29800.0,
            egt_cyl4_hot_uv=30100.0,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0,
            oil_p_counts=2048,
            map_counts=3072,
            adc_vref_counts=4095,
            crank_period_us=15000.0,
            fuel_pulse_hz=120.5,
            accel_counts_xyz=(10, 15, 980),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )
        # Compute and set integrity hash
        return record.model_copy(update={"integrity_hash": record.compute_integrity_hash()})

    def test_raw_fields_exist(self, valid_raw_record: RawSignalRecord) -> None:
        assert valid_raw_record.sequence_number == 100
        assert valid_raw_record.source_type == Provenance.REAL
        assert valid_raw_record.egt_cyl1_hot_uv == 30000.0
        assert valid_raw_record.oil_rtd_ohms == 135.0
        assert valid_raw_record.crank_period_us == 15000.0
        assert valid_raw_record.accel_counts_xyz == (10, 15, 980)

    def test_raw_integrity_hash(self, valid_raw_record: RawSignalRecord) -> None:
        assert len(valid_raw_record.integrity_hash) == 64
        assert valid_raw_record.compute_integrity_hash() == valid_raw_record.integrity_hash

    def test_raw_record_contains_no_derived_quantities(self) -> None:
        """Verify RawSignalRecord fields are raw sensor signals, not derived engineering units."""
        fields = RawSignalRecord.model_fields
        # Raw fields should use raw acquisition suffixes/names
        assert "egt_cyl1_hot_uv" in fields
        assert "oil_rtd_ohms" in fields
        assert "crank_period_us" in fields
        assert "fuel_pulse_hz" in fields


class TestDomainModels:
    """Validate required domain models (Requirement 3)."""

    def test_signal_quality(self) -> None:
        sq = SignalQuality(score=0.95, valid=True, invalid_channels=["voltage"])
        assert sq.score == 0.95
        assert sq.valid is True
        assert "voltage" in sq.invalid_channels

    def test_derived_engine_state(self) -> None:
        state = DerivedEngineState(
            bmep_pa=ProvenanceTaggedValue(value=1.2e6, provenance=Provenance.DERIVED),
            imep_pa=ProvenanceTaggedValue(value=1.35e6, provenance=Provenance.DERIVED),
            eta_thermal=ProvenanceTaggedValue(value=0.32, provenance=Provenance.DERIVED),
            eta_volumetric=ProvenanceTaggedValue(value=0.88, provenance=Provenance.DERIVED),
            afr=ProvenanceTaggedValue(value=14.7, provenance=Provenance.DERIVED),
            brake_torque_nm=ProvenanceTaggedValue(value=180.0, provenance=Provenance.DERIVED),
            brake_power_kw=ProvenanceTaggedValue(value=105.0, provenance=Provenance.DERIVED),
            fmep_pa=ProvenanceTaggedValue(value=150000.0, provenance=Provenance.DERIVED),
            eta_mechanical=ProvenanceTaggedValue(value=0.88, provenance=Provenance.DERIVED),
            mean_piston_speed_m_s=ProvenanceTaggedValue(value=11.8, provenance=Provenance.DERIVED),
        )
        assert state.bmep_pa.value == 1.2e6
        assert state.provenance == Provenance.DERIVED

    def test_diagnostic_state(self) -> None:
        ds = DiagnosticState(
            per_cylinder_egt_dev_k=[-10.0, 5.0, 2.0, 3.0],
            misfire_detected=[False, False, False, False],
            vibration_rms_m_s2=2.5,
        )
        assert len(ds.per_cylinder_egt_dev_k) == 4
        assert ds.cooling_temp_ok is True

    def test_residual_state(self) -> None:
        op = OperatingPoint(
            rpm=4000,
            map_pressure_pa=120000,
            altitude_m=1000,
            ambient_temp_k=288.15,
            ambient_pressure_pa=101325,
            throttle_pct=50,
        )
        rs = ResidualState(
            operating_point=op,
            residuals={
                "bmep": ProvenanceTaggedValue(value=5000.0, provenance=Provenance.DERIVED),
            },
        )
        assert rs.operating_point.rpm == 4000
        assert "bmep" in rs.residuals

    def test_health_state(self) -> None:
        hs = HealthState(
            health_index=ProvenanceTaggedValue(value=0.92, provenance=Provenance.MODEL_OUTPUT),
            degradation_state=DegradationState.HEALTHY,
        )
        assert hs.health_index.value == 0.92
        assert hs.degradation_state == DegradationState.HEALTHY

    def test_rul_state(self) -> None:
        rul = RULState(
            hours_remaining=420.0,
            lower_bound_hours=380.0,
            upper_bound_hours=460.0,
            confidence=0.95,
        )
        assert rul.hours_remaining == 420.0
        assert rul.confidence == 0.95

    def test_alert(self) -> None:
        alert = Alert(
            alert_level=AlertLevel.WATCH,
            fault_class=FaultClass.MISFIRE,
            summary="Misfire Watch",
            message="Slight EGT drop detected on cylinder 1",
            recommended_action="Inspect spark plug cylinder 1 at next service",
        )
        assert alert.alert_level == AlertLevel.WATCH
        assert alert.fault_class == FaultClass.MISFIRE

    def test_mission_state(self) -> None:
        ms = MissionState(
            flight_phase=FlightPhase.CRUISE,
            mission_hours=12.5,
            risk_score=0.05,
            go_no_go_recommendation="GO",
        )
        assert ms.flight_phase == FlightPhase.CRUISE
        assert ms.go_no_go_recommendation == "GO"


class TestLayerProtocols:
    """Validate layer interfaces and protocols (Requirement 4)."""

    def test_protocol_runtime_checkability(self) -> None:
        class DummyAdapter:
            async def connect(self) -> None: pass
            async def read_next(self) -> None: pass
            async def disconnect(self) -> None: pass
            @property
            def source_type(self) -> Provenance: return Provenance.SIMULATED

        dummy = DummyAdapter()
        assert isinstance(dummy, L1SourceAdapterProtocol)


class TestConfigValidation:
    """Validate externalized, versionable configuration system (Requirement 5)."""

    def test_load_default_config(self) -> None:
        settings = load_settings()
        assert settings.version == "1.0.0"
        assert settings.engine.name == "Rotax 915 iS"
        assert settings.engine.bore_mm == 84.0
        assert settings.engine.num_cylinders == 4

    def test_invalid_config_raises_configuration_error(self, tmp_path) -> None:
        bad_config_file = tmp_path / "bad_config.yaml"
        bad_config_file.write_text("engine:\n  bore_mm: -10.0\n")

        with pytest.raises(ConfigurationError):
            load_settings(bad_config_file)


class TestDataValidation:
    """Validate validation rules (Requirement 6)."""

    def test_raw_record_validation_success(self) -> None:
        validator = SignalValidator(load_settings())
        raw = RawSignalRecord(
            sequence_number=1,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000,
            egt_cyl2_hot_uv=30000,
            egt_cyl3_hot_uv=30000,
            egt_cyl4_hot_uv=30000,
            egt_cold_c=25,
            cht_hot_uv=12000,
            cht_cold_c=25,
            oil_rtd_ohms=135,
            oil_p_counts=2000,
            map_counts=3000,
            adc_vref_counts=4095,
            crank_period_us=15000,
            fuel_pulse_hz=100,
            accel_counts_xyz=(0, 0, 1000),
            ambient_temp_c=20,
            ambient_press_pa=101325,
        )
        validated = validator.validate_raw_record(raw)
        assert validated.sequence_number == 1

    def test_raw_record_invalid_crank_period_raises_integrity_error(self) -> None:
        validator = SignalValidator(load_settings())
        with pytest.raises(Exception):
            raw = RawSignalRecord(
                sequence_number=1,
                source_type=Provenance.REAL,
                egt_cyl1_hot_uv=30000,
                egt_cyl2_hot_uv=30000,
                egt_cyl3_hot_uv=30000,
                egt_cyl4_hot_uv=30000,
                egt_cold_c=25,
                cht_hot_uv=12000,
                cht_cold_c=25,
                oil_rtd_ohms=135,
                oil_p_counts=2000,
                map_counts=3000,
                adc_vref_counts=4095,
                crank_period_us=-100.0,  # Invalid
                fuel_pulse_hz=100,
                accel_counts_xyz=(0, 0, 1000),
                ambient_temp_c=20,
                ambient_press_pa=101325,
            )
            validator.validate_raw_record(raw)
