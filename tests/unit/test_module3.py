"""
Unit & Integration Tests for Original Module 3 — Telemetry Validation, Integrity & Anti-Replay Security.

Validates all Module 3 requirements:
    - Raw Range Validation & Failure Isolation
    - Stale Signal & Timestamp Freshness Detection
    - Packet Integrity (HMAC Signer & Verifier, constant-time comparison, payload tampering rejection)
    - Anti-Replay Protection (Monotonic sequence check, duplicate rejection, reset)
    - TelemetryValidator 6-stage pipeline & Security vs Channel Failure Isolation
    - Quality & Provenance Preservation
"""

import os
import time
from datetime import datetime, timedelta, timezone
import pytest

from src.core.config import AppSettings, load_settings
from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import (
    PacketSigner,
    PacketVerifier,
    ReplayProtector,
    StaleSignalDetector,
)
from src.l1_data.telemetry_validator import TelemetryValidator, ValidationResult


@pytest.fixture
def base_raw_record() -> RawSignalRecord:
    rec = RawSignalRecord(
        sequence_number=1,
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
        map_counts=2048,
        adc_vref_counts=4095,
        crank_period_us=15000.0,
        fuel_pulse_hz=100.0,
        accel_counts_xyz=(0, 0, 1000),
        ambient_temp_c=20.0,
        ambient_press_pa=101325.0,
        signal_quality=SignalQuality(score=1.0),
    )
    return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})


class TestPacketIntegrity:
    """Validate HMAC PacketSigner and PacketVerifier."""

    def test_valid_hmac_accepted(self, base_raw_record: RawSignalRecord) -> None:
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record, secret_key="test_secret_123")
        assert verifier.verify_signature(base_raw_record, sig, secret_key="test_secret_123")

    def test_modified_payload_rejected(self, base_raw_record: RawSignalRecord) -> None:
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record, secret_key="test_secret_123")

        # Alter payload field (EGT)
        tampered_record = base_raw_record.model_copy(update={"egt_cyl1_hot_uv": 55000.0})
        assert not verifier.verify_signature(tampered_record, sig, secret_key="test_secret_123")

    def test_modified_timestamp_rejected(self, base_raw_record: RawSignalRecord) -> None:
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record, secret_key="test_secret_123")

        tampered_record = base_raw_record.model_copy(
            update={"timestamp": base_raw_record.timestamp + timedelta(seconds=10)}
        )
        assert not verifier.verify_signature(tampered_record, sig, secret_key="test_secret_123")

    def test_modified_sequence_rejected(self, base_raw_record: RawSignalRecord) -> None:
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record, secret_key="test_secret_123")

        tampered_record = base_raw_record.model_copy(update={"sequence_number": 999})
        assert not verifier.verify_signature(tampered_record, sig, secret_key="test_secret_123")

    def test_wrong_secret_key_rejected(self, base_raw_record: RawSignalRecord) -> None:
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record, secret_key="correct_key")
        assert not verifier.verify_signature(base_raw_record, sig, secret_key="wrong_key")

    def test_secret_key_from_environment(self, base_raw_record: RawSignalRecord, monkeypatch) -> None:
        monkeypatch.setenv("TELEMETRY_SECRET_KEY", "env_secret_key_xyz")
        signer = PacketSigner()
        verifier = PacketVerifier()

        sig = signer.sign_record(base_raw_record)
        assert verifier.verify_signature(base_raw_record, sig)


class TestReplayProtection:
    """Validate ReplayProtector sequence state & anti-replay checks."""

    def test_first_valid_sequence_accepted(self) -> None:
        protector = ReplayProtector()
        valid, msg = protector.validate_sequence(sequence_number=1, source_id="unit_test")
        assert valid
        assert "valid" in msg.lower()

    def test_duplicate_sequence_rejected(self) -> None:
        protector = ReplayProtector()
        protector.validate_sequence(sequence_number=10, source_id="unit_test")

        valid, msg = protector.validate_sequence(sequence_number=10, source_id="unit_test")
        assert not valid
        assert "replay attack" in msg.lower()

    def test_replayed_older_sequence_rejected(self) -> None:
        protector = ReplayProtector()
        protector.validate_sequence(sequence_number=20, source_id="unit_test")

        valid, msg = protector.validate_sequence(sequence_number=15, source_id="unit_test")
        assert not valid
        assert "replay attack" in msg.lower()

    def test_valid_next_sequence_accepted(self) -> None:
        protector = ReplayProtector()
        protector.validate_sequence(sequence_number=1, source_id="unit_test")
        valid, _ = protector.validate_sequence(sequence_number=2, source_id="unit_test")
        assert valid

    def test_reset_clears_replay_state(self) -> None:
        protector = ReplayProtector()
        protector.validate_sequence(sequence_number=100, source_id="unit_test")
        protector.reset(source_id="unit_test")

        valid, _ = protector.validate_sequence(sequence_number=1, source_id="unit_test")
        assert valid


class TestStaleSignalDetection:
    """Validate StaleSignalDetector."""

    def test_fresh_signal_accepted(self, base_raw_record: RawSignalRecord) -> None:
        detector = StaleSignalDetector()
        stale_map = detector.check_stale(base_raw_record)
        assert len(stale_map) == 0

    def test_stale_signal_flagged_on_timeout(self, base_raw_record: RawSignalRecord) -> None:
        detector = StaleSignalDetector()

        t0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        r0 = base_raw_record.model_copy(update={"timestamp": t0})
        detector.check_stale(r0)

        # 10 seconds later (> 3s timeout for EGT)
        t1 = t0 + timedelta(seconds=10)
        r1 = base_raw_record.model_copy(update={"timestamp": t1})
        stale_map = detector.check_stale(r1)

        assert len(stale_map) > 0
        assert "egt_cyl1_hot_uv" in stale_map


class TestTelemetryValidatorPipeline:
    """Validate complete TelemetryValidator pipeline and Failure Isolation."""

    def test_security_failure_rejects_packet(self, base_raw_record: RawSignalRecord) -> None:
        validator = TelemetryValidator()
        # Missing/invalid signature when signature required
        result = validator.validate_packet(base_raw_record, signature="invalid_sig_hex")
        assert not result.accepted
        assert "Security failure" in result.rejection_reason

    def test_replay_failure_rejects_packet(self, base_raw_record: RawSignalRecord) -> None:
        settings = load_settings()
        signer = PacketSigner(settings)
        validator = TelemetryValidator(settings)

        sig1 = signer.sign_record(base_raw_record)
        res1 = validator.validate_packet(base_raw_record, signature=sig1)
        assert res1.accepted

        # Duplicate sequence packet
        res2 = validator.validate_packet(base_raw_record, signature=sig1)
        assert not res2.accepted
        assert "Replay protection failure" in res2.rejection_reason

    def test_range_violation_flags_channel_without_rejecting_packet(self, base_raw_record: RawSignalRecord) -> None:
        settings = load_settings()
        signer = PacketSigner(settings)
        validator = TelemetryValidator(settings)

        # Set EGT cyl 1 out of range (80,000 uV > max 60,000 uV)
        bad_egt_record = base_raw_record.model_copy(update={"egt_cyl1_hot_uv": 80000.0})
        sig = signer.sign_record(bad_egt_record)

        result = validator.validate_packet(bad_egt_record, signature=sig)
        assert result.accepted  # Packet accepted (Failure Isolation)
        assert result.record is not None
        assert "egt_cyl1_hot_uv" in result.invalid_channels
        assert not result.record.signal_quality.valid
        assert result.record.egt_cyl2_hot_uv == 30500.0  # Unaffected channel preserved intact

    def test_structural_malformation_rejects_packet(self, base_raw_record: RawSignalRecord) -> None:
        validator = TelemetryValidator()
        malformed = base_raw_record.model_copy(update={"crank_period_us": -50.0})
        result = validator.validate_packet(malformed)
        assert not result.accepted
        assert "Structural validation failed" in result.rejection_reason
