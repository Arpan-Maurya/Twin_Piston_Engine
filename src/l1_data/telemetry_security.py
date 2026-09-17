"""
Telemetry Security & Integrity Protection Module.

PROTOTYPE-GRADE INTEGRITY PROTECTION ONLY — NOT DEFENSE CERTIFIED CRYPTOGRAPHY.

Provides:
    - PacketSigner: Generates HMAC signatures over canonical telemetry payloads.
    - PacketVerifier: Validates HMAC signatures using constant-time comparison.
    - ReplayProtector: Anti-replay protection enforcing monotonic sequence tracking.
    - StaleSignalDetector: Per-channel staleness & missing update detection.

SECURITY RULES:
    1. Secrets MUST come from external configuration or environment variables.
    2. NEVER hardcode or log secret keys.
    3. Use constant-time comparison (hmac.compare_digest) to prevent timing attacks.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


def serialize_canonical_payload(record: RawSignalRecord) -> bytes:
    """Serialize canonical RawSignalRecord payload deterministically for HMAC.

    Includes sequence_number, timestamp, source_type, and raw sensor fields.
    """
    payload_dict: dict[str, Any] = {
        "timestamp": record.timestamp.isoformat(),
        "sequence_number": record.sequence_number,
        "source_type": record.source_type.value,
        "egt_cyl1_hot_uv": record.egt_cyl1_hot_uv,
        "egt_cyl2_hot_uv": record.egt_cyl2_hot_uv,
        "egt_cyl3_hot_uv": record.egt_cyl3_hot_uv,
        "egt_cyl4_hot_uv": record.egt_cyl4_hot_uv,
        "egt_cold_c": record.egt_cold_c,
        "cht_hot_uv": record.cht_hot_uv,
        "cht_cold_c": record.cht_cold_c,
        "oil_rtd_ohms": record.oil_rtd_ohms,
        "oil_p_counts": record.oil_p_counts,
        "map_counts": record.map_counts,
        "adc_vref_counts": record.adc_vref_counts,
        "crank_period_us": record.crank_period_us,
        "fuel_pulse_hz": record.fuel_pulse_hz,
        "accel_counts_xyz": list(record.accel_counts_xyz),
        "ambient_temp_c": record.ambient_temp_c,
        "ambient_press_pa": record.ambient_press_pa,
    }
    return json.dumps(payload_dict, sort_keys=True).encode("utf-8")


class PacketSigner:
    """HMAC Packet Signer for telemetry payload authentication."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def _get_secret(self, override_secret: str | bytes | None = None) -> bytes:
        if override_secret is not None:
            return override_secret.encode("utf-8") if isinstance(override_secret, str) else override_secret

        env_var = self._settings.security.secret_key_env_var
        secret = os.environ.get(env_var)
        if not secret:
            if self._settings.app_env.lower() in ("production", "prod"):
                raise ValueError(
                    f"SEC-002: Production environment requires secret key injected via environment variable '{env_var}'."
                )
            secret = self._settings.security.default_secret_key
        return secret.encode("utf-8")

    def sign_record(self, record: RawSignalRecord, secret_key: str | bytes | None = None) -> str:
        """Sign a RawSignalRecord canonical payload using HMAC.

        Returns hex digest string.
        """
        secret = self._get_secret(secret_key)
        algo = self._settings.security.hmac_algorithm
        payload = serialize_canonical_payload(record)
        return hmac.new(secret, payload, getattr(hashlib, algo)).hexdigest()


class PacketVerifier:
    """HMAC Packet Verifier using constant-time signature comparison."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._signer = PacketSigner(self._settings)

    def verify_signature(
        self,
        record: RawSignalRecord,
        signature: str,
        secret_key: str | bytes | None = None,
    ) -> bool:
        """Verify HMAC signature of RawSignalRecord.

        Uses constant-time hmac.compare_digest to protect against timing attacks.
        """
        if not signature:
            return False

        expected_sig = self._signer.sign_record(record, secret_key)
        return hmac.compare_digest(expected_sig.lower(), signature.lower())


class ReplayProtector:
    """Anti-Replay Protection Service.

    Tracks sequence progression per source and rejects duplicate/replayed packets.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._last_seen_sequence: dict[str, int] = {}

    def validate_sequence(self, sequence_number: int, source_id: str = "default") -> tuple[bool, str]:
        """Validate sequence number against replay attack state.

        Returns (valid: bool, reason: str).
        """
        if source_id in self._last_seen_sequence:
            last_seq = self._last_seen_sequence[source_id]
            if sequence_number <= last_seq:
                return False, f"Replay attack detected: sequence {sequence_number} <= last seen {last_seq}"

            max_gap = self._settings.security.max_sequence_gap
            if sequence_number - last_seq > max_gap:
                return False, f"Sequence gap exceeded: {sequence_number} - {last_seq} > {max_gap}"

        self._last_seen_sequence[source_id] = sequence_number
        return True, "Sequence valid"

    def reset(self, source_id: str | None = None) -> None:
        """Reset replay protection state."""
        if source_id is not None:
            self._last_seen_sequence.pop(source_id, None)
        else:
            self._last_seen_sequence.clear()


class StaleSignalDetector:
    """Stale & Missing Signal Detector.

    Monitors update timestamps and signal changes per channel to detect frozen/stale sensors.
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._last_update_time: dict[str, datetime] = {}
        self._last_channel_value: dict[str, float] = {}

    def check_stale(self, record: RawSignalRecord) -> dict[str, str]:
        """Check raw channels for staleness or frozen values.

        Returns dict of channel_name -> stale_reason.
        """
        stale_channels: dict[str, str] = {}
        now = record.timestamp

        # Check timestamp freshness against max allowed age
        default_timeout = self._settings.security.stale_timeout_s
        per_channel = self._settings.security.per_channel_stale_timeouts_s

        # Channels to inspect on RawSignalRecord
        raw_fields = {
            "egt_cyl1_hot_uv": record.egt_cyl1_hot_uv,
            "egt_cyl2_hot_uv": record.egt_cyl2_hot_uv,
            "egt_cyl3_hot_uv": record.egt_cyl3_hot_uv,
            "egt_cyl4_hot_uv": record.egt_cyl4_hot_uv,
            "cht_hot_uv": record.cht_hot_uv,
            "oil_rtd_ohms": record.oil_rtd_ohms,
            "oil_p_counts": float(record.oil_p_counts),
            "map_counts": float(record.map_counts),
            "crank_period_us": record.crank_period_us,
            "fuel_pulse_hz": record.fuel_pulse_hz,
        }

        for ch_name, val in raw_fields.items():
            timeout = per_channel.get(ch_name, default_timeout)

            # Check timestamp gap if channel seen before
            if ch_name in self._last_update_time:
                dt = (now - self._last_update_time[ch_name]).total_seconds()
                if dt > timeout:
                    stale_channels[ch_name] = f"Signal stale: no update for {dt:.1f}s (timeout={timeout}s)"

            self._last_update_time[ch_name] = now
            self._last_channel_value[ch_name] = val

        return stale_channels

    def reset(self) -> None:
        """Reset staleness tracker state."""
        self._last_update_time.clear()
        self._last_channel_value.clear()
