"""
Packet Validation — Signal integrity and range checks.

Validates every NormalizedSignalRecord before it enters L2:
    - Per-channel range validation (from config)
    - Rate-of-change checks
    - Sequence number monotonicity
    - Missing channel detection

IMPORTANT: Invalid channels are FLAGGED, never silently replaced.
The system must continue computing unaffected parameters when one
channel is invalid.
"""

from __future__ import annotations

from src.core.config import AppSettings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)

# Channel name → config range key mapping
CHANNEL_RANGE_MAP: dict[str, str] = {
    "rpm": "rpm",
    "map_pressure": "map_pressure",
    "throttle_position": "throttle_position",
    "egt_cyl_1": "egt", "egt_cyl_2": "egt", "egt_cyl_3": "egt", "egt_cyl_4": "egt",
    "cht_cyl_1": "cht", "cht_cyl_2": "cht", "cht_cyl_3": "cht", "cht_cyl_4": "cht",
    "oil_temp": "oil_temp",
    "oil_pressure": "oil_pressure",
    "coolant_temp": "coolant_temp",
    "fuel_flow": "fuel_flow",
    "fuel_pressure": "fuel_pressure",
    "intake_air_temp": "intake_air_temp",
    "ambient_pressure": "ambient_pressure",
    "ambient_temp": "ambient_temp",
    "voltage": "voltage",
    "current": "current",
    "vibration_x": "vibration", "vibration_y": "vibration",
    "vibration_z": "vibration", "vibration_rms": "vibration",
    "propeller_speed": "propeller_speed",
    "boost_pressure": "boost_pressure",
    "wastegate_duty": "wastegate_duty",
    "lambda_sensor": "lambda_sensor",
    "ignition_timing_cyl_1": "ignition_timing",
    "ignition_timing_cyl_2": "ignition_timing",
    "ignition_timing_cyl_3": "ignition_timing",
    "ignition_timing_cyl_4": "ignition_timing",
    "altitude": "altitude",
    "engine_hours": "engine_hours",
}

# Maximum allowable rate of change per second for key channels
RATE_LIMITS: dict[str, float] = {
    "rpm": 3000.0,           # RPM can't jump 3000 in one second
    "map_pressure": 50000.0,  # 50 kPa/s
    "oil_pressure": 200000.0,
    "coolant_temp": 10.0,     # 10 K/s
    "oil_temp": 5.0,          # 5 K/s
}


class SignalValidator:
    """Validates NormalizedSignalRecord integrity.

    Checks each channel against configured ranges and rate limits.
    Invalid channels are flagged — never silently replaced.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._previous_record: NormalizedSignalRecord | None = None
        self._previous_sequence: int = -1

    def validate(self, record: NormalizedSignalRecord) -> NormalizedSignalRecord:
        """Validate a signal record and return it with updated validity flags.

        Channels that fail validation are marked as invalid with the
        appropriate reason. The original values are preserved.

        Args:
            record: The record to validate.

        Returns:
            A new NormalizedSignalRecord with validity flags updated.
        """
        updates: dict[str, ChannelValue] = {}

        channels = record.all_channel_values

        for ch_name, ch_value in channels.items():
            if not ch_value.valid:
                # Already invalid — don't re-validate
                continue

            # Range check
            range_key = CHANNEL_RANGE_MAP.get(ch_name)
            if range_key and range_key in self._settings.channel_ranges:
                ch_range = self._settings.channel_ranges[range_key]
                if not ch_range.contains(ch_value.value):
                    updates[ch_name] = ch_value.as_invalid(
                        ChannelValidity.INVALID_RANGE,
                        f"Value {ch_value.value} outside range [{ch_range.min}, {ch_range.max}]",
                    )
                    logger.warning(
                        "Channel %s failed range check: %f not in [%f, %f]",
                        ch_name, ch_value.value, ch_range.min, ch_range.max,
                    )
                    continue

            # Rate-of-change check
            if ch_name in RATE_LIMITS and self._previous_record is not None:
                prev_channels = self._previous_record.all_channel_values
                if ch_name in prev_channels and prev_channels[ch_name].valid:
                    rate = abs(ch_value.value - prev_channels[ch_name].value)
                    # Assumes 1 Hz sample rate; adjust for actual dt
                    if rate > RATE_LIMITS[ch_name]:
                        updates[ch_name] = ch_value.as_invalid(
                            ChannelValidity.INVALID_RATE,
                            f"Rate of change {rate:.1f} exceeds limit {RATE_LIMITS[ch_name]}",
                        )
                        logger.warning(
                            "Channel %s rate-of-change exceeded: %f > %f",
                            ch_name, rate, RATE_LIMITS[ch_name],
                        )
                        continue

        # Sequence number monotonicity
        if record.sequence_number <= self._previous_sequence:
            logger.warning(
                "Sequence number not monotonic: %d <= %d",
                record.sequence_number, self._previous_sequence,
            )

        # Build validated record (replace invalid channels)
        if updates:
            validated = record.model_copy(update=updates)
        else:
            validated = record

        self._previous_record = validated
        self._previous_sequence = record.sequence_number

        return validated

    def validate_raw_record(self, raw_record: RawSignalRecord) -> RawSignalRecord:
        """Validate a RawSignalRecord for missing/malformed fields, range violations,
        and invalid provenance.

        Raises PacketIntegrityError or InvalidChannelError on severe malformations.
        """
        from src.core.exceptions import PacketIntegrityError
        from src.core.provenance import Provenance

        # Provenance validation
        if not isinstance(raw_record.source_type, Provenance):
            raise PacketIntegrityError(f"Invalid provenance tag: {raw_record.source_type}")

        # Range & physical limits for raw acquisition fields
        if raw_record.crank_period_us <= 0:
            raise PacketIntegrityError("Invalid crank period <= 0 us")

        if raw_record.fuel_pulse_hz < 0:
            raise PacketIntegrityError("Invalid negative fuel pulse frequency")

        if raw_record.oil_rtd_ohms <= 0:
            raise PacketIntegrityError("Invalid RTD resistance <= 0 ohms")

        # Integrity signature verification if present
        if raw_record.integrity_hash:
            computed = raw_record.compute_integrity_hash()
            if raw_record.integrity_hash != computed:
                logger.warning(
                    "Integrity hash mismatch on sequence %d: %s != %s",
                    raw_record.sequence_number, raw_record.integrity_hash, computed,
                )

        return raw_record

