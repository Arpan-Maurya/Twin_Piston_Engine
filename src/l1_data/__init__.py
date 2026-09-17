"""
L1 Data — Data Ingestion Layer.

Responsible for acquiring raw engine telemetry from any source (simulator,
CSV replay, or live hardware), validating packet integrity & security (Module 3),
persisting raw telemetry history (Module 4), and normalizing it into universal signal records.
"""

from src.l1_data.lineage import LineageTracker
from src.l1_data.raw_repository import (
    InMemoryRawTelemetryRepository,
    RawTelemetryRepositoryProtocol,
    SQLiteRawTelemetryRepository,
    create_raw_repository,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l1_data.telemetry_security import (
    PacketSigner,
    PacketVerifier,
    ReplayProtector,
    StaleSignalDetector,
)
from src.l1_data.telemetry_validator import TelemetryValidator, ValidationResult
from src.l1_data.validation import SignalValidator

__all__ = [
    "ChannelValue",
    "InMemoryRawTelemetryRepository",
    "LineageTracker",
    "NormalizedSignalRecord",
    "PacketSigner",
    "PacketVerifier",
    "RawSignalRecord",
    "RawTelemetryRepositoryProtocol",
    "ReplayProtector",
    "SQLiteRawTelemetryRepository",
    "SignalValidator",
    "StaleSignalDetector",
    "TelemetryValidator",
    "ValidationResult",
    "create_raw_repository",
]
