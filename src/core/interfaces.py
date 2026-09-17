"""
Layer Interfaces — Clean Protocol abstractions between layers.

Defines the exact operational boundaries:
    L1 Data → L2 Digital Twin → L3 ML/Supervision → L4 Advisory → L5 Interface
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.provenance import Provenance
from src.core.schemas import (
    Alert,
    DerivedEngineState,
    DiagnosticState,
    HealthState,
    ResidualState,
    RULState,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import NormalizedSignalRecord


@runtime_checkable
class L1SourceAdapterProtocol(Protocol):
    """Protocol for L1 data ingestion source adapters."""

    async def connect(self) -> None:
        """Establish connection to data source."""
        ...

    async def read_next(self) -> NormalizedSignalRecord | RawSignalRecord | None:
        """Read the next telemetry frame from the source."""
        ...

    async def disconnect(self) -> None:
        """Disconnect and release adapter resources."""
        ...

    @property
    def source_type(self) -> Provenance:
        """Return the overall provenance type of this source."""
        ...


@runtime_checkable
class L2DigitalTwinProtocol(Protocol):
    """Protocol for L2 Physics Digital Twin processing engine."""

    def process_record(
        self, record: NormalizedSignalRecord
    ) -> tuple[DerivedEngineState, DiagnosticState, ResidualState]:
        """Transform normalized raw telemetry into derived engineering parameters,
        subsystem diagnostics, and baseline residuals.
        """
        ...


@runtime_checkable
class L3SupervisionProtocol(Protocol):
    """Protocol for L3 ML supervision and health assessment engine."""

    def evaluate_health(
        self, derived: DerivedEngineState, residuals: ResidualState
    ) -> tuple[HealthState, RULState | None]:
        """Evaluate engine health, detect anomalies, classify faults, and estimate RUL."""
        ...


@runtime_checkable
class L4AdvisoryProtocol(Protocol):
    """Protocol for L4 advisory and maintenance recommendation engine."""

    def generate_advisory(
        self, health: HealthState, diagnostics: DiagnosticState
    ) -> Alert:
        """Generate human-readable physics narrative explanations and advisories."""
        ...


@runtime_checkable
class L5InterfaceProtocol(Protocol):
    """Protocol for L5 interface streaming and presentation."""

    async def broadcast_update(
        self,
        raw: NormalizedSignalRecord,
        derived: DerivedEngineState,
        health: HealthState,
        advisory: Alert,
    ) -> None:
        """Broadcast live twin updates to connected clients (REST/WebSocket)."""
        ...
