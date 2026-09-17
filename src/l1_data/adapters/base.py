"""
Base Source Adapter — Abstract interface for all data sources.

Every data source (simulator, CSV, live telemetry) must implement this
interface. The contract is simple:
    1. connect() — establish connection to the data source
    2. read_next() — produce the next NormalizedSignalRecord
    3. disconnect() — clean up

All adapters must tag their output with the correct Provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.provenance import Provenance
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import NormalizedSignalRecord


class SourceAdapter(ABC):
    """Abstract base class for all data source adapters.

    Contract:
        - Every adapter produces RawSignalRecord instances
        - Every adapter tags provenance correctly
        - Every adapter is responsible for its own connection lifecycle
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source.

        Raises:
            AdapterConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def read_next(self) -> RawSignalRecord | None:
        """Read the next raw telemetry record from the source.

        Returns:
            RawSignalRecord if data is available, None if source is exhausted.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source and clean up resources."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> Provenance:
        """The provenance type for this adapter's output."""
        ...

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this adapter instance."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the adapter is currently connected."""
        return False

    async def __aenter__(self) -> SourceAdapter:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()
