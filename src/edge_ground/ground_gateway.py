"""
Ground Telemetry Gateway — Module 21 Ground Station Receiver.

Ground/Server receiver component.
Responsibilities:
1. Receives canonical telemetry envelopes (EdgeTelemetryEnvelope) from Edge onboard nodes.
2. Persists raw records into historical repository (RawTelemetryRepositoryProtocol).
3. Reconstructs and executes Ground analytics pipeline (Modules 5–19):
   - Engineering state conversion (Module 5)
   - Thermodynamic & Mechanical Digital Twin (Module 6)
   - EGT, Lubrication, Vibration, Combustion Diagnostics (Modules 7–10)
   - Physics Residuals (Module 11)
   - ML Anomaly & Nine-Class Fault Classification (Modules 12–13)
   - Health Index & Degradation Supervision (Module 14)
   - RUL Estimation with Uncertainty (Module 15)
   - Mission Phase & Risk Assessment (Module 16)
   - Decision-Support Advisories & Explainability (Module 19)
4. Exposes results to REST/WebSocket API clients (Module 20).

STRICT BOUNDARY INVARIANTS:
- Receives ONLY canonical RawSignalRecord/EdgeTelemetryEnvelope.
- Reconstructs higher-level states derived purely from incoming telemetry stream.
- Zero actuation or control commands transmitted back to Edge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.exceptions import RoleViolationError
from src.core.logging import get_logger
from src.core.provenance import DeploymentRole
from src.edge_ground.envelope import EdgeTelemetryEnvelope
from src.l1_data.raw_repository import (
    InMemoryRawTelemetryRepository,
    RawTelemetryRepositoryProtocol,
)
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, PipelineStepResult

logger = get_logger(__name__)


class GroundTelemetryGateway:
    """Ground Station Telemetry Ingestion Receiver and Analytics Orchestrator."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        repository: RawTelemetryRepositoryProtocol | None = None,
        adapter: PipelineReplayAdapter | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        # Role enforcement guard
        if self._settings.deployment.role == DeploymentRole.EDGE:
            raise RoleViolationError(
                "GroundTelemetryGateway initialized on a node configured with DeploymentRole.EDGE."
            )

        self._repository = repository or InMemoryRawTelemetryRepository(max_records=100000)
        self._adapter = adapter or PipelineReplayAdapter(self._settings)

    @property
    def repository(self) -> RawTelemetryRepositoryProtocol:
        return self._repository

    @property
    def adapter(self) -> PipelineReplayAdapter:
        return self._adapter

    def process_envelope(self, envelope: EdgeTelemetryEnvelope) -> PipelineStepResult:
        """Process incoming EdgeTelemetryEnvelope on Ground station.

        Persists raw record and executes full L1->L2->L3->L4->L5 analytics pipeline.
        Returns complete PipelineStepResult containing derived physics, ML, RUL, risk, and advisories.
        """
        record = envelope.record

        # Persist raw telemetry
        self._repository.save(record)

        # Execute ground analytics pipeline
        step_results = self._adapter.process_sequence([record])
        latest_result = step_results[-1]

        logger.info(
            f"Ground processed packet seq={record.sequence_number}: "
            f"HI={latest_result.health_index:.3f}, Fault={latest_result.predicted_fault_class.name}, "
            f"RUL={latest_result.rul_hours:.1f}h"
        )
        return latest_result

    def process_batch(
        self, envelopes: list[EdgeTelemetryEnvelope]
    ) -> list[PipelineStepResult]:
        """Process a batch of incoming EdgeTelemetryEnvelopes sequentially."""
        results = []
        for env in envelopes:
            results.append(self.process_envelope(env))
        return results
