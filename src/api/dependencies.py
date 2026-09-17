"""
API Dependency Injection Providers — Shared application singletons (Module 20).

Provides FastAPI dependencies for settings, repository, validator, digital twin,
ML supervision, replay engine, what-if engine, and advisory services.
"""

from __future__ import annotations

from typing import Generator

from src.core.config import AppSettings, get_settings
from src.l1_data.raw_repository import (
    InMemoryRawTelemetryRepository,
    RawTelemetryRepositoryProtocol,
)
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import (
    PipelineReplayAdapter,
    ReplayEngine,
    ScenarioComparator,
    WhatIfEngine,
)
from src.l3_ml.advisory_explainability import (
    AdvisoryEngine,
    DiagnosticQueryService,
    ExplainabilityEngine,
)

# Shared global singletons for stateful API services
_repository_instance: RawTelemetryRepositoryProtocol | None = None
_replay_engine_instance: ReplayEngine | None = None
_what_if_engine_instance: WhatIfEngine | None = None
_pipeline_adapter_instance: PipelineReplayAdapter | None = None


def get_app_settings() -> AppSettings:
    """Dependency provider for global AppSettings."""
    return get_settings()


def get_telemetry_repository() -> RawTelemetryRepositoryProtocol:
    """Dependency provider for raw telemetry persistence repository."""
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = InMemoryRawTelemetryRepository(max_records=10000)
    return _repository_instance


def get_telemetry_validator() -> TelemetryValidator:
    """Dependency provider for TelemetryValidator pipeline service."""
    return TelemetryValidator()


def get_packet_signer() -> PacketSigner:
    """Dependency provider for PacketSigner service."""
    return PacketSigner()


def get_replay_engine() -> ReplayEngine:
    """Dependency provider for ReplayEngine singleton."""
    global _replay_engine_instance
    if _replay_engine_instance is None:
        _replay_engine_instance = ReplayEngine(scenario_id="default_scenario")
        # Load a default 10-second scenario on startup
        runner = ScenarioRunner(seed=42)
        records, ground_truths, _ = runner.run_scenario(duration_s=10.0, dt_s=1.0)
        _replay_engine_instance.load_scenario("default_scenario", records, ground_truths)
    return _replay_engine_instance


def get_what_if_engine() -> WhatIfEngine:
    """Dependency provider for WhatIfEngine singleton."""
    global _what_if_engine_instance
    if _what_if_engine_instance is None:
        _what_if_engine_instance = WhatIfEngine()
    return _what_if_engine_instance


def get_pipeline_adapter() -> PipelineReplayAdapter:
    """Dependency provider for PipelineReplayAdapter singleton."""
    global _pipeline_adapter_instance
    if _pipeline_adapter_instance is None:
        _pipeline_adapter_instance = PipelineReplayAdapter()
    return _pipeline_adapter_instance


def get_scenario_comparator() -> ScenarioComparator:
    """Dependency provider for ScenarioComparator."""
    return ScenarioComparator()


def get_explainability_engine() -> ExplainabilityEngine:
    """Dependency provider for ExplainabilityEngine."""
    return ExplainabilityEngine()


def get_advisory_engine() -> AdvisoryEngine:
    """Dependency provider for AdvisoryEngine."""
    return AdvisoryEngine()


def get_diagnostic_query_service() -> DiagnosticQueryService:
    """Dependency provider for DiagnosticQueryService."""
    return DiagnosticQueryService()
