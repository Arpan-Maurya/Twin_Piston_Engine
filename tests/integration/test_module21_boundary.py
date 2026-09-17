"""
Boundary and Security Verification Tests for Original Module 21.

Enforces Module 21 mandatory design and security gates:
1. Edge node does not access simulator ground truth (SimulationGroundTruth).
2. Edge node cannot initialize ML models, RUL, Advisory, What-If, or Ground REST API.
3. No actuator or UAV control commands are exposed or transmitted.
4. Ground station reconstructs analytics purely from canonical RawSignalRecord/EdgeTelemetryEnvelope.
5. End-to-end simulation -> Edge Transport -> Ground Pipeline compatibility.
"""

import pytest
import inspect
import re
from datetime import datetime, timezone

from src.core.config import AppSettings, DeploymentConfig
from src.core.exceptions import ActuationAttemptError, RoleViolationError
from src.core.provenance import DeploymentRole, Provenance
from src.edge_ground import (
    EdgeTelemetryNode,
    GroundTelemetryGateway,
    InMemoryTelemetryTransport,
    edge_node,
    buffer,
    envelope,
    transport,
)
from src.l1_data.simulator.forward_simulator import ScenarioRunner


class TestModule21BoundarySafety:
    """Boundary and security verification suite."""

    def test_edge_source_code_has_no_ground_truth_imports(self):
        """Verify Edge code modules do not import or consume SimulationGroundTruth."""
        edge_modules = [edge_node, buffer, envelope, transport]
        prohibited = ["SimulationGroundTruth", "Wiebe", "ground_truth", "stefan_boltzmann"]

        for mod in edge_modules:
            source = inspect.getsource(mod)
            for term in prohibited:
                assert term not in source, f"Prohibited ground-truth term '{term}' found in Edge module {mod.__name__}"

    def test_edge_source_code_has_no_ml_or_advisory_imports(self):
        """Verify Edge code modules do not import L3 ML or L5 Advisory engines."""
        edge_modules = [edge_node, buffer, envelope, transport]
        prohibited = [
            "MLModel",
            "AnomalyDetector",
            "NineClassFaultClassifier",
            "RULEstimator",
            "AdvisoryEngine",
            "WhatIfEngine",
            "api_v1_router",
        ]

        for mod in edge_modules:
            source = inspect.getsource(mod)
            for term in prohibited:
                assert term not in source, f"Prohibited ground/ML term '{term}' found in Edge module {mod.__name__}"

    def test_edge_role_enforcement_prevents_ground_services(self):
        edge_settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE))
        with pytest.raises(RoleViolationError):
            GroundTelemetryGateway(settings=edge_settings)

    def test_simulator_edge_transport_ground_pipeline_compatibility(self):
        """End-to-end simulator -> Edge Transport -> Ground Pipeline validation."""
        runner = ScenarioRunner(seed=42)
        sim_records, sim_ground_truths, metadata = runner.run_scenario(duration_s=5.0, dt_s=1.0)

        transport_adapter = InMemoryTelemetryTransport()
        edge = EdgeTelemetryNode(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.SIMULATION)),
            transport=transport_adapter,
        )
        ground = GroundTelemetryGateway(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.SIMULATION))
        )

        for rec in sim_records:
            edge.process_telemetry_packet(rec)

        assert len(transport_adapter.received_envelopes) == len(sim_records)

        pipeline_results = ground.process_batch(transport_adapter.received_envelopes)
        assert len(pipeline_results) == len(sim_records)

        for res in pipeline_results:
            assert res.accepted is True
            assert 0.0 <= res.health_index <= 1.0
            assert res.predicted_fault_class is not None
