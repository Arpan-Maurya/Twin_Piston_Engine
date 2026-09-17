"""
Module 21 — Edge/Ground-Station Partition Package.

Defines deployment role boundaries, transport abstractions, store-and-forward buffering,
Edge onboarding node, and Ground station ingestion gateway.
"""

from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy
from src.edge_ground.buffer import EdgeStoreAndForwardBuffer
from src.edge_ground.edge_node import EdgeTelemetryNode
from src.edge_ground.envelope import EdgeTelemetryEnvelope
from src.edge_ground.ground_gateway import GroundTelemetryGateway
from src.edge_ground.role_enforcement import check_role_permission, require_deployment_role
from src.edge_ground.transport import (
    HTTPTelemetryTransport,
    InMemoryTelemetryTransport,
    ITelemetryTransport,
)

__all__ = [
    "DeploymentRole",
    "CommunicationState",
    "OverflowPolicy",
    "EdgeTelemetryEnvelope",
    "EdgeStoreAndForwardBuffer",
    "ITelemetryTransport",
    "InMemoryTelemetryTransport",
    "HTTPTelemetryTransport",
    "EdgeTelemetryNode",
    "GroundTelemetryGateway",
    "check_role_permission",
    "require_deployment_role",
]
