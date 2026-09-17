"""
Integration tests for Deployment Partition and Component Ownership (Module 21).
"""

import pytest
from datetime import datetime, timezone

from src.core.config import AppSettings, DeploymentConfig
from src.core.exceptions import RoleViolationError
from src.core.provenance import DeploymentRole, Provenance
from src.edge_ground import (
    EdgeTelemetryNode,
    GroundTelemetryGateway,
    InMemoryTelemetryTransport,
)
from src.l1_data.raw_signal_record import RawSignalRecord


@pytest.fixture
def sample_raw_record():
    return RawSignalRecord(
        timestamp=datetime.now(timezone.utc),
        sequence_number=1,
        source_type=Provenance.SIMULATED,
        egt_cyl1_hot_uv=12000.0,
        egt_cyl2_hot_uv=12100.0,
        egt_cyl3_hot_uv=11950.0,
        egt_cyl4_hot_uv=12050.0,
        egt_cold_c=25.0,
        cht_hot_uv=8000.0,
        cht_cold_c=25.0,
        oil_rtd_ohms=110.0,
        oil_p_counts=2048,
        map_counts=2048,
        adc_vref_counts=4095,
        crank_period_us=12000.0,
        fuel_pulse_hz=50.0,
        accel_counts_xyz=(2048, 2048, 2048),
        ambient_temp_c=25.0,
        ambient_press_pa=101325.0,
    )


class TestDeploymentPartition:

    def test_edge_node_runs_l1_validation_and_buffering(self, sample_raw_record):
        edge_settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE))
        transport = InMemoryTelemetryTransport()
        edge_node = EdgeTelemetryNode(settings=edge_settings, transport=transport)

        res = edge_node.process_telemetry_packet(sample_raw_record)
        assert res.accepted is True
        assert len(transport.received_envelopes) == 1

    def test_ground_gateway_ingests_and_executes_pipeline(self, sample_raw_record):
        ground_settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.GROUND))
        gateway = GroundTelemetryGateway(settings=ground_settings)

        transport = InMemoryTelemetryTransport()
        edge_settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE))
        edge_node = EdgeTelemetryNode(settings=edge_settings, transport=transport)

        edge_node.process_telemetry_packet(sample_raw_record)
        env = transport.received_envelopes[0]

        step_result = gateway.process_envelope(env)
        assert step_result.accepted is True
        assert 0.0 <= step_result.health_index <= 1.0
        assert step_result.predicted_fault_class.name is not None
        assert step_result.rul_hours >= 0.0

    def test_end_to_end_partition_flow(self, sample_raw_record):
        transport = InMemoryTelemetryTransport()
        edge_node = EdgeTelemetryNode(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE)),
            transport=transport,
        )
        gateway = GroundTelemetryGateway(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.GROUND))
        )

        for i in range(1, 4):
            rec = sample_raw_record.model_copy(update={"sequence_number": i})
            edge_node.process_telemetry_packet(rec)

        envelopes = transport.received_envelopes
        assert len(envelopes) == 3

        results = gateway.process_batch(envelopes)
        assert len(results) == 3
        assert gateway.repository.count() == 3
