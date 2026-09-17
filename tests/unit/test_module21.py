"""
Unit tests for Original Module 21: Edge/Ground-Station Partition.
"""

import pytest
from datetime import datetime, timezone

from src.core.config import AppSettings, DeploymentConfig
from src.core.exceptions import RoleViolationError
from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy, Provenance
from src.edge_ground import (
    EdgeStoreAndForwardBuffer,
    EdgeTelemetryEnvelope,
    EdgeTelemetryNode,
    GroundTelemetryGateway,
    HTTPTelemetryTransport,
    InMemoryTelemetryTransport,
    check_role_permission,
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


class TestEdgeTelemetryEnvelope:

    def test_envelope_creation(self, sample_raw_record):
        envelope = EdgeTelemetryEnvelope.from_record(sample_raw_record, signature="hmac_test_sig")
        assert envelope.sequence_number == 1
        assert envelope.signature == "hmac_test_sig"
        assert envelope.schema_version == "1.0.0"
        assert envelope.node_id == "edge_node_01"


class TestEdgeStoreAndForwardBuffer:

    def test_buffer_enqueue_dequeue(self, sample_raw_record):
        buf = EdgeStoreAndForwardBuffer(max_capacity=5)
        env1 = EdgeTelemetryEnvelope.from_record(sample_raw_record)
        assert buf.enqueue(env1) is True
        assert len(buf) == 1

        dequeued = buf.dequeue()
        assert dequeued is not None
        assert dequeued.sequence_number == 1
        assert len(buf) == 0

    def test_buffer_discard_oldest_policy(self, sample_raw_record):
        buf = EdgeStoreAndForwardBuffer(max_capacity=2, overflow_policy=OverflowPolicy.DISCARD_OLDEST)
        record1 = sample_raw_record.model_copy(update={"sequence_number": 1})
        record2 = sample_raw_record.model_copy(update={"sequence_number": 2})
        record3 = sample_raw_record.model_copy(update={"sequence_number": 3})

        buf.enqueue(EdgeTelemetryEnvelope.from_record(record1))
        buf.enqueue(EdgeTelemetryEnvelope.from_record(record2))
        assert len(buf) == 2

        # Overflow triggers discard of seq 1
        assert buf.enqueue(EdgeTelemetryEnvelope.from_record(record3)) is True
        assert len(buf) == 2

        first = buf.dequeue()
        assert first.sequence_number == 2
        second = buf.dequeue()
        assert second.sequence_number == 3

        meta = buf.get_freshness_metadata()
        assert meta["dropped_packet_count"] == 1

    def test_buffer_discard_newest_policy(self, sample_raw_record):
        buf = EdgeStoreAndForwardBuffer(max_capacity=2, overflow_policy=OverflowPolicy.DISCARD_NEWEST)
        record1 = sample_raw_record.model_copy(update={"sequence_number": 1})
        record2 = sample_raw_record.model_copy(update={"sequence_number": 2})
        record3 = sample_raw_record.model_copy(update={"sequence_number": 3})

        buf.enqueue(EdgeTelemetryEnvelope.from_record(record1))
        buf.enqueue(EdgeTelemetryEnvelope.from_record(record2))

        # Overflow triggers discard of seq 3
        assert buf.enqueue(EdgeTelemetryEnvelope.from_record(record3)) is False
        assert len(buf) == 2

        first = buf.dequeue()
        assert first.sequence_number == 1


class TestRoleEnforcement:

    def test_check_role_permission_simulation_allows_all(self):
        settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.SIMULATION))
        # Should not raise exception
        check_role_permission([DeploymentRole.EDGE], "test_component", settings=settings)
        check_role_permission([DeploymentRole.GROUND], "test_component", settings=settings)

    def test_check_role_permission_edge_restricts_ground(self):
        settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE))
        with pytest.raises(RoleViolationError):
            check_role_permission([DeploymentRole.GROUND], "ground_ml_service", settings=settings)

    def test_edge_node_rejects_ground_config(self):
        settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.GROUND))
        with pytest.raises(RoleViolationError):
            EdgeTelemetryNode(settings=settings)

    def test_ground_gateway_rejects_edge_config(self):
        settings = AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE))
        with pytest.raises(RoleViolationError):
            GroundTelemetryGateway(settings=settings)
