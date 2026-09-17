"""
Integration tests for Offline Operation, Store-and-Forward Buffering, and Recovery (Module 21).
"""

import pytest
from datetime import datetime, timezone

from src.core.config import AppSettings, DeploymentConfig
from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy, Provenance
from src.edge_ground import (
    EdgeTelemetryNode,
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


class TestFailureRecovery:

    def test_offline_buffering_when_link_disconnected(self, sample_raw_record):
        transport = InMemoryTelemetryTransport()
        edge_node = EdgeTelemetryNode(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE)),
            transport=transport,
            buffer_capacity=100,
        )

        # Disconnect link
        transport.disconnect()
        assert edge_node.get_communication_state() == CommunicationState.DISCONNECTED

        # Process 5 records offline
        for i in range(1, 6):
            rec = sample_raw_record.model_copy(update={"sequence_number": i})
            edge_node.process_telemetry_packet(rec)

        # Envelopes should accumulate in buffer, none sent over transport
        assert len(transport.received_envelopes) == 0
        assert len(edge_node.buffer) == 5
        assert edge_node.get_communication_state() == CommunicationState.BUFFERING

        meta = edge_node.get_freshness_metadata()
        assert meta["queue_depth"] == 5
        assert meta["communication_state"] == "BUFFERING"

    def test_reconnect_flush_preserves_fifo_sequence(self, sample_raw_record):
        transport = InMemoryTelemetryTransport()
        edge_node = EdgeTelemetryNode(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE)),
            transport=transport,
            buffer_capacity=100,
        )

        # Disconnect link and buffer records
        transport.disconnect()
        for i in range(10, 15):
            rec = sample_raw_record.model_copy(update={"sequence_number": i})
            edge_node.process_telemetry_packet(rec)

        assert len(edge_node.buffer) == 5
        assert len(transport.received_envelopes) == 0

        # Reconnect link
        transport.connect()
        success, failed = edge_node.flush_buffer()

        assert success == 5
        assert failed == 0
        assert len(edge_node.buffer) == 0
        assert len(transport.received_envelopes) == 5

        # Verify FIFO sequence order
        received_seqs = [env.sequence_number for env in transport.received_envelopes]
        assert received_seqs == [10, 11, 12, 13, 14]

    def test_bounded_capacity_drop_tracking(self, sample_raw_record):
        transport = InMemoryTelemetryTransport()
        transport.disconnect()

        edge_node = EdgeTelemetryNode(
            settings=AppSettings(deployment=DeploymentConfig(role=DeploymentRole.EDGE)),
            transport=transport,
            buffer_capacity=3,
            overflow_policy=OverflowPolicy.DISCARD_OLDEST,
        )

        # Process 5 records into capacity-3 buffer
        for i in range(1, 6):
            rec = sample_raw_record.model_copy(update={"sequence_number": i})
            edge_node.process_telemetry_packet(rec)

        assert len(edge_node.buffer) == 3
        meta = edge_node.get_freshness_metadata()
        assert meta["dropped_packet_count"] == 2

        # Reconnect and verify remaining sequence numbers are 3, 4, 5
        transport.connect()
        edge_node.flush_buffer()
        received_seqs = [env.sequence_number for env in transport.received_envelopes]
        assert received_seqs == [3, 4, 5]
