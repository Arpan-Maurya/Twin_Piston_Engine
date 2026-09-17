"""
Integration tests for Communication Transport Layer (Module 21).
"""

import pytest
from datetime import datetime, timezone

from src.core.provenance import CommunicationState, Provenance
from src.edge_ground import (
    EdgeTelemetryEnvelope,
    InMemoryTelemetryTransport,
)
from src.l1_data.raw_signal_record import RawSignalRecord


@pytest.fixture
def sample_envelope():
    record = RawSignalRecord(
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
    return EdgeTelemetryEnvelope.from_record(record)


class TestTransportLayer:

    def test_in_memory_transport_send_single(self, sample_envelope):
        transport = InMemoryTelemetryTransport()
        assert transport.get_status() == CommunicationState.CONNECTED
        assert transport.send_telemetry(sample_envelope) is True
        assert len(transport.received_envelopes) == 1

    def test_in_memory_transport_disconnect(self, sample_envelope):
        transport = InMemoryTelemetryTransport()
        transport.disconnect()
        assert transport.get_status() == CommunicationState.DISCONNECTED
        assert transport.send_telemetry(sample_envelope) is False
        assert len(transport.received_envelopes) == 0

    def test_in_memory_transport_simulated_network_failure(self, sample_envelope):
        transport = InMemoryTelemetryTransport(simulate_network_failure=True)
        assert transport.get_status() == CommunicationState.CONNECTED  # initially connected state object
        assert transport.connect() is False
        assert transport.get_status() == CommunicationState.DISCONNECTED
        assert transport.send_telemetry(sample_envelope) is False

        # Restore connection
        transport.set_simulate_failure(False)
        assert transport.connect() is True
        assert transport.send_telemetry(sample_envelope) is True
        assert len(transport.received_envelopes) == 1

    def test_in_memory_transport_batch_send(self, sample_envelope):
        transport = InMemoryTelemetryTransport()
        batch = [sample_envelope.model_copy(update={"sequence_number": i}) for i in range(1, 6)]
        success, failed = transport.send_batch(batch)
        assert success == 5
        assert failed == 0
        assert len(transport.received_envelopes) == 5
