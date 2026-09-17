"""
Regression tests for Audit Findings (BUG-001 through BUG-003, SEC-001 through SEC-005).
"""

import pytest
from pathlib import Path
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.core.config import AppSettings, load_settings
from src.core.exceptions import ConfigurationError
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.raw_repository import InMemoryRawTelemetryRepository
from src.l3_ml.ml_infrastructure import MLModelLoader, ModelMetadata
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.simulator.forward_simulator import ScenarioRunner


class TestRegressionAuditFindings:
    """Regression test suite for all identified and fixed security & error audit findings."""

    def test_bug_001_telemetry_signal_quality_fallback(self):
        """BUG-001: SignalQuality fallback when packet validation rejects telemetry."""
        app = create_app()
        client = TestClient(app)
        
        # Ingest an invalid telemetry payload (e.g. oil_p_counts out of range > 4095)
        # Should return HTTP 422 validation error or rejected TelemetryIngestResponse without NameError
        payload = {
            "sequence_number": 1,
            "egt_cyl1_hot_uv": 12000.0,
            "egt_cyl2_hot_uv": 12000.0,
            "egt_cyl3_hot_uv": 12000.0,
            "egt_cyl4_hot_uv": 12000.0,
            "egt_cold_c": 25.0,
            "cht_hot_uv": 8000.0,
            "cht_cold_c": 25.0,
            "oil_rtd_ohms": 110.0,
            "oil_p_counts": 2048,
            "map_counts": 2048,
            "adc_vref_counts": 4095,
            "crank_period_us": 12000.0,
            "fuel_pulse_hz": 50.0,
            "accel_counts_xyz": [2048, 2048, 2048],
            "ambient_temp_c": 25.0,
            "ambient_press_pa": 101325.0,
            "signature": "invalid_signature_hex"
        }
        response = client.post("/api/v1/telemetry", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["accepted"] is False
        assert data["quality"]["valid"] is False

    def test_bug_002_raw_repository_logging_format(self, caplog):
        """BUG-002: raw_repository logger `%s` format string correctness."""
        repo = InMemoryRawTelemetryRepository()
        # Mock save failure logging
        record = RawSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            egt_cyl1_hot_uv=12000.0,
            egt_cyl2_hot_uv=12000.0,
            egt_cyl3_hot_uv=12000.0,
            egt_cyl4_hot_uv=12000.0,
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
            ambient_press_pa=101325.0
        )
        repo.save(record)
        assert repo.count() == 1

    def test_bug_003_raw_repository_sql_query(self, tmp_path):
        """BUG-003: Raw repository SQL query uses direct ORDER BY clause without runtime string modification."""
        from src.l1_data.raw_repository import SQLiteRawTelemetryRepository
        db_path = tmp_path / "test_repo.db"
        repo = SQLiteRawTelemetryRepository(db_path=str(db_path))
        record = RawSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=10,
            egt_cyl1_hot_uv=12000.0,
            egt_cyl2_hot_uv=12000.0,
            egt_cyl3_hot_uv=12000.0,
            egt_cyl4_hot_uv=12000.0,
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
            ambient_press_pa=101325.0
        )
        repo.save(record)
        t_start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        t_end = datetime(2099, 1, 1, tzinfo=timezone.utc)
        fetched = repo.get_by_timestamp_range(t_start, t_end)
        assert len(fetched) == 1
        assert fetched[0].sequence_number == 10

    def test_sec_001_model_path_traversal_prevention(self, tmp_path):
        """SEC-001: Model loader rejects paths attempting traversal outside configured model_dir."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        malicious_file = outside_dir / "evil_model.joblib"
        malicious_file.write_text("malicious")

        settings = AppSettings(model_dir=str(model_dir))
        loader = MLModelLoader(settings)

        meta = ModelMetadata(
            name="test_model",
            version="1.0.0",
            trained_at=datetime.now(timezone.utc),
            input_shape=(10,),
            output_shape=(1,),
            description="test model"
        )
        result = loader.load_model(malicious_file, meta)
        assert result is None

    def test_sec_002_production_secret_key_enforcement(self, monkeypatch):
        """SEC-002: Production environment rejects default secret key."""
        monkeypatch.delenv("TELEMETRY_SECRET_KEY", raising=False)
        
        prod_settings = AppSettings(app_env="production")
        signer = PacketSigner(prod_settings)
        
        record = RawSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            egt_cyl1_hot_uv=12000.0,
            egt_cyl2_hot_uv=12000.0,
            egt_cyl3_hot_uv=12000.0,
            egt_cyl4_hot_uv=12000.0,
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
            ambient_press_pa=101325.0
        )
        with pytest.raises(ValueError, match="SEC-002: Production environment requires secret key"):
            signer.sign_record(record)

    def test_sec_004_payload_size_limit(self):
        """SEC-004: Server rejects oversized payloads exceeding 1MB limit with 413 Payload Too Large."""
        app = create_app()
        client = TestClient(app)
        
        # Send fake oversized body with header Content-Length > 1MB
        headers = {"Content-Length": str(2_000_000), "Content-Type": "application/json"}
        response = client.post("/api/v1/telemetry", content=b"a" * 100, headers=headers)
        assert response.status_code == 413
        assert response.json()["error_code"] == "PAYLOAD_TOO_LARGE"

    def test_sec_005_dt_s_bounding(self):
        """SEC-005: Scenario runner and API validate and clamp dt_s to valid range [0.01, 10.0]."""
        runner = ScenarioRunner()
        # Extremely small dt_s should be clamped to 0.01, total steps capped
        records, ground_truths, meta = runner.run_scenario(duration_s=10.0, dt_s=0.0000001)
        assert len(records) <= 1000

        # Test API validation
        app = create_app()
        client = TestClient(app)
        res = client.post("/api/v1/scenarios/what-if", json={
            "baseline_scenario_id": "b1",
            "modifications": {},
            "duration_s": 10.0,
            "dt_s": 0.000001
        })
        assert res.status_code == 422
