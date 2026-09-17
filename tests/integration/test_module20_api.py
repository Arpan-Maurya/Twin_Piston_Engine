"""
Integration tests for Original Module 20 REST API Endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app

@pytest.fixture(scope="module")
def api_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestSystemEndpoints:

    def test_get_system_health(self, api_client):
        response = api_client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service_name" in data

    def test_get_system_status(self, api_client):
        response = api_client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["service_status"] == "OPERATIONAL"
        assert data["telemetry_pipeline_status"] == "READY"


class TestTelemetryEndpoints:

    def test_ingest_telemetry_valid(self, api_client):
        payload = {
            "sequence_number": 1,
            "source_type": "SIMULATED",
            "egt_cyl1_hot_uv": 12000.0,
            "egt_cyl2_hot_uv": 12100.0,
            "egt_cyl3_hot_uv": 11950.0,
            "egt_cyl4_hot_uv": 12050.0,
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
            "ambient_press_pa": 101325.0
        }
        response = api_client.post("/api/v1/telemetry", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "accepted" in data

    def test_get_latest_telemetry(self, api_client):
        response = api_client.get("/api/v1/telemetry/latest")
        assert response.status_code in (200, 404)

    def test_get_telemetry_history(self, api_client):
        response = api_client.get("/api/v1/telemetry/history?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["page"] == 1


class TestEngineHealthEndpoints:

    def test_get_engine_health(self, api_client):
        response = api_client.get("/api/v1/engine/health")
        assert response.status_code == 200
        data = response.json()
        assert "health_index" in data
        assert 0.0 <= data["health_index"] <= 1.0
        assert "degradation_state" in data


class TestDiagnosticsEndpoints:

    def test_get_diagnostics_summary(self, api_client):
        response = api_client.get("/api/v1/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert "rpm" in data
        assert "egt_mean_k" in data

    def test_get_egt_diagnostics(self, api_client):
        response = api_client.get("/api/v1/diagnostics/egt")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_get_lubrication_diagnostics(self, api_client):
        response = api_client.get("/api/v1/diagnostics/lubrication")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_get_vibration_diagnostics(self, api_client):
        response = api_client.get("/api/v1/diagnostics/vibration")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_get_combustion_diagnostics(self, api_client):
        response = api_client.get("/api/v1/diagnostics/combustion")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list))


class TestFaultAnomalyEndpoints:

    def test_get_anomaly_status(self, api_client):
        response = api_client.get("/api/v1/diagnostics/anomaly")
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data

    def test_get_fault_classification(self, api_client):
        response = api_client.get("/api/v1/diagnostics/fault")
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "class_id" in data


class TestRULEndpoints:

    def test_get_rul(self, api_client):
        response = api_client.get("/api/v1/engine/rul")
        assert response.status_code == 200
        data = response.json()
        assert "hours_remaining" in data
        assert data["unit"] == "hours"


class TestMissionRiskEndpoints:

    def test_get_mission_risk(self, api_client):
        response = api_client.get("/api/v1/mission/risk")
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "flight_phase" in data


class TestAdvisoriesEndpoints:

    def test_get_advisories(self, api_client):
        response = api_client.get("/api/v1/advisories")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_advisory_by_id(self, api_client):
        response = api_client.get("/api/v1/advisories/adv_001")
        assert response.status_code in (200, 404)

    def test_get_explanations(self, api_client):
        response = api_client.get("/api/v1/explanations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "explanation_id" in data

    def test_diagnostic_query(self, api_client):
        payload = {"question_type": "HEALTH_STATUS"}
        response = api_client.post("/api/v1/diagnostic/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["question_type"] == "HEALTH_STATUS"


class TestHistoryEndpoints:

    def test_get_health_history(self, api_client):
        response = api_client.get("/api/v1/history/health?page=1&page_size=5")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_get_diagnostics_history(self, api_client):
        response = api_client.get("/api/v1/history/diagnostics?page=1&page_size=5")
        assert response.status_code == 200

    def test_get_advisories_history(self, api_client):
        response = api_client.get("/api/v1/history/advisories?page=1&page_size=5")
        assert response.status_code == 200


class TestReplayEndpoints:

    def test_get_replay_state(self, api_client):
        response = api_client.get("/api/v1/replay/state")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_replay_start(self, api_client):
        payload = {"scenario_id": "test_scenario_01"}
        response = api_client.post("/api/v1/replay/start", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] in ("PLAYING", "STOPPED", "running", "stopped")

    def test_replay_pause_resume_stop(self, api_client):
        res_pause = api_client.post("/api/v1/replay/pause")
        assert res_pause.status_code == 200
        res_resume = api_client.post("/api/v1/replay/resume")
        assert res_resume.status_code == 200
        res_stop = api_client.post("/api/v1/replay/stop")
        assert res_stop.status_code == 200


class TestWhatIfEndpoints:

    def test_execute_what_if(self, api_client):
        payload = {
            "baseline_scenario_id": "baseline_01",
            "modifications": {},
            "duration_s": 10.0,
            "dt_s": 1.0
        }
        response = api_client.post("/api/v1/scenarios/what-if", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "what_if_id" in data

    def test_compare_scenarios(self, api_client):
        payload = {
            "baseline_scenario_id": "baseline_01",
            "what_if_id": "what_if_01"
        }
        response = api_client.post("/api/v1/scenarios/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "metric_deltas" in data
