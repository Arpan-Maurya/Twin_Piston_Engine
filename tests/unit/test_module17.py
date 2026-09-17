"""
Unit tests for Original Module 17:
Physics Forward Model, Fault Injection and Scenario Simulator.
"""

from datetime import datetime, timezone
import pytest
import numpy as np

from src.core.provenance import FaultClass, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
    ScenarioRunner,
    SimulationGroundTruth,
)


class TestForwardPhysicsModel:
    """Tests for the forward physical model of Rotax 915 iS."""

    def test_nominal_physics(self):
        physics = ForwardPhysicsModel()
        gt = physics.step(time_s=1.0, rpm=5000.0, throttle_pct=80.0, altitude_m=500.0)

        assert isinstance(gt, SimulationGroundTruth)
        assert gt.rpm == 5000.0
        assert gt.throttle_pct == 80.0
        assert 50.0 < gt.power_kw <= 110.0
        assert 100.0 < gt.torque_nm < 250.0
        assert 100.0 < gt.map_pressure_pa < 250000.0
        assert len(gt.egt_k) == 4
        assert gt.oil_temp_k > 280.0
        assert gt.oil_pressure_pa > 100000.0

    def test_fault_injection_misfire(self):
        faults = [
            FaultScenarioConfig(
                fault_class=FaultClass.MISFIRE,
                affected_cylinder=1,
                severity=0.8,
                start_time_s=0.0,
                duration_s=10.0,
            )
        ]
        physics = ForwardPhysicsModel(fault_configs=faults)
        gt = physics.compute_ground_truth(time_s=2.0, rpm=5000.0, throttle_pct=80.0, fault_scenario=faults[0])

        assert gt.active_fault == FaultClass.MISFIRE
        assert gt.egt_k[0] < gt.egt_k[1]  # Cylinder 1 EGT is lower due to misfire energy drop
        assert gt.vibration_exc_g > 0.0

    def test_all_nine_fault_classes(self):
        fault_classes = [
            FaultClass.NOMINAL,
            FaultClass.MISFIRE,
            FaultClass.DETONATION_KNOCK,
            FaultClass.EXHAUST_VALVE_LEAK,
            FaultClass.INTAKE_BOOST_LEAK,
            FaultClass.OIL_DEGRADATION,
            FaultClass.COOLING_FAULT,
            FaultClass.BEARING_WEAR,
            FaultClass.SENSOR_FAULT,
        ]

        for fc in fault_classes:
            faults = []
            f_scen = None
            if fc != FaultClass.NOMINAL:
                f_scen = FaultScenarioConfig(
                    fault_class=fc,
                    affected_cylinder=2,
                    affected_channel="egt_cyl1_hot_uv",
                    severity=0.5,
                    start_time_s=0.0,
                    duration_s=10.0,
                )
                faults.append(f_scen)
            physics = ForwardPhysicsModel(fault_configs=faults)
            gt = physics.compute_ground_truth(time_s=1.0, rpm=4500.0, throttle_pct=60.0, fault_scenario=f_scen)
            assert gt is not None
            if fc != FaultClass.NOMINAL:
                assert gt.active_fault == fc

    def test_invalid_fault_severity_raises(self):
        with pytest.raises(ValueError, match="Severity must be in range"):
            FaultScenarioConfig(
                fault_class=FaultClass.MISFIRE,
                severity=1.5,
            )


class TestSensorForwardModel:
    """Tests for sensor forward conversion and noise/effects."""

    def test_sensor_conversion_nominal(self):
        physics = ForwardPhysicsModel()
        gt = physics.step(time_s=1.0, rpm=5000.0, throttle_pct=80.0)

        sensor_model = SensorForwardModel(seed=42)
        raw_rec = sensor_model.generate_record(gt)

        assert isinstance(raw_rec, RawSignalRecord)
        assert raw_rec.source_type == Provenance.SIMULATED
        assert raw_rec.crank_period_us > 0
        assert raw_rec.egt_cyl1_hot_uv > 0.0
        assert raw_rec.oil_rtd_ohms > 0.0

    def test_sensor_fault_injection(self):
        faults = [
            FaultScenarioConfig(
                fault_class=FaultClass.SENSOR_FAULT,
                affected_channel="egt_cyl1_hot_uv",
                severity=1.0,
                start_time_s=0.0,
                duration_s=10.0,
            )
        ]
        physics = ForwardPhysicsModel(fault_configs=faults)
        gt = physics.step(time_s=1.0, rpm=5000.0, throttle_pct=80.0)

        sensor_model = SensorForwardModel(seed=42)
        raw_rec = sensor_model.generate_record(gt, fault_scenario=faults[0])

        # Sensor fault produces missing sample / corrupted voltage
        assert raw_rec.egt_cyl1_hot_uv == -999.0 or raw_rec.signal_quality.valid is False

    def test_deterministic_sensor_noise(self):
        physics = ForwardPhysicsModel(seed=123)
        gt1 = physics.step(time_s=1.0, rpm=5000.0, throttle_pct=80.0)
        gt2 = physics.step(time_s=1.0, rpm=5000.0, throttle_pct=80.0)

        sensor_model1 = SensorForwardModel(seed=42)
        raw_rec1 = sensor_model1.generate_record(gt1)

        sensor_model2 = SensorForwardModel(seed=42)
        raw_rec2 = sensor_model2.generate_record(gt2)

        assert raw_rec1.crank_period_us == raw_rec2.crank_period_us
        assert raw_rec1.egt_cyl1_hot_uv == raw_rec2.egt_cyl1_hot_uv
        assert raw_rec1.map_counts == raw_rec2.map_counts


class TestScenarioRunner:
    """Tests for ScenarioRunner time-series simulation."""

    def test_scenario_execution(self):
        runner = ScenarioRunner(seed=100)
        records, ground_truths, metadata = runner.run_scenario(
            duration_s=5.0,
            dt_s=1.0,
            rpm_profile=[3000.0, 4000.0, 5000.0, 5000.0, 5000.0],
            throttle_profile=[40.0, 60.0, 80.0, 80.0, 80.0],
        )

        assert len(records) == 5
        assert len(ground_truths) == 5
        assert metadata.total_samples == 5
        assert metadata.seed == 100

        # Verify ground truth is not attached inside RawSignalRecord
        for rec in records:
            assert rec.source_type == Provenance.SIMULATED
            assert not hasattr(rec, "ground_truth")
            assert not hasattr(rec, "true_power_kw")

    def test_deterministic_scenario_replay(self):
        runner1 = ScenarioRunner(seed=999)
        recs1, gts1, meta1 = runner1.run_scenario(duration_s=3.0, dt_s=1.0)

        runner2 = ScenarioRunner(seed=999)
        recs2, gts2, meta2 = runner2.run_scenario(duration_s=3.0, dt_s=1.0)

        for r1, r2 in zip(recs1, recs2):
            assert r1.crank_period_us == r2.crank_period_us
            assert r1.egt_cyl1_hot_uv == r2.egt_cyl1_hot_uv

        for g1, g2 in zip(gts1, gts2):
            assert g1.power_kw == g2.power_kw
            assert g1.oil_temp_k == g2.oil_temp_k
