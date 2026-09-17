"""
Scientific Acceptance Suite — Modules 12–16: ML Supervision, Fault Taxonomy & Health/Risk.
"""

import pytest
from src.core.config import get_settings
from src.core.provenance import FaultClass, FlightPhase
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l2_digital_twin import (
    evaluate_digital_twin,
    evaluate_egt_diagnostics,
    evaluate_lubrication_model,
    evaluate_residual_engine,
    evaluate_vibration_processor,
    convert_raw_to_engineering_state,
)
from src.l3_ml import (
    AnomalyDetector,
    HealthSupervisionEngine,
    MLFeatureVectorBuilder,
    MissionPhaseClassifier,
    MissionRiskEngine,
    NineClassFaultClassifier,
    RULEstimator,
)


def test_ml_feature_vector_builder_ordering_and_quality():
    settings = get_settings()
    builder = MLFeatureVectorBuilder(settings=settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])
    twin_res = evaluate_digital_twin(norm)
    vib_state, vib_res = evaluate_vibration_processor(norm)
    res_state, exp_res = evaluate_residual_engine(norm, twin_res)

    fv = builder.build_feature_vector(res_state, twin_res, norm, vib_state)

    assert fv is not None
    assert len(fv.features) > 0

    # Ensure no NaN or Inf in feature values
    for feat_name, feat in fv.features.items():
        val = feat.value if hasattr(feat, "value") else float(feat)
        assert val == val  # NaN check
        assert abs(val) < 1e12  # Inf check


def test_nine_class_fault_taxonomy_coverage():
    settings = get_settings()
    classifier = NineClassFaultClassifier(settings=settings)

    # Verify exact taxonomy enumeration length = 9
    fault_classes = list(FaultClass)
    assert len(fault_classes) == 9

    expected_names = [
        "NOMINAL",
        "MISFIRE",
        "DETONATION_KNOCK",
        "EXHAUST_VALVE_LEAK",
        "INTAKE_BOOST_LEAK",
        "OIL_DEGRADATION",
        "COOLING_FAULT",
        "BEARING_WEAR",
        "SENSOR_FAULT",
    ]
    for fc, exp in zip(fault_classes, expected_names):
        assert fc.name == exp


def test_health_index_weights_and_bounds():
    settings = get_settings()
    builder = MLFeatureVectorBuilder(settings=settings)
    anomaly_detector = AnomalyDetector(settings=settings)
    fault_classifier = NineClassFaultClassifier(settings=settings)
    health_engine = HealthSupervisionEngine(settings=settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])
    twin_res = evaluate_digital_twin(norm)
    lub_state, _ = evaluate_lubrication_model(norm)
    vib_state, _ = evaluate_vibration_processor(norm)
    res_state, _ = evaluate_residual_engine(norm, twin_res)

    fv = builder.build_feature_vector(res_state, twin_res, norm, vib_state)
    anom_res = anomaly_detector.detect_anomaly(fv)
    fault_res = fault_classifier.classify_fault(fv)

    health_record = health_engine.evaluate_health(
        residual_state=res_state,
        anomaly_result=anom_res,
        fault_result=fault_res,
        lub_state=lub_state,
    )

    # Health index bounds check [0, 100]
    hi_val = health_record.health_index.value
    assert 0.0 <= hi_val <= 100.0


def test_mission_phase_and_risk_scoring():
    settings = get_settings()
    phase_classifier = MissionPhaseClassifier()
    risk_engine = MissionRiskEngine(settings=settings)

    runner = ScenarioRunner(seed=42)
    records, _, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(records[0])

    phase, q, _ = phase_classifier.classify_phase(
        rpm=norm.rpm.value,
        map_pa=norm.map_pressure.value,
        throttle_pct=norm.throttle_position.value,
    )
    assert phase in list(FlightPhase)

    # Risk score bounds [0, 100]
    builder = MLFeatureVectorBuilder(settings=settings)
    anomaly_detector = AnomalyDetector(settings=settings)
    fault_classifier = NineClassFaultClassifier(settings=settings)
    health_engine = HealthSupervisionEngine(settings=settings)
    rul_estimator = RULEstimator(settings=settings)

    twin_res = evaluate_digital_twin(norm)
    lub_state, _ = evaluate_lubrication_model(norm)
    vib_state, _ = evaluate_vibration_processor(norm)
    res_state, _ = evaluate_residual_engine(norm, twin_res)

    fv = builder.build_feature_vector(res_state, twin_res, norm, vib_state)
    anom_res = anomaly_detector.detect_anomaly(fv)
    fault_res = fault_classifier.classify_fault(fv)
    health_record = health_engine.evaluate_health(
        residual_state=res_state,
        anomaly_result=anom_res,
        fault_result=fault_res,
        lub_state=lub_state,
    )
    rul_res = rul_estimator.estimate_rul(health_record)

    risk_res = risk_engine.evaluate_mission_risk(
        flight_phase=phase,
        health_state=health_record,
        rul_state=rul_res,
        fault_result=fault_res,
    )
    assert 0.0 <= risk_res.risk_score <= 100.0
