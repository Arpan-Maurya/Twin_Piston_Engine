"""
Forward Physics Model, Fault Injection and Scenario Simulator — Original Module 17.

First stage of L1 Forward Simulation & Scenario Generation.
Provides deterministic forward physics modeling, sensor forward conversion to canonical
RawSignalRecord, controlled 9-class fault injection, and scenario replay capabilities.

STRICT BOUNDARY CONSTRAINTS:
    - Simulator is a source of simulated telemetry ONLY.
    - Ground truth physical state remains in SimulationGroundTruth and MUST NOT be inserted into RawSignalRecord.
    - L2 and L3 MUST NOT import or access simulator ground truth or private internals.
    - Emits canonical RawSignalRecord with Provenance.SIMULATED.
    - 100% deterministic replay given identical scenario parameters and seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import FaultClass, FlightPhase, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


@dataclass
class SimulationGroundTruth:
    """Isolated, simulation-only ground truth state.

    MUST NOT be inserted into RawSignalRecord or exposed to L2/L3.
    """

    timestamp: datetime
    time_s: float
    sequence_number: int
    rpm: float
    map_pressure_pa: float
    throttle_pct: float
    altitude_m: float
    ambient_temp_k: float
    ambient_pressure_pa: float
    egt_k: list[float] = field(default_factory=lambda: [950.0, 950.0, 950.0, 950.0])
    cht_k: float = 383.15  # 110 °C
    oil_temp_k: float = 363.15  # 90 °C
    oil_pressure_pa: float = 400000.0  # 4 bar
    vibration_rms_m_s2: float = 5.0
    vibration_exc_g: float = 0.5
    fuel_flow_kg_s: float = 0.005
    brake_power_kw: float = 80.0
    torque_nm: float = 185.0
    active_fault: FaultClass = FaultClass.NOMINAL
    injected_fault_class: FaultClass = FaultClass.NOMINAL
    injected_fault_severity: float = 0.0
    affected_cylinders: list[int] = field(default_factory=lambda: [1])
    provenance: Provenance = Provenance.SIMULATED

    @property
    def power_kw(self) -> float:
        return self.brake_power_kw


@dataclass
class FaultScenarioConfig:
    """Configuration for a single fault injection scenario event."""

    fault_class: FaultClass = FaultClass.NOMINAL
    severity: float = 0.0  # Bounded [0.0, 1.0]
    onset_time_s: float = 0.0  # Start time in simulation [s]
    start_time_s: float | None = None  # Alias for onset_time_s
    duration_s: float = 300.0  # Duration of fault event [s]
    affected_cylinders: list[int] = field(default_factory=lambda: [1])
    affected_cylinder: int | None = None  # Alias for single cylinder
    affected_channel: str | None = None  # For SENSOR_FAULT
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"Severity must be in range [0.0, 1.0], got {self.severity}")

        if self.start_time_s is not None:
            self.onset_time_s = self.start_time_s

        if self.affected_cylinder is not None:
            self.affected_cylinders = [self.affected_cylinder]


@dataclass
class SimulationMetadata:
    """Explicit simulation run metadata."""

    scenario_id: str
    seed: int
    simulator_version: str = "1.0.0"
    configuration_version: str = "1.0.0"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float = 0.0
    total_samples: int = 0
    fault_events: list[FaultScenarioConfig] = field(default_factory=list)


class ForwardPhysicsModel:
    """Deterministic forward-physics engine model for the Rotax 915 iS."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        fault_configs: list[FaultScenarioConfig] | None = None,
        seed: int = 42,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine_params = self._settings.engine
        self._fault_configs = fault_configs or []
        self._seed = seed

    def compute_ground_truth(
        self,
        time_s: float,
        sequence_number: int = 1,
        rpm: float = 4000.0,
        map_pa: float = 100000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 0.0,
        ambient_temp_k: float = 288.15,
        ambient_pressure_pa: float = 101325.0,
        fault_scenario: FaultScenarioConfig | None = None,
        timestamp: datetime | None = None,
    ) -> SimulationGroundTruth:
        """Compute true physical engine state from operating parameters and fault injections."""
        ts = timestamp or (datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=time_s))

        # Clamp operating inputs
        c_rpm = max(500.0, min(6000.0, rpm))
        c_map = max(30000.0, min(160000.0, map_pa))
        c_throttle = max(0.0, min(100.0, throttle_pct))

        # Nominal physics derivations based on Rotax 915 iS reference parameters
        base_egt_k = 900.0 + (c_map / 100000.0) * 50.0 + (c_rpm / 6000.0) * 30.0
        egt_k = [base_egt_k] * 4

        cht_k = 363.15 + (c_map / 100000.0) * 15.0 + (c_rpm / 6000.0) * 10.0
        oil_temp_k = 353.15 + (c_rpm / 6000.0) * 20.0
        oil_pressure_pa = 200000.0 + (c_rpm / 6000.0) * 300000.0
        vibration_rms_m_s2 = 3.0 + (c_rpm / 6000.0) * 4.0
        vibration_exc_g = 0.3 + (c_rpm / 6000.0) * 0.4
        fuel_flow_kg_s = 0.001 + (c_map / 100000.0) * 0.004 + (c_throttle / 100.0) * 0.002
        brake_power_kw = (c_rpm / 5800.0) * (c_map / 135000.0) * 105.0
        torque_nm = (brake_power_kw * 1000.0) / max(1.0, (2.0 * math.pi * c_rpm / 60.0))

        active_fault = FaultClass.NOMINAL
        severity = 0.0
        aff_cyls: list[int] = [1]

        # Determine active fault scenario from explicit parameter or fault_configs list
        f_active = fault_scenario
        if f_active is None and self._fault_configs:
            for f_cfg in self._fault_configs:
                f_start = f_cfg.onset_time_s
                f_end = f_start + f_cfg.duration_s
                if f_start <= time_s <= f_end:
                    f_active = f_cfg
                    break

        # Apply active fault injection scenario effects on physical state
        if f_active and f_active.fault_class != FaultClass.NOMINAL:
            f_start = f_active.onset_time_s
            f_end = f_start + f_active.duration_s
            if f_start <= time_s <= f_end:
                active_fault = f_active.fault_class
                severity = max(0.0, min(1.0, f_active.severity))
                aff_cyls = f_active.affected_cylinders

                if active_fault == FaultClass.MISFIRE:
                    for c_id in aff_cyls:
                        if 1 <= c_id <= 4:
                            egt_k[c_id - 1] -= severity * 150.0
                    brake_power_kw *= max(0.2, 1.0 - severity * 0.25)
                    vibration_rms_m_s2 += severity * 8.0
                    vibration_exc_g += severity * 1.5

                elif active_fault == FaultClass.DETONATION_KNOCK:
                    cht_k += severity * 40.0
                    for c_id in aff_cyls:
                        if 1 <= c_id <= 4:
                            egt_k[c_id - 1] += severity * 50.0
                    vibration_rms_m_s2 += severity * 12.0
                    vibration_exc_g += severity * 2.0

                elif active_fault == FaultClass.EXHAUST_VALVE_LEAK:
                    for c_id in aff_cyls:
                        if 1 <= c_id <= 4:
                            egt_k[c_id - 1] -= severity * 180.0
                    brake_power_kw *= max(0.5, 1.0 - severity * 0.15)

                elif active_fault == FaultClass.INTAKE_BOOST_LEAK:
                    c_map = max(40000.0, c_map - severity * 35000.0)
                    brake_power_kw *= max(0.4, 1.0 - severity * 0.3)

                elif active_fault == FaultClass.OIL_DEGRADATION:
                    oil_pressure_pa = max(50000.0, oil_pressure_pa - severity * 220000.0)
                    oil_temp_k += severity * 18.0

                elif active_fault == FaultClass.COOLING_FAULT:
                    cht_k += severity * 35.0
                    oil_temp_k += severity * 25.0

                elif active_fault == FaultClass.BEARING_WEAR:
                    vibration_rms_m_s2 += severity * 22.0
                    vibration_exc_g += severity * 3.0
                    brake_power_kw *= max(0.7, 1.0 - severity * 0.1)

                elif active_fault == FaultClass.SENSOR_FAULT:
                    # Physical ground truth unaltered
                    pass

        return SimulationGroundTruth(
            timestamp=ts,
            time_s=time_s,
            sequence_number=sequence_number,
            rpm=c_rpm,
            map_pressure_pa=c_map,
            throttle_pct=c_throttle,
            altitude_m=altitude_m,
            ambient_temp_k=ambient_temp_k,
            ambient_pressure_pa=ambient_pressure_pa,
            egt_k=egt_k,
            cht_k=cht_k,
            oil_temp_k=oil_temp_k,
            oil_pressure_pa=oil_pressure_pa,
            vibration_rms_m_s2=vibration_rms_m_s2,
            vibration_exc_g=vibration_exc_g,
            fuel_flow_kg_s=fuel_flow_kg_s,
            brake_power_kw=brake_power_kw,
            torque_nm=torque_nm,
            active_fault=active_fault,
            injected_fault_class=active_fault,
            injected_fault_severity=severity,
            affected_cylinders=aff_cyls,
            provenance=Provenance.SIMULATED,
        )

    def step(
        self,
        time_s: float = 0.0,
        sequence_number: int = 1,
        rpm: float = 4000.0,
        map_pa: float = 100000.0,
        throttle_pct: float = 50.0,
        altitude_m: float = 0.0,
    ) -> SimulationGroundTruth:
        """Convenience single-step wrapper for compute_ground_truth."""
        return self.compute_ground_truth(
            time_s=time_s,
            sequence_number=sequence_number,
            rpm=rpm,
            map_pa=map_pa,
            throttle_pct=throttle_pct,
            altitude_m=altitude_m,
        )


class SensorForwardModel:
    """Converts physical ground truth into canonical acquisition RawSignalRecord."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset RNG seed for 100% deterministic replay."""
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def convert_to_raw_record(
        self,
        ground_truth: SimulationGroundTruth,
        fault_scenario: FaultScenarioConfig | None = None,
        enable_noise: bool = True,
    ) -> RawSignalRecord:
        """Convert ground truth physical state into canonical RawSignalRecord."""
        # 1. Type-K Thermocouple conversion (EGT & CHT in microvolts)
        egt_cold_c = ground_truth.ambient_temp_k - 273.15
        type_k_uv_per_c = 41.27

        egt_uv = [
            (k - 273.15 - egt_cold_c) * type_k_uv_per_c
            for k in ground_truth.egt_k
        ]

        cht_cold_c = egt_cold_c
        cht_uv = (ground_truth.cht_k - 273.15 - cht_cold_c) * type_k_uv_per_c

        # 2. PT100 RTD Oil Temperature conversion (Ohms)
        oil_temp_c = ground_truth.oil_temp_k - 273.15
        oil_rtd_ohms = 100.0 * (1.0 + 0.00385 * oil_temp_c)

        # 3. ADC Pressure conversion (Counts)
        oil_p_counts = int((ground_truth.oil_pressure_pa / 800000.0) * 4095.0)
        map_counts = int((ground_truth.map_pressure_pa / 200000.0) * 4095.0)
        adc_vref_counts = 4095

        # 4. Crankshaft period (microseconds)
        crank_period_us = (60.0 * 1e6) / max(100.0, ground_truth.rpm)

        # 5. Fuel flow pulse frequency (Hz)
        fuel_pulse_hz = ground_truth.fuel_flow_kg_s / 0.0001

        # 6. Accelerometer ADC counts (X, Y, Z)
        vib_count = int(ground_truth.vibration_rms_m_s2 * 10.0)
        accel_counts_xyz = (vib_count, vib_count, vib_count)

        # Ambient
        amb_temp_c = ground_truth.ambient_temp_k - 273.15
        amb_press_pa = ground_truth.ambient_pressure_pa

        # 7. Add deterministic noise if enabled
        if enable_noise:
            noise_egt = self._rng.normal(0.0, 50.0, size=4)
            egt_uv = [u + float(n) for u, n in zip(egt_uv, noise_egt)]
            cht_uv += float(self._rng.normal(0.0, 30.0))
            oil_rtd_ohms += float(self._rng.normal(0.0, 0.05))
            oil_p_counts = int(oil_p_counts + self._rng.normal(0.0, 5.0))
            map_counts = int(map_counts + self._rng.normal(0.0, 5.0))
            crank_period_us += float(self._rng.normal(0.0, 2.0))
            fuel_pulse_hz += float(self._rng.normal(0.0, 0.1))

        # 8. Sensor Fault Injection handling
        signal_quality = SignalQuality(valid=True, quality=1.0)
        if fault_scenario and fault_scenario.fault_class == FaultClass.SENSOR_FAULT:
            f_start = fault_scenario.onset_time_s
            f_end = f_start + fault_scenario.duration_s
            if f_start <= ground_truth.time_s <= f_end:
                target_chan = fault_scenario.affected_channel or "egt_hot_junction_mv_c1"
                if target_chan in ("egt_hot_junction_mv_c1", "egt_cyl1_hot_uv"):
                    egt_uv[0] = -999.0  # Dropout / invalid reading
                    signal_quality = SignalQuality(valid=False, quality=0.0, fault_flag="SENSOR_DROPOUT")
                elif target_chan == "map_counts":
                    map_counts = 0
                    signal_quality = SignalQuality(valid=False, quality=0.0, fault_flag="SENSOR_DROPOUT")
                elif target_chan == "oil_p_counts":
                    oil_p_counts = 4095  # Saturation
                    signal_quality = SignalQuality(valid=False, quality=0.0, fault_flag="SENSOR_SATURATION")

        # Clamp ADC counts to [0, 4095]
        oil_p_counts = max(0, min(4095, oil_p_counts))
        map_counts = max(0, min(4095, map_counts))
        adc_vref_counts = max(0, min(4095, adc_vref_counts))

        rec = RawSignalRecord(
            timestamp=ground_truth.timestamp,
            sequence_number=ground_truth.sequence_number,
            source_type=Provenance.SIMULATED,
            egt_cyl1_hot_uv=egt_uv[0],
            egt_cyl2_hot_uv=egt_uv[1],
            egt_cyl3_hot_uv=egt_uv[2],
            egt_cyl4_hot_uv=egt_uv[3],
            egt_cold_c=egt_cold_c,
            cht_hot_uv=cht_uv,
            cht_cold_c=cht_cold_c,
            oil_rtd_ohms=oil_rtd_ohms,
            oil_p_counts=oil_p_counts,
            map_counts=map_counts,
            adc_vref_counts=adc_vref_counts,
            crank_period_us=max(100.0, crank_period_us),
            fuel_pulse_hz=max(0.0, fuel_pulse_hz),
            accel_counts_xyz=accel_counts_xyz,
            ambient_temp_c=amb_temp_c,
            ambient_press_pa=amb_press_pa,
            signal_quality=signal_quality,
        )

        return rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash()})

    def generate_record(
        self,
        ground_truth: SimulationGroundTruth,
        fault_scenario: FaultScenarioConfig | None = None,
        enable_noise: bool = True,
    ) -> RawSignalRecord:
        """Alias for convert_to_raw_record."""
        return self.convert_to_raw_record(
            ground_truth=ground_truth,
            fault_scenario=fault_scenario,
            enable_noise=enable_noise,
        )


class ScenarioRunner:
    """Deterministic scenario runner for generating raw telemetry sequences and ground truth logs."""

    def __init__(self, settings: AppSettings | None = None, seed: int = 42) -> None:
        self._settings = settings or get_settings()
        self._seed = seed
        self._physics_model = ForwardPhysicsModel(settings=self._settings, seed=seed)

    def run_scenario(
        self,
        duration_s: float = 60.0,
        dt_s: float = 0.1,
        timestep_s: float | None = None,
        rpm_profile: list[float] | None = None,
        throttle_profile: list[float] | None = None,
        operating_profile: list[dict[str, float]] | None = None,
        fault_scenarios: list[FaultScenarioConfig] | None = None,
        seed: int | None = None,
        enable_noise: bool = True,
    ) -> tuple[list[RawSignalRecord], list[SimulationGroundTruth], SimulationMetadata]:
        """Run a deterministic scenario simulation over duration_s.

        Returns (raw_telemetry_records, ground_truth_logs, metadata).
        """
        active_seed = seed if seed is not None else self._seed
        sensor_model = SensorForwardModel(seed=active_seed)
        raw_records: list[RawSignalRecord] = []
        ground_truth_logs: list[SimulationGroundTruth] = []

        actual_dt = dt_s if timestep_s is None else timestep_s
        actual_dt = max(0.01, min(10.0, float(actual_dt)))
        total_steps = min(int(duration_s / actual_dt), 36000)
        base_time = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        for step in range(total_steps):
            t_s = step * actual_dt

            # Default profile: 4000 RPM, 100 kPa MAP, 50% throttle
            rpm = 4000.0
            map_pa = 100000.0
            throttle_pct = 50.0
            altitude_m = 0.0

            if rpm_profile and step < len(rpm_profile):
                rpm = rpm_profile[step]

            if throttle_profile and step < len(throttle_profile):
                throttle_pct = throttle_profile[step]

            # Override from operating_profile if specified
            if operating_profile:
                for prof in operating_profile:
                    p_start = prof.get("start_time_s", 0.0)
                    p_end = prof.get("end_time_s", duration_s)
                    if p_start <= t_s <= p_end:
                        rpm = prof.get("rpm", rpm)
                        map_pa = prof.get("map_pa", map_pa)
                        throttle_pct = prof.get("throttle_pct", throttle_pct)
                        altitude_m = prof.get("altitude_m", altitude_m)

            # Find active fault scenario for current timestep
            active_fault: FaultScenarioConfig | None = None
            if fault_scenarios:
                for f_scen in fault_scenarios:
                    if f_scen.onset_time_s <= t_s <= (f_scen.onset_time_s + f_scen.duration_s):
                        active_fault = f_scen
                        break

            ts = base_time + timedelta(seconds=t_s)
            gt = self._physics_model.compute_ground_truth(
                time_s=t_s,
                sequence_number=step + 1,
                rpm=rpm,
                map_pa=map_pa,
                throttle_pct=throttle_pct,
                altitude_m=altitude_m,
                fault_scenario=active_fault,
                timestamp=ts,
            )

            raw_rec = sensor_model.convert_to_raw_record(
                ground_truth=gt,
                fault_scenario=active_fault,
                enable_noise=enable_noise,
            )

            ground_truth_logs.append(gt)
            raw_records.append(raw_rec)

        metadata = SimulationMetadata(
            scenario_id=f"scen_{active_seed}_{int(duration_s)}s",
            seed=active_seed,
            duration_s=duration_s,
            total_samples=len(raw_records),
            fault_events=fault_scenarios or [],
        )

        return raw_records, ground_truth_logs, metadata
