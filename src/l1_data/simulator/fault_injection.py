"""
Fault Injection Engine — Injects realistic fault scenarios into the simulator.

Supports all 9 fault classes defined in the project taxonomy:
    0. Nominal (no fault)
    1. Misfire
    2. Detonation / Knock
    3. Exhaust Valve Leak
    4. Intake / Boost Leak
    5. Oil Degradation
    6. Cooling Fault
    7. Bearing Wear
    8. Sensor Fault

Each fault has:
    - severity: 0.0 (nominal) to 1.0 (complete failure)
    - onset_time_s: when the fault begins
    - progression_rate: how fast severity increases (0 = sudden, >0 = gradual)
    - affected_cylinders: which cylinders are affected (if applicable)

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class FaultMode(IntEnum):
    """Fault modes matching the project's 9-class taxonomy."""

    NOMINAL = 0
    MISFIRE = 1
    DETONATION_KNOCK = 2
    EXHAUST_VALVE_LEAK = 3
    INTAKE_BOOST_LEAK = 4
    OIL_DEGRADATION = 5
    COOLING_FAULT = 6
    BEARING_WEAR = 7
    SENSOR_FAULT = 8


@dataclass
class FaultScenario:
    """Configuration for a single fault injection scenario."""

    mode: FaultMode = FaultMode.NOMINAL
    severity: float = 0.0                 # 0.0 to 1.0
    onset_time_s: float = 0.0            # When fault begins [s]
    progression_rate: float = 0.0         # Severity increase per second
    affected_cylinders: list[int] = field(default_factory=lambda: [1])
    affected_sensor: str = ""             # For SENSOR_FAULT: which sensor
    sensor_fault_type: str = "stuck"      # stuck, drift, noise


@dataclass
class FaultState:
    """Current state of fault injection."""

    active_severity: float = 0.0
    time_active_s: float = 0.0
    is_active: bool = False


@dataclass
class FaultInjector:
    """Applies fault effects to simulator ground truth.

    Usage:
        injector = FaultInjector(scenario=FaultScenario(mode=FaultMode.MISFIRE, ...))
        modified_truth = injector.apply(ground_truth, current_time_s)
    """

    scenario: FaultScenario = field(default_factory=FaultScenario)
    fault_state: FaultState = field(default_factory=FaultState)

    def current_severity(self, time_s: float) -> float:
        """Compute current fault severity based on time and progression.

        Args:
            time_s: Current simulation time [s].

        Returns:
            Current severity [0.0, 1.0].
        """
        if self.scenario.mode == FaultMode.NOMINAL:
            return 0.0

        if time_s < self.scenario.onset_time_s:
            return 0.0

        elapsed = time_s - self.scenario.onset_time_s

        if self.scenario.progression_rate > 0:
            # Gradual degradation
            severity = min(
                self.scenario.severity,
                self.scenario.progression_rate * elapsed,
            )
        else:
            # Sudden onset
            severity = self.scenario.severity

        return min(severity, 1.0)

    def apply(self, ground_truth: dict, time_s: float) -> dict:
        """Apply fault effects to simulator ground truth values.

        This modifies the ground truth BEFORE it goes through the sensor
        model. The fault effects are realistic physical consequences of
        the fault mode.

        Args:
            ground_truth: Dictionary of true engine state values.
            time_s: Current simulation time [s].

        Returns:
            Modified ground truth dictionary.
        """
        severity = self.current_severity(time_s)
        self.fault_state = FaultState(
            active_severity=severity,
            time_active_s=max(0, time_s - self.scenario.onset_time_s),
            is_active=severity > 0,
        )

        if severity <= 0:
            return ground_truth

        # Copy to avoid mutating the original
        gt = dict(ground_truth)
        mode = self.scenario.mode

        if mode == FaultMode.MISFIRE:
            gt = self._apply_misfire(gt, severity)
        elif mode == FaultMode.DETONATION_KNOCK:
            gt = self._apply_detonation(gt, severity)
        elif mode == FaultMode.EXHAUST_VALVE_LEAK:
            gt = self._apply_exhaust_valve_leak(gt, severity)
        elif mode == FaultMode.INTAKE_BOOST_LEAK:
            gt = self._apply_intake_boost_leak(gt, severity)
        elif mode == FaultMode.OIL_DEGRADATION:
            gt = self._apply_oil_degradation(gt, severity)
        elif mode == FaultMode.COOLING_FAULT:
            gt = self._apply_cooling_fault(gt, severity)
        elif mode == FaultMode.BEARING_WEAR:
            gt = self._apply_bearing_wear(gt, severity)
        elif mode == FaultMode.SENSOR_FAULT:
            gt = self._apply_sensor_fault(gt, severity)

        return gt

    def _apply_misfire(self, gt: dict, severity: float) -> dict:
        """Misfire: zero heat release on affected cylinder(s).

        Physical effects:
        - EGT drops (no combustion → cooler exhaust)
        - Lambda goes lean (unburned fuel/air)
        - RPM fluctuation (missing power pulse)
        """
        for cyl in self.scenario.affected_cylinders:
            egt_key = f"egt_cyl_{cyl}"
            if egt_key in gt:
                # EGT drops significantly — unburned mixture
                gt[egt_key] = gt[egt_key] * (1.0 - 0.4 * severity)

            # Mark cylinder as misfiring (for cylinder model)
            gt[f"misfire_cyl_{cyl}"] = severity > 0.5

        # Lambda shifts lean (unburned charge passes through)
        gt["lambda_sensor"] = gt.get("lambda_sensor", 1.0) + 0.15 * severity
        return gt

    def _apply_detonation(self, gt: dict, severity: float) -> dict:
        """Detonation/Knock: pressure spikes, advanced timing effect.

        Physical effects:
        - EGT increases (excess heat)
        - CHT increases
        - Vibration increases (sharp pressure waves)
        """
        for cyl in self.scenario.affected_cylinders:
            egt_key = f"egt_cyl_{cyl}"
            cht_key = f"cht_cyl_{cyl}"
            if egt_key in gt:
                gt[egt_key] = gt[egt_key] * (1.0 + 0.15 * severity)
            if cht_key in gt:
                gt[cht_key] = gt[cht_key] + 15.0 * severity

        # Vibration increase from knock
        for axis in ["vibration_x", "vibration_y", "vibration_z"]:
            if axis in gt:
                gt[axis] = gt[axis] * (1.0 + 2.0 * severity)
        return gt

    def _apply_exhaust_valve_leak(self, gt: dict, severity: float) -> dict:
        """Exhaust valve leak: poor sealing reduces effective compression.

        Physical effects:
        - EGT rises on affected cylinder (hot gas leak)
        - Power loss (reduced compression)
        - Slightly lean lambda
        """
        for cyl in self.scenario.affected_cylinders:
            egt_key = f"egt_cyl_{cyl}"
            if egt_key in gt:
                gt[egt_key] = gt[egt_key] * (1.0 + 0.1 * severity)
        return gt

    def _apply_intake_boost_leak(self, gt: dict, severity: float) -> dict:
        """Intake/boost leak: MAP drops, power loss.

        Physical effects:
        - MAP decreases
        - Boost pressure decreases
        - Power loss proportional to MAP drop
        """
        map_reduction = 0.3 * severity  # Up to 30% MAP loss
        gt["map_pressure"] = gt["map_pressure"] * (1.0 - map_reduction)
        gt["boost_pressure"] = gt.get("boost_pressure", gt["map_pressure"]) * (1.0 - map_reduction)
        return gt

    def _apply_oil_degradation(self, gt: dict, severity: float) -> dict:
        """Oil system degradation: pressure drop, temperature rise.

        Physical effects:
        - Oil pressure decreases (worn pump, leaks)
        - Oil temperature increases (degraded cooling)
        """
        gt["oil_pressure"] = gt["oil_pressure"] * (1.0 - 0.5 * severity)
        gt["oil_temp"] = gt["oil_temp"] + 25.0 * severity
        return gt

    def _apply_cooling_fault(self, gt: dict, severity: float) -> dict:
        """Cooling system fault: CHT runaway.

        Physical effects:
        - CHT increases on all cylinders
        - Coolant temperature rises
        """
        for i in range(1, 5):
            cht_key = f"cht_cyl_{i}"
            if cht_key in gt:
                gt[cht_key] = gt[cht_key] + 30.0 * severity

        gt["coolant_temp"] = gt["coolant_temp"] + 20.0 * severity
        return gt

    def _apply_bearing_wear(self, gt: dict, severity: float) -> dict:
        """Bearing wear: increased friction and vibration.

        Physical effects:
        - Vibration amplitude increases
        - Oil temperature rises (more friction)
        - Slight power loss
        """
        for axis in ["vibration_x", "vibration_y", "vibration_z"]:
            if axis in gt:
                gt[axis] = gt[axis] * (1.0 + 3.0 * severity)

        gt["oil_temp"] = gt["oil_temp"] + 10.0 * severity
        return gt

    def _apply_sensor_fault(self, gt: dict, severity: float) -> dict:
        """Sensor fault: stuck, drift, or noise on selected sensor.

        This fault affects the SENSOR, not the engine.
        The ground truth is modified to simulate what a faulty sensor
        would report.
        """
        sensor = self.scenario.affected_sensor
        if not sensor or sensor not in gt:
            return gt

        fault_type = self.scenario.sensor_fault_type

        if fault_type == "stuck":
            # Sensor reads a constant value (freezes)
            if not hasattr(self, '_stuck_value'):
                self._stuck_value = gt[sensor]
            gt[sensor] = self._stuck_value

        elif fault_type == "drift":
            # Sensor drifts linearly from true value
            drift_amount = gt[sensor] * 0.2 * severity
            gt[sensor] = gt[sensor] + drift_amount

        elif fault_type == "noise":
            # Sensor noise increases dramatically
            import numpy as np
            rng = np.random.default_rng()
            noise_amp = gt[sensor] * 0.1 * severity
            gt[sensor] = gt[sensor] + float(rng.normal(0, noise_amp))

        return gt
