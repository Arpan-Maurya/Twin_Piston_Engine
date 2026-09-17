"""
Virtual Sensor Models — Converts simulator ground truth to realistic signals.

This is the critical module that transforms perfect simulator state into
realistic sensor readings by adding:
    - Measurement noise (Gaussian)
    - Sensor dynamics (first-order lag)
    - Quantization (ADC resolution)
    - Occasional dropouts

After this module, the data looks like it came from real hardware.
The NormalizedSignalRecord produced is indistinguishable in format
from what a real telemetry system would produce.

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.core.provenance import FlightPhase, Provenance
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord


@dataclass
class SensorNoiseConfig:
    """Noise parameters for each sensor type."""

    rpm_std: float = 5.0
    temperature_std_k: float = 1.5
    pressure_std_pa: float = 500.0
    vibration_std_m_s2: float = 0.2
    fuel_flow_std_kg_s: float = 0.0001
    voltage_std_v: float = 0.05
    lambda_std: float = 0.005
    timing_std_deg: float = 0.1
    dropout_probability: float = 0.001


@dataclass
class SensorModel:
    """Virtual sensor layer that converts ground truth to realistic readings.

    Each sensor channel has:
    - Gaussian noise with configurable standard deviation
    - First-order lag (sensor dynamics, time constant τ)
    - Random dropout probability
    """

    noise_config: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    # Previous values for first-order lag (keyed by channel name)
    _previous: dict[str, float] = field(default_factory=dict)

    # Sensor time constant [s] — how quickly sensor responds
    sensor_tau_s: float = 0.1

    def _add_noise(self, value: float, std: float) -> float:
        """Add Gaussian measurement noise."""
        return value + float(self.rng.normal(0.0, std))

    def _apply_lag(self, channel: str, value: float, dt_s: float) -> float:
        """Apply first-order sensor lag.

        y[n] = y[n-1] + α · (x[n] - y[n-1])
        where α = dt / (τ + dt)

        Args:
            channel: Channel name for state tracking.
            value: True (noisy) value.
            dt_s: Time step [s].

        Returns:
            Lagged value.
        """
        alpha = dt_s / (self.sensor_tau_s + dt_s) if (self.sensor_tau_s + dt_s) > 0 else 1.0
        prev = self._previous.get(channel, value)
        filtered = prev + alpha * (value - prev)
        self._previous[channel] = filtered
        return filtered

    def _is_dropout(self) -> bool:
        """Random sensor dropout."""
        return float(self.rng.random()) < self.noise_config.dropout_probability

    def _make_channel(
        self,
        name: str,
        true_value: float,
        noise_std: float,
        dt_s: float,
        min_val: float = -float("inf"),
        max_val: float = float("inf"),
    ) -> ChannelValue:
        """Create a realistic channel value from ground truth.

        Args:
            name: Channel name.
            true_value: Simulator ground truth value.
            noise_std: Noise standard deviation.
            dt_s: Time step for sensor lag.
            min_val: Physical minimum (clamp).
            max_val: Physical maximum (clamp).

        Returns:
            ChannelValue with noise, lag, and potential dropout.
        """
        if self._is_dropout():
            return ChannelValue(
                value=self._previous.get(name, true_value),
                provenance=Provenance.SIMULATED,
                valid=False,
                quality=0.0,
                fault_flag="sensor_dropout",
            )

        noisy = self._add_noise(true_value, noise_std)
        lagged = self._apply_lag(name, noisy, dt_s)
        clamped = max(min_val, min(max_val, lagged))

        return ChannelValue(
            value=clamped,
            provenance=Provenance.SIMULATED,
            valid=True,
            quality=0.95,  # Simulated data quality
        )

    def generate_record(
        self,
        ground_truth: dict,
        sequence_number: int,
        dt_s: float,
    ) -> NormalizedSignalRecord:
        """Convert simulator ground truth into a NormalizedSignalRecord.

        This is where ground truth becomes a realistic raw signal.
        After this function, the data is indistinguishable in format
        from real hardware telemetry.

        Args:
            ground_truth: Dictionary of true values from the simulator.
            sequence_number: Monotonic sequence counter.
            dt_s: Time step since last sample [s].

        Returns:
            NormalizedSignalRecord ready for L2 consumption.
        """
        nc = self.noise_config

        # Build all channels
        channels = {}

        channels["rpm"] = self._make_channel(
            "rpm", ground_truth["rpm"], nc.rpm_std, dt_s, 0, 6500
        )
        channels["map_pressure"] = self._make_channel(
            "map_pressure", ground_truth["map_pressure"], nc.pressure_std_pa, dt_s, 20000, 200000
        )
        channels["throttle_position"] = self._make_channel(
            "throttle_position", ground_truth["throttle_position"], 0.5, dt_s, 0, 100
        )

        # Per-cylinder EGT
        for i in range(1, 5):
            key = f"egt_cyl_{i}"
            channels[key] = self._make_channel(
                key, ground_truth[key], nc.temperature_std_k, dt_s, 300, 1200
            )

        # Per-cylinder CHT
        for i in range(1, 5):
            key = f"cht_cyl_{i}"
            channels[key] = self._make_channel(
                key, ground_truth[key], nc.temperature_std_k, dt_s, 300, 500
            )

        # Oil system
        channels["oil_temp"] = self._make_channel(
            "oil_temp", ground_truth["oil_temp"], nc.temperature_std_k, dt_s, 270, 430
        )
        channels["oil_pressure"] = self._make_channel(
            "oil_pressure", ground_truth["oil_pressure"], nc.pressure_std_pa, dt_s, 0, 800000
        )

        # Cooling
        channels["coolant_temp"] = self._make_channel(
            "coolant_temp", ground_truth["coolant_temp"], nc.temperature_std_k, dt_s, 270, 420
        )

        # Fuel
        channels["fuel_flow"] = self._make_channel(
            "fuel_flow", ground_truth["fuel_flow"], nc.fuel_flow_std_kg_s, dt_s, 0, 0.015
        )
        channels["fuel_pressure"] = self._make_channel(
            "fuel_pressure", ground_truth["fuel_pressure"], nc.pressure_std_pa, dt_s, 100000, 600000
        )

        # Intake
        channels["intake_air_temp"] = self._make_channel(
            "intake_air_temp", ground_truth["intake_air_temp"], nc.temperature_std_k, dt_s, 220, 340
        )

        # Ambient
        channels["ambient_pressure"] = self._make_channel(
            "ambient_pressure", ground_truth["ambient_pressure"], nc.pressure_std_pa * 0.1, dt_s,
            30000, 110000
        )
        channels["ambient_temp"] = self._make_channel(
            "ambient_temp", ground_truth["ambient_temp"], nc.temperature_std_k * 0.5, dt_s,
            210, 330
        )

        # Electrical
        channels["voltage"] = self._make_channel(
            "voltage", ground_truth.get("voltage", 13.8), nc.voltage_std_v, dt_s, 0, 16
        )
        channels["current"] = self._make_channel(
            "current", ground_truth.get("current", 15.0), 0.5, dt_s, 0, 50
        )

        # Vibration
        channels["vibration_x"] = self._make_channel(
            "vibration_x", ground_truth.get("vibration_x", 2.0), nc.vibration_std_m_s2, dt_s, 0, 500
        )
        channels["vibration_y"] = self._make_channel(
            "vibration_y", ground_truth.get("vibration_y", 2.0), nc.vibration_std_m_s2, dt_s, 0, 500
        )
        channels["vibration_z"] = self._make_channel(
            "vibration_z", ground_truth.get("vibration_z", 3.0), nc.vibration_std_m_s2, dt_s, 0, 500
        )
        vib_rms = math.sqrt(
            channels["vibration_x"].value ** 2
            + channels["vibration_y"].value ** 2
            + channels["vibration_z"].value ** 2
        )
        channels["vibration_rms"] = ChannelValue(
            value=vib_rms, provenance=Provenance.SIMULATED, valid=True, quality=0.95
        )

        # Propeller
        channels["propeller_speed"] = self._make_channel(
            "propeller_speed", ground_truth.get("propeller_speed", ground_truth["rpm"] * 0.4),
            nc.rpm_std, dt_s, 0, 3000
        )

        # Turbo
        channels["boost_pressure"] = self._make_channel(
            "boost_pressure", ground_truth["boost_pressure"], nc.pressure_std_pa, dt_s,
            80000, 200000
        )
        channels["wastegate_duty"] = self._make_channel(
            "wastegate_duty", ground_truth.get("wastegate_duty", 50.0), 1.0, dt_s, 0, 100
        )

        # Lambda
        channels["lambda_sensor"] = self._make_channel(
            "lambda_sensor", ground_truth.get("lambda_sensor", 1.0), nc.lambda_std, dt_s, 0.5, 1.5
        )

        # Ignition timing per cylinder
        for i in range(1, 5):
            key = f"ignition_timing_cyl_{i}"
            channels[key] = self._make_channel(
                key, ground_truth.get(key, 25.0), nc.timing_std_deg, dt_s, 0, 60
            )

        # Flight context
        channels["altitude"] = self._make_channel(
            "altitude", ground_truth.get("altitude", 0.0), 1.0, dt_s, 0, 15000
        )
        channels["engine_hours"] = ChannelValue(
            value=ground_truth.get("engine_hours", 0.0),
            provenance=Provenance.SIMULATED,
            valid=True,
            quality=1.0,
        )

        # Compute integrity hash
        hash_input = json.dumps(
            {k: v.value for k, v in channels.items()},
            sort_keys=True,
        ).encode()
        integrity_hash = hashlib.sha256(hash_input).hexdigest()

        flight_phase = ground_truth.get("flight_phase", FlightPhase.GROUND)
        if isinstance(flight_phase, str):
            flight_phase = FlightPhase(flight_phase)

        return NormalizedSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=sequence_number,
            source_id="rotax_915is_simulator",
            provenance=Provenance.SIMULATED,
            integrity_hash=integrity_hash,
            flight_phase=flight_phase,
            **channels,
        )
