"""
Turbocharger Model — Compressor, turbine, and wastegate simulation.

Models the turbocharging system of the Rotax 915 iS:
- Centrifugal compressor with isentropic efficiency
- Radial turbine extracting exhaust enthalpy
- Wastegate with PI controller targeting MAP setpoint

Key equations:
    - Compressor outlet temperature: Heywood (2018), §6.3
    - Turbine power extraction:      Heywood (2018), §6.3
    - Turbo shaft dynamics:           Watson & Janota (1982), Ch. 4

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.core.constants import CP_AIR, GAMMA_AIR
from src.l1_data.simulator.rotax_915is_params import Rotax915iSParams


@dataclass
class TurboState:
    """Instantaneous turbocharger state."""

    shaft_speed_rpm: float = 50000.0   # Turbo shaft speed
    compressor_pr: float = 1.0         # Compressor pressure ratio
    compressor_outlet_temp_k: float = 288.15
    compressor_outlet_pressure_pa: float = 101325.0
    compressor_power_w: float = 0.0
    turbine_power_w: float = 0.0
    turbine_inlet_temp_k: float = 800.0
    wastegate_position: float = 0.0    # 0 = closed, 1 = fully open
    wastegate_duty_pct: float = 0.0

    # PI controller state
    integral_error: float = 0.0


@dataclass
class TurbochargerModel:
    """Turbocharger model for the Rotax 915 iS.

    Simplified model using isentropic efficiency for both compressor
    and turbine. The wastegate is modulated by a PI controller to
    maintain the target MAP.
    """

    params: Rotax915iSParams
    state: TurboState = field(default_factory=TurboState)

    def update(
        self,
        ambient_pressure_pa: float,
        ambient_temp_k: float,
        exhaust_temp_k: float,
        exhaust_mass_flow_kg_s: float,
        target_map_pa: float,
        current_map_pa: float,
        dt_s: float,
    ) -> TurboState:
        """Update turbocharger state for one time step.

        Args:
            ambient_pressure_pa: Ambient atmospheric pressure [Pa].
            ambient_temp_k: Ambient air temperature [K].
            exhaust_temp_k: Mean exhaust gas temperature [K].
            exhaust_mass_flow_kg_s: Exhaust mass flow rate [kg/s].
            target_map_pa: Target manifold pressure [Pa].
            current_map_pa: Current actual MAP [Pa].
            dt_s: Time step [s].

        Returns:
            Updated TurboState.
        """
        p = self.params

        # --- Wastegate PI controller ---
        # Error: positive means boost is too high → open wastegate
        error = current_map_pa - target_map_pa
        self.state.integral_error += error * dt_s

        # Anti-windup: clamp integral
        max_integral = 50000.0
        self.state.integral_error = max(
            -max_integral, min(max_integral, self.state.integral_error)
        )

        # Wastegate duty cycle
        kp = 2.0   # from config
        ki = 0.5
        wastegate_cmd = kp * error / 100000.0 + ki * self.state.integral_error / 100000.0
        wastegate_pos = max(0.0, min(1.0, wastegate_cmd))

        # --- Turbine power ---
        # Turbine expansion ratio ≈ compressor PR (simplified)
        gamma = GAMMA_AIR
        gamma_ratio = (gamma - 1.0) / gamma

        # Turbine power: P_t = ṁ_t · c_p · T_3 · η_t · (1 - (1/PR_t)^((γ-1)/γ))
        # Source: Heywood (2018), §6.3
        turbine_pr = max(self.state.compressor_pr, 1.01)
        bypassed_fraction = wastegate_pos * 0.7  # wastegate can bypass up to 70%
        effective_exhaust_flow = exhaust_mass_flow_kg_s * (1.0 - bypassed_fraction)

        turbine_power = (
            effective_exhaust_flow
            * CP_AIR
            * exhaust_temp_k
            * p.turbine_efficiency
            * (1.0 - (1.0 / turbine_pr) ** gamma_ratio)
        )

        # --- Compressor power requirement ---
        # Intake air mass flow ≈ exhaust - fuel (approximately equal)
        intake_flow = exhaust_mass_flow_kg_s * 0.93  # ~7% fuel mass fraction

        # Compressor pressure ratio from current MAP and ambient
        comp_pr = max(current_map_pa / ambient_pressure_pa, 1.0)

        # Compressor power: P_c = ṁ_c · c_p · T_1 · (PR^((γ-1)/γ) - 1) / η_c
        # Source: Heywood (2018), §6.3
        compressor_power = (
            intake_flow
            * CP_AIR
            * ambient_temp_k
            * (comp_pr ** gamma_ratio - 1.0)
            / max(p.compressor_efficiency, 0.5)
        )

        # --- Shaft dynamics ---
        # τ_net = (P_t - P_c) / ω
        # dω/dt = τ_net / J
        # Source: Watson & Janota, "Turbocharging the IC Engine", 1982, Ch. 4
        shaft_speed_rad_s = self.state.shaft_speed_rpm * math.pi / 30.0
        if shaft_speed_rad_s < 100.0:
            shaft_speed_rad_s = 100.0  # Minimum speed floor

        net_torque = (turbine_power - compressor_power) / shaft_speed_rad_s
        angular_accel = net_torque / p.turbo_inertia_kg_m2
        new_speed_rad_s = shaft_speed_rad_s + angular_accel * dt_s
        new_speed_rad_s = max(100.0, min(new_speed_rad_s, 200000.0 * math.pi / 30.0))
        new_speed_rpm = new_speed_rad_s * 30.0 / math.pi

        # --- Compressor outlet conditions ---
        # T_2 = T_1 · (1 + (PR^((γ-1)/γ) - 1) / η_c)
        # Source: Heywood (2018), §6.3
        comp_outlet_temp = ambient_temp_k * (
            1.0 + (comp_pr ** gamma_ratio - 1.0) / max(p.compressor_efficiency, 0.5)
        )

        comp_outlet_pressure = ambient_pressure_pa * comp_pr

        self.state = TurboState(
            shaft_speed_rpm=new_speed_rpm,
            compressor_pr=comp_pr,
            compressor_outlet_temp_k=comp_outlet_temp,
            compressor_outlet_pressure_pa=comp_outlet_pressure,
            compressor_power_w=compressor_power,
            turbine_power_w=turbine_power,
            turbine_inlet_temp_k=exhaust_temp_k,
            wastegate_position=wastegate_pos,
            wastegate_duty_pct=wastegate_pos * 100.0,
            integral_error=self.state.integral_error,
        )

        return self.state
