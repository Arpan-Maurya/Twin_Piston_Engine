"""
Cooling System Model — Liquid cooling circuit for the Rotax 915 iS.

Models heat rejection from cylinders through the coolant circuit:
- CHT evolution based on combustion heat input vs. coolant removal
- Thermostat behavior
- Coolant temperature rise

Key equations:
    - Newton's law of cooling:  Q̇ = h·A·ΔT
    - Coolant heat balance:     ΔT = Q̇ / (ṁ · c_p)
    - Thermal inertia:          dT/dt = (T_target - T) / τ

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.l1_data.simulator.rotax_915is_params import Rotax915iSParams


@dataclass
class CoolingState:
    """Instantaneous cooling system state."""

    coolant_temp_k: float = 353.15    # 80°C
    cht_avg_k: float = 373.15        # 100°C (average across cylinders)
    heat_rejected_w: float = 0.0
    thermostat_position: float = 0.0  # 0 = closed, 1 = fully open


@dataclass
class CoolingModel:
    """Liquid cooling circuit model.

    Simplified model of the Rotax 915 iS cooling system:
    - Coolant absorbs heat from cylinder walls
    - Thermostat opens progressively between 85°C and 95°C
    - Radiator rejects heat to ambient air
    """

    params: Rotax915iSParams
    state: CoolingState = field(default_factory=CoolingState)

    # Thermal inertia time constants [s]
    coolant_tau_s: float = 30.0     # Coolant responds over ~30s
    cht_tau_s: float = 10.0         # CHT responds over ~10s

    # Heat transfer coefficient (cylinder wall → coolant) [W/K]
    wall_to_coolant_htc: float = 500.0

    def update(
        self,
        cylinder_heat_w: float,
        ambient_temp_k: float,
        rpm: float,
        dt_s: float,
    ) -> CoolingState:
        """Update cooling system state for one time step.

        Args:
            cylinder_heat_w: Total heat from all cylinders to coolant [W].
            ambient_temp_k: Ambient air temperature [K].
            rpm: Engine speed [rev/min] (affects coolant pump flow).
            dt_s: Time step [s].

        Returns:
            Updated CoolingState.
        """
        p = self.params

        # Thermostat position (linear ramp between open and full-open temps)
        thermostat_open_k = p.thermostat_open_k
        thermostat_full_k = thermostat_open_k + 10.0  # Full open at 95°C
        if self.state.coolant_temp_k <= thermostat_open_k:
            thermostat_pos = 0.0
        elif self.state.coolant_temp_k >= thermostat_full_k:
            thermostat_pos = 1.0
        else:
            thermostat_pos = (
                (self.state.coolant_temp_k - thermostat_open_k)
                / (thermostat_full_k - thermostat_open_k)
            )

        # Coolant flow rate scales with RPM (mechanical pump)
        rpm_fraction = min(rpm / p.rated_rpm, 1.5) if p.rated_rpm > 0 else 1.0
        effective_flow = p.coolant_flow_kg_s * rpm_fraction * max(thermostat_pos, 0.1)

        # Heat rejected to radiator
        # Q̇_rad = ε · ṁ · c_p · (T_coolant - T_ambient)
        # Source: basic heat exchanger effectiveness model
        radiator_effectiveness = 0.65
        q_rejected = (
            radiator_effectiveness
            * effective_flow
            * p.coolant_cp_j_kg_k
            * max(self.state.coolant_temp_k - ambient_temp_k, 0.0)
        )

        # Coolant temperature: gains heat from engine, loses to radiator
        # ṁ·c_p·dT = Q̇_engine - Q̇_radiator
        net_heat = cylinder_heat_w - q_rejected
        coolant_thermal_mass = effective_flow * p.coolant_cp_j_kg_k * self.coolant_tau_s
        if coolant_thermal_mass > 0:
            dt_coolant = net_heat * dt_s / coolant_thermal_mass
        else:
            dt_coolant = 0.0

        new_coolant_temp = self.state.coolant_temp_k + dt_coolant
        new_coolant_temp = max(ambient_temp_k, min(new_coolant_temp, 430.0))

        # Average CHT tracks towards target based on heat input
        cht_target = new_coolant_temp + cylinder_heat_w / max(self.wall_to_coolant_htc, 1.0)
        cht_alpha = min(dt_s / self.cht_tau_s, 1.0)
        new_cht = self.state.cht_avg_k + cht_alpha * (cht_target - self.state.cht_avg_k)
        new_cht = max(ambient_temp_k, min(new_cht, 500.0))

        self.state = CoolingState(
            coolant_temp_k=new_coolant_temp,
            cht_avg_k=new_cht,
            heat_rejected_w=q_rejected,
            thermostat_position=thermostat_pos,
        )

        return self.state
