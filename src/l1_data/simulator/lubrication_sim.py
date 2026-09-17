"""
Lubrication System Model — Oil circuit simulation for the Rotax 915 iS.

Models:
- Oil pressure from pump speed (crankshaft-driven)
- Oil temperature from friction heat input minus oil cooler rejection
- Viscosity vs. temperature (Vogel equation)

Key equations:
    - Vogel viscosity:   μ = A · exp(B / (T - C))
    - Heat balance:      dT/dt = (Q̇_in - Q̇_out) / (m · c_p)

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.l1_data.simulator.rotax_915is_params import Rotax915iSParams


@dataclass
class LubricationState:
    """Instantaneous lubrication system state."""

    oil_temp_k: float = 363.15       # 90°C nominal
    oil_pressure_pa: float = 400000.0  # 4 bar nominal
    viscosity_pa_s: float = 0.03     # Dynamic viscosity
    oil_flow_rate_m3_s: float = 0.0


@dataclass
class LubricationModel:
    """Oil circuit simulation for the Rotax 915 iS.

    The oil pump is mechanically driven off the crankshaft, so oil
    pressure scales with RPM. Oil temperature is driven by friction
    heat input and oil cooler heat rejection.
    """

    params: Rotax915iSParams
    state: LubricationState = field(default_factory=LubricationState)

    # Oil system parameters
    oil_mass_kg: float = 3.5           # Oil charge [kg]
    oil_cp_j_kg_k: float = 2000.0     # Specific heat of engine oil [J/(kg·K)]
    cooler_effectiveness: float = 0.5   # Oil cooler effectiveness
    oil_cooler_flow_kg_s: float = 0.2  # Oil flow through cooler [kg/s]

    # Vogel equation coefficients
    # Source: Stachowiak & Batchelor, "Engineering Tribology", 4th ed., 2014
    # μ = A · exp(B / (T[K] - C))
    vogel_a: float = 0.0002
    vogel_b: float = 1200.0
    vogel_c: float = 140.0

    def viscosity(self, temp_k: float) -> float:
        """Oil dynamic viscosity using Vogel equation [Pa·s].

        Source: Stachowiak & Batchelor (2014), "Engineering Tribology", §2.4
            μ = A · exp(B / (T - C))

        where T is in Kelvin and A, B, C are oil-specific constants.

        Args:
            temp_k: Oil temperature [K].

        Returns:
            Dynamic viscosity [Pa·s].
        """
        denom = temp_k - self.vogel_c
        if denom <= 0:
            return 1.0  # Very cold — high viscosity cap
        return self.vogel_a * math.exp(self.vogel_b / denom)

    def update(
        self,
        rpm: float,
        friction_power_w: float,
        ambient_temp_k: float,
        dt_s: float,
    ) -> LubricationState:
        """Update lubrication system state for one time step.

        Args:
            rpm: Engine speed [rev/min].
            friction_power_w: Total engine friction power loss [W].
            ambient_temp_k: Ambient temperature [K] (for oil cooler).
            dt_s: Time step [s].

        Returns:
            Updated LubricationState.
        """
        p = self.params

        # Oil pressure: mechanical pump driven by crankshaft
        # Pressure scales roughly linearly with RPM up to relief valve
        if p.rated_rpm > 0:
            rpm_ratio = rpm / p.rated_rpm
        else:
            rpm_ratio = 0.0

        # Relief valve caps pressure at max_oil_pressure
        raw_pressure = p.nominal_oil_pressure_pa * rpm_ratio * 1.2
        oil_pressure = min(raw_pressure, 700000.0)  # 7 bar relief
        oil_pressure = max(oil_pressure, 50000.0)   # Minimum from pump

        # Heat input: friction power dissipated into oil
        # Approximately 60% of friction power heats the oil
        q_friction_to_oil = friction_power_w * 0.6

        # Heat rejection through oil cooler
        # Q̇ = ε · ṁ · c_p · (T_oil - T_ambient)
        q_cooler = (
            self.cooler_effectiveness
            * self.oil_cooler_flow_kg_s
            * self.oil_cp_j_kg_k
            * max(self.state.oil_temp_k - ambient_temp_k, 0.0)
        )

        # Temperature update: dT = (Q̇_in - Q̇_out) · dt / (m · c_p)
        net_heat = q_friction_to_oil - q_cooler
        thermal_mass = self.oil_mass_kg * self.oil_cp_j_kg_k
        if thermal_mass > 0:
            dt_oil = net_heat * dt_s / thermal_mass
        else:
            dt_oil = 0.0

        new_oil_temp = self.state.oil_temp_k + dt_oil
        new_oil_temp = max(ambient_temp_k, min(new_oil_temp, 430.0))

        # Update viscosity
        new_viscosity = self.viscosity(new_oil_temp)

        self.state = LubricationState(
            oil_temp_k=new_oil_temp,
            oil_pressure_pa=oil_pressure,
            viscosity_pa_s=new_viscosity,
            oil_flow_rate_m3_s=rpm_ratio * 5e-4,  # ~0.5 L/s at rated
        )

        return self.state
