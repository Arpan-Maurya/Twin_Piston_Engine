"""
Single-Cylinder Otto Cycle Model.

Simulates one cylinder of the Rotax 915 iS through a complete four-stroke
Otto cycle (intake → compression → combustion/expansion → exhaust).

Key equations and their sources:
    - Cylinder volume vs. crank angle: Heywood (2018), §2.2
    - Wiebe burn fraction:             Heywood (2018), §9.2, Eq. 9.29
    - First-law heat release:          Heywood (2018), §9.2
    - Woschni heat transfer:           Woschni (1967), SAE 670931
    - IMEP calculation:                Heywood (2018), §2.8

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.core.constants import GAMMA_AIR, R_AIR
from src.l1_data.simulator.rotax_915is_params import Rotax915iSParams


@dataclass
class CylinderState:
    """Instantaneous state of a single cylinder."""

    # Thermodynamic state
    pressure_pa: float = 101325.0
    temperature_k: float = 300.0
    volume_m3: float = 0.0

    # Combustion
    burn_fraction: float = 0.0
    heat_released_j: float = 0.0
    is_misfiring: bool = False

    # Temperatures (cycle-averaged or peak)
    egt_k: float = 700.0       # Exhaust gas temperature
    cht_k: float = 373.15      # Cylinder head temperature
    peak_pressure_pa: float = 0.0

    # Work output
    imep_pa: float = 0.0       # Indicated mean effective pressure
    work_j: float = 0.0        # Net indicated work per cycle

    # Ignition
    ignition_timing_deg: float = 25.0


@dataclass
class CylinderModel:
    """Single-cylinder thermodynamic model for a four-stroke SI engine.

    Models the complete Otto cycle:
        0–180°: Intake (assumed at MAP)
        180–360°: Compression (polytropic)
        Combustion (Wiebe burn around TDC)
        360–540°: Expansion (polytropic)
        540–720°: Exhaust (assumed at exhaust pressure)

    All crank angles in degrees, 0° = TDC firing.
    """

    params: Rotax915iSParams
    cylinder_id: int = 1
    state: CylinderState = field(default_factory=CylinderState)

    def cylinder_volume(self, theta_deg: float) -> float:
        """Instantaneous cylinder volume at crank angle θ [m³].

        Source: Heywood (2018), §2.2, Eq. 2.4
            V(θ) = V_c + (A_p · r / 2) · (1 - cos θ + (1/λ)(1 - √(1 - λ²sin²θ)))

        where:
            V_c = clearance volume
            A_p = piston area
            r   = crank radius
            λ   = rod ratio (r / l)
            θ   = crank angle from TDC

        Args:
            theta_deg: Crank angle from TDC [degrees].

        Returns:
            Cylinder volume [m³].
        """
        theta = math.radians(theta_deg)
        r = self.params.crank_radius_m
        lam = self.params.rod_ratio
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Slider-crank displacement
        displacement = r * (
            1.0 - cos_theta + (1.0 / lam) * (1.0 - math.sqrt(1.0 - lam**2 * sin_theta**2))
        )

        return self.params.clearance_volume_m3 + self.params.bore_area_m2 * displacement

    def wiebe_burn_fraction(self, theta_deg: float, theta_start_deg: float) -> float:
        """Mass fraction burned using the Wiebe function.

        Source: Heywood (2018), §9.2, Eq. 9.29
            x_b(θ) = 1 - exp(-a · ((θ - θ₀) / Δθ)^(m+1))

        where:
            a  = 6.908 (99.9% burn)
            m  = 2 (form factor)
            θ₀ = start of combustion
            Δθ = combustion duration

        Args:
            theta_deg: Current crank angle [degrees from TDC].
            theta_start_deg: Start of combustion [degrees from TDC].

        Returns:
            Mass fraction burned [0, 1].
        """
        if theta_deg <= theta_start_deg:
            return 0.0

        progress = (theta_deg - theta_start_deg) / self.params.combustion_duration_deg

        if progress >= 1.0:
            return 1.0

        a = self.params.wiebe_a
        m = self.params.wiebe_m

        return 1.0 - math.exp(-a * progress ** (m + 1.0))

    def run_cycle(
        self,
        rpm: float,
        map_pa: float,
        intake_temp_k: float,
        fuel_mass_kg: float,
        lambda_actual: float,
        ambient_pressure_pa: float,
        ignition_advance_deg: float | None = None,
    ) -> CylinderState:
        """Simulate one complete four-stroke cycle (720° of crank rotation).

        Uses a simplified crank-angle-resolved model:
        - Compression and expansion are modelled as polytropic processes
          with Wiebe-function-based heat addition during combustion.
        - Heat transfer via Woschni correlation reduces gas temperature.
        - IMEP is computed by integrating p·dV over the cycle.

        Args:
            rpm: Engine speed [rev/min].
            map_pa: Manifold absolute pressure at intake [Pa].
            intake_temp_k: Intake air temperature [K].
            fuel_mass_kg: Fuel mass per cylinder per cycle [kg].
            lambda_actual: Actual air-fuel equivalence ratio.
            ambient_pressure_pa: Ambient (exhaust back) pressure [Pa].
            ignition_advance_deg: Ignition timing [deg BTDC], or None for default.

        Returns:
            Updated CylinderState with cycle results.
        """
        p = self.params
        ign_timing = ignition_advance_deg or p.ignition_timing_btdc_deg

        # Combustion start angle: TDC (360°) minus ignition advance
        theta_ign = 360.0 - ign_timing

        # Total charge energy
        if self.state.is_misfiring:
            q_total = 0.0  # Misfire: no combustion
        else:
            # Combustion efficiency ~95% for SI engines
            # Source: Heywood (2018), §4.6
            combustion_efficiency = 0.95
            q_total = fuel_mass_kg * p.lhv_j_per_kg * combustion_efficiency

        # Crank-angle resolution (2° steps for balance of speed and accuracy)
        dtheta_deg = 2.0
        n_steps = int(720.0 / dtheta_deg)

        # Initial conditions at BDC before compression (θ = 180°)
        # Cylinder filled at MAP with intake temperature
        v_bdc = self.cylinder_volume(180.0)
        p_gas = map_pa
        t_gas = intake_temp_k

        # Gamma for the working fluid (air-fuel mixture)
        gamma = GAMMA_AIR  # Simplified: constant γ

        # Work accumulator for IMEP
        work_total = 0.0
        peak_pressure = 0.0
        peak_temperature = 0.0

        # Step through 720° of crank rotation
        # We model compression → combustion → expansion (180° → 540°)
        # Intake (0–180°) and exhaust (540–720°) are simplified
        for i in range(n_steps):
            theta = i * dtheta_deg

            v_current = self.cylinder_volume(theta)
            v_next = self.cylinder_volume(theta + dtheta_deg)
            dv = v_next - v_current

            if 0.0 <= theta < 180.0:
                # Intake stroke — constant pressure at MAP
                p_gas = map_pa
                t_gas = intake_temp_k
            elif 180.0 <= theta < 540.0:
                # Compression → Combustion → Expansion

                # Polytropic compression/expansion
                if abs(dv) > 1e-15 and v_next > 0:
                    # p·V^γ = const (isentropic approximation)
                    p_new = p_gas * (v_current / v_next) ** gamma

                    # Add heat release during combustion window
                    if theta_ign <= theta < (theta_ign + p.combustion_duration_deg):
                        x_b_current = self.wiebe_burn_fraction(theta, theta_ign)
                        x_b_next = self.wiebe_burn_fraction(theta + dtheta_deg, theta_ign)
                        dq = q_total * (x_b_next - x_b_current)

                        # First law: dQ = m·cv·dT + p·dV
                        # For constant volume heat addition approximation:
                        # Source: Heywood (2018), §9.2
                        v_avg = (v_current + v_next) / 2.0
                        if v_avg > 0:
                            p_new += (gamma - 1.0) * dq / v_avg

                    p_gas = max(p_new, 1000.0)  # Floor at 1 kPa

                # Update temperature via ideal gas law
                # Source: Heywood (2018), §3.3, p·V = m·R·T
                air_mass = map_pa * v_bdc / (R_AIR * intake_temp_k)
                total_mass = air_mass + fuel_mass_kg
                t_gas = p_gas * v_next / (total_mass * R_AIR) if total_mass > 0 else t_gas

                # Woschni heat transfer loss (simplified)
                # Source: Woschni (1967), SAE 670931
                # h = C1 · B^(-0.2) · p^0.8 · T^(-0.55) · w^0.8
                mean_piston_speed = 2.0 * p.stroke_m * rpm / 60.0
                if t_gas > 0:
                    h_woschni = (
                        p.woschni_c1
                        * p.bore_m ** (-0.2)
                        * (p_gas / 1e5) ** 0.8  # pressure in bar for correlation
                        * t_gas ** (-0.55)
                        * max(mean_piston_speed, 1.0) ** 0.8
                    )
                    # Heat transfer: dQ_ht = h · A · (T_gas - T_wall) · dt
                    # A ≈ bore area (simplified surface area)
                    # dt = dθ / (6·N) for dθ in degrees and N in RPM
                    dt = dtheta_deg / (6.0 * rpm) if rpm > 0 else 0.0
                    dq_ht = h_woschni * p.bore_area_m2 * (t_gas - p.wall_temp_k) * dt
                    # Reduce gas temperature by heat loss
                    if total_mass > 0 and t_gas > p.wall_temp_k:
                        t_gas -= dq_ht / (total_mass * R_AIR / (gamma - 1.0))
                        t_gas = max(t_gas, p.wall_temp_k)

            elif 540.0 <= theta < 720.0:
                # Exhaust stroke — pressure slightly above ambient
                exhaust_back_pressure = ambient_pressure_pa * 1.05
                p_gas = exhaust_back_pressure

            # Accumulate work: W = ∫ p·dV
            work_total += p_gas * dv

            # Track peak values
            peak_pressure = max(peak_pressure, p_gas)
            peak_temperature = max(peak_temperature, t_gas)

        # Compute IMEP
        # Source: Heywood (2018), §2.8
        # IMEP = W_cycle / V_d
        v_displaced = p.stroke_volume_m3
        imep = work_total / v_displaced if v_displaced > 0 else 0.0

        # Exhaust gas temperature (Heywood §4.9: expansion & blowdown cooling)
        # T_egt ≈ T_peak / (CR^(γ-1)) * 0.75 ≈ 0.37 * T_peak
        egt = peak_temperature * 0.37 if not self.state.is_misfiring else intake_temp_k + 50.0

        # CHT update (simple thermal inertia model)
        # CHT responds slowly — weighted average of previous and heat input
        heat_input_fraction = q_total / (p.lhv_j_per_kg * 0.01) if p.lhv_j_per_kg > 0 else 0
        cht_target = p.nominal_cht_k + heat_input_fraction * 20.0
        cht_alpha = 0.05  # Slow thermal response
        new_cht = self.state.cht_k + cht_alpha * (cht_target - self.state.cht_k)

        self.state = CylinderState(
            pressure_pa=p_gas,
            temperature_k=t_gas,
            volume_m3=self.cylinder_volume(0.0),
            burn_fraction=1.0 if not self.state.is_misfiring else 0.0,
            heat_released_j=q_total,
            is_misfiring=self.state.is_misfiring,
            egt_k=min(max(egt, 400.0), 1050.0),
            cht_k=min(max(new_cht, 300.0), 500.0),
            peak_pressure_pa=peak_pressure,
            imep_pa=imep,
            work_j=work_total,
            ignition_timing_deg=ign_timing,
        )

        return self.state

    def mean_piston_speed(self, rpm: float) -> float:
        """Mean piston speed [m/s].

        Source: Heywood (2018), §2.2
            v̄_p = 2 · S · N / 60

        Args:
            rpm: Engine speed [rev/min].

        Returns:
            Mean piston speed [m/s].
        """
        return 2.0 * self.params.stroke_m * rpm / 60.0

    def gas_force(self, pressure_pa: float) -> float:
        """Gas force on piston [N].

        F_gas = p · A_piston

        Source: Heywood (2018), §2.3

        Args:
            pressure_pa: In-cylinder pressure [Pa].

        Returns:
            Gas force [N].
        """
        return pressure_pa * self.params.bore_area_m2

    def inertia_force(self, rpm: float, theta_deg: float) -> float:
        """Reciprocating inertia force [N].

        Source: Heywood (2018), §2.3; Taylor (1985), Ch. 8
            F_inertia = m_recip · r · ω² · (cos θ + λ · cos 2θ)

        Args:
            rpm: Engine speed [rev/min].
            theta_deg: Crank angle from TDC [degrees].

        Returns:
            Inertia force [N] (positive = toward crankshaft).
        """
        omega = rpm * math.pi / 30.0
        theta = math.radians(theta_deg)
        r = self.params.crank_radius_m
        lam = self.params.rod_ratio

        return self.params.reciprocating_mass_kg * r * omega**2 * (
            math.cos(theta) + lam * math.cos(2.0 * theta)
        )
