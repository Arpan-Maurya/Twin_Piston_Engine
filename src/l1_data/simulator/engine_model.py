"""
Rotax 915 iS Engine Simulator — Top-level orchestrator.

Brings together all sub-models (cylinders, turbocharger, cooling,
lubrication, sensors, fault injection) into a unified engine simulation.

The simulator outputs ONLY NormalizedSignalRecord instances.
Its internal state (ground truth, cylinder pressures, etc.) is never
exposed to L2 or any downstream layer.

ARCHITECTURAL BOUNDARY:
    This module and everything in l1_data.simulator is internal to L1.
    L2 must NEVER import from this package.

This module is INTERNAL to the simulator. L2 must NOT import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.constants import GAMMA_AIR, R_AIR
from src.core.provenance import FlightPhase
from src.core.units import isa_pressure, isa_temperature
from src.l1_data.signal_record import NormalizedSignalRecord
from src.l1_data.simulator.cooling import CoolingModel
from src.l1_data.simulator.cylinder import CylinderModel
from src.l1_data.simulator.fault_injection import FaultInjector, FaultScenario
from src.l1_data.simulator.lubrication_sim import LubricationModel
from src.l1_data.simulator.rotax_915is_params import ROTAX_915IS, Rotax915iSParams
from src.l1_data.simulator.sensors import SensorModel, SensorNoiseConfig
from src.l1_data.simulator.turbocharger import TurbochargerModel


@dataclass
class EngineOperatingPoint:
    """Input operating conditions for the simulator."""

    rpm: float = 4000.0
    throttle_pct: float = 50.0
    altitude_m: float = 1000.0
    ambient_temp_k: float | None = None   # None → use ISA
    ambient_pressure_pa: float | None = None  # None → use ISA
    flight_phase: FlightPhase = FlightPhase.CRUISE
    engine_hours: float = 0.0


@dataclass
class RotaxEngineSimulator:
    """Top-level Rotax 915 iS engine simulator.

    Orchestrates four cylinder models, turbocharger, cooling, lubrication,
    sensor noise, and fault injection into a unified simulation.

    Usage:
        sim = RotaxEngineSimulator()
        record = sim.step(EngineOperatingPoint(rpm=4000, throttle_pct=60))
        # record is a NormalizedSignalRecord — same format as real hardware

    The simulator NEVER exposes internal state outside of L1.
    """

    params: Rotax915iSParams = field(default_factory=lambda: ROTAX_915IS)

    # Sub-models
    cylinders: list[CylinderModel] = field(default_factory=list)
    turbo: TurbochargerModel = field(default_factory=lambda: TurbochargerModel(ROTAX_915IS))
    cooling: CoolingModel = field(default_factory=lambda: CoolingModel(ROTAX_915IS))
    lubrication: LubricationModel = field(default_factory=lambda: LubricationModel(ROTAX_915IS))
    sensors: SensorModel = field(default_factory=SensorModel)
    fault_injector: FaultInjector = field(default_factory=FaultInjector)

    # State
    sequence_number: int = 0
    sim_time_s: float = 0.0
    dt_s: float = 1.0  # Time step per output (at 1 Hz sample rate)

    def __post_init__(self) -> None:
        """Initialize cylinder models if not provided."""
        if not self.cylinders:
            self.cylinders = [
                CylinderModel(params=self.params, cylinder_id=i)
                for i in range(1, self.params.num_cylinders + 1)
            ]

    def configure_fault(self, scenario: FaultScenario) -> None:
        """Configure a fault injection scenario.

        Args:
            scenario: Fault scenario to inject.
        """
        self.fault_injector = FaultInjector(scenario=scenario)

    def step(self, operating_point: EngineOperatingPoint) -> NormalizedSignalRecord:
        """Run one simulation step and return a NormalizedSignalRecord.

        This is the ONLY public output of the simulator.

        Args:
            operating_point: Current operating conditions.

        Returns:
            NormalizedSignalRecord ready for L2 consumption.
        """
        p = self.params

        # Resolve ambient conditions (ISA model if not specified)
        altitude = operating_point.altitude_m
        ambient_temp = operating_point.ambient_temp_k or isa_temperature(altitude)
        ambient_pressure = operating_point.ambient_pressure_pa or isa_pressure(altitude)

        rpm = operating_point.rpm
        throttle = operating_point.throttle_pct

        # --- Turbocharger ---
        # Target MAP from throttle position
        # At 100% throttle → wastegate target; at 0% → ambient
        target_map = ambient_pressure + (p.wastegate_target_pa - ambient_pressure) * throttle / 100.0

        # Estimate exhaust conditions from previous cycle
        mean_egt = sum(c.state.egt_k for c in self.cylinders) / len(self.cylinders)

        # Estimate mass flow from RPM and MAP
        # ṁ_air ≈ η_vol · ρ_intake · V_d · N / (2 · 60)  [4-stroke]
        eta_vol = 0.85  # Assumed volumetric efficiency
        rho_intake = self.turbo.state.compressor_outlet_pressure_pa / (
            R_AIR * max(self.turbo.state.compressor_outlet_temp_k, 250.0)
        )
        air_mass_flow = (
            eta_vol * rho_intake * p.total_displacement_m3 * rpm / (2.0 * 60.0)
        )
        fuel_mass_flow = air_mass_flow / (p.stoichiometric_afr * p.target_lambda)

        # Exhaust mass flow ≈ air + fuel
        exhaust_flow = air_mass_flow + fuel_mass_flow

        self.turbo.update(
            ambient_pressure_pa=ambient_pressure,
            ambient_temp_k=ambient_temp,
            exhaust_temp_k=mean_egt,
            exhaust_mass_flow_kg_s=exhaust_flow,
            target_map_pa=target_map,
            current_map_pa=self.turbo.state.compressor_outlet_pressure_pa,
            dt_s=self.dt_s,
        )

        # Actual MAP (from turbo outlet)
        actual_map = self.turbo.state.compressor_outlet_pressure_pa
        intake_temp = self.turbo.state.compressor_outlet_temp_k

        # --- Cylinders ---
        # Fuel per cylinder per cycle
        # At 1 Hz output and rpm/120 cycles/s per cylinder (4-stroke):
        cycles_per_second = rpm / 120.0  # For a 4-stroke engine
        if cycles_per_second > 0:
            fuel_per_cylinder_per_cycle = fuel_mass_flow / (p.num_cylinders * cycles_per_second)
        else:
            fuel_per_cylinder_per_cycle = 0.0

        total_heat = 0.0
        total_work = 0.0

        for cyl in self.cylinders:
            cyl_state = cyl.run_cycle(
                rpm=rpm,
                map_pa=actual_map,
                intake_temp_k=intake_temp,
                fuel_mass_kg=fuel_per_cylinder_per_cycle,
                lambda_actual=p.target_lambda,
                ambient_pressure_pa=ambient_pressure,
            )
            total_heat += cyl_state.heat_released_j * cycles_per_second
            total_work += cyl_state.work_j

        # --- Friction (Chen-Flynn model) ---
        # FMEP = C0 + C1·p_max + C2·v̄_p + C3·v̄_p²
        # Source: Chen & Flynn (1965), SAE 650733
        mean_piston_speed = 2.0 * p.stroke_m * rpm / 60.0
        peak_pressure = max(c.state.peak_pressure_pa for c in self.cylinders)
        fmep = (
            p.cf_c0_pa
            + p.cf_c1 * peak_pressure
            + p.cf_c2_pa_s_m * mean_piston_speed
            + p.cf_c3_pa_s2_m2 * mean_piston_speed ** 2
        )
        friction_power = fmep * p.total_displacement_m3 * rpm / (2.0 * 60.0)

        # --- Cooling ---
        # Heat to coolant ≈ 30% of total heat released
        heat_to_coolant = total_heat * 0.30
        self.cooling.update(
            cylinder_heat_w=heat_to_coolant,
            ambient_temp_k=ambient_temp,
            rpm=rpm,
            dt_s=self.dt_s,
        )

        # --- Lubrication ---
        self.lubrication.update(
            rpm=rpm,
            friction_power_w=friction_power,
            ambient_temp_k=ambient_temp,
            dt_s=self.dt_s,
        )

        # --- Build ground truth ---
        # Vibration: base level scales with RPM² (rotational imbalance)
        vib_base = 1.5 + (rpm / p.rated_rpm) ** 2 * 3.0

        ground_truth: dict = {
            "rpm": rpm,
            "map_pressure": actual_map,
            "throttle_position": throttle,
            "egt_cyl_1": self.cylinders[0].state.egt_k,
            "egt_cyl_2": self.cylinders[1].state.egt_k,
            "egt_cyl_3": self.cylinders[2].state.egt_k,
            "egt_cyl_4": self.cylinders[3].state.egt_k,
            "cht_cyl_1": self.cylinders[0].state.cht_k,
            "cht_cyl_2": self.cylinders[1].state.cht_k,
            "cht_cyl_3": self.cylinders[2].state.cht_k,
            "cht_cyl_4": self.cylinders[3].state.cht_k,
            "oil_temp": self.lubrication.state.oil_temp_k,
            "oil_pressure": self.lubrication.state.oil_pressure_pa,
            "coolant_temp": self.cooling.state.coolant_temp_k,
            "fuel_flow": fuel_mass_flow,
            "fuel_pressure": 350000.0,  # Nominal fuel pressure ~3.5 bar
            "intake_air_temp": intake_temp,
            "ambient_pressure": ambient_pressure,
            "ambient_temp": ambient_temp,
            "voltage": 13.8,   # Nominal electrical
            "current": 15.0 + rpm / 1000.0,
            "vibration_x": vib_base * 0.7,
            "vibration_y": vib_base * 0.6,
            "vibration_z": vib_base * 1.0,
            "propeller_speed": rpm * 0.414,  # Gear reduction ratio
            "boost_pressure": actual_map,
            "wastegate_duty": self.turbo.state.wastegate_duty_pct,
            "lambda_sensor": p.target_lambda,
            "ignition_timing_cyl_1": self.cylinders[0].state.ignition_timing_deg,
            "ignition_timing_cyl_2": self.cylinders[1].state.ignition_timing_deg,
            "ignition_timing_cyl_3": self.cylinders[2].state.ignition_timing_deg,
            "ignition_timing_cyl_4": self.cylinders[3].state.ignition_timing_deg,
            "altitude": altitude,
            "flight_phase": operating_point.flight_phase,
            "engine_hours": operating_point.engine_hours,
        }

        # --- Apply fault injection ---
        ground_truth = self.fault_injector.apply(ground_truth, self.sim_time_s)

        # --- Convert to sensor readings ---
        record = self.sensors.generate_record(
            ground_truth=ground_truth,
            sequence_number=self.sequence_number,
            dt_s=self.dt_s,
        )

        # Update state
        self.sequence_number += 1
        self.sim_time_s += self.dt_s

        return record
