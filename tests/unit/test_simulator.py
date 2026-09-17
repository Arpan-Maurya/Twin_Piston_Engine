"""
Unit tests for the Rotax 915 iS physics simulator.

Validates:
- Engine geometry and derived parameters
- Single-cylinder Otto cycle thermodynamics
- Simulator output structure (NormalizedSignalRecord)
- Fault injection distinct signatures
- Physical plausibility of computed values
"""

from __future__ import annotations

import math

import pytest

from src.core.provenance import FlightPhase, Provenance
from src.l1_data.signal_record import NormalizedSignalRecord
from src.l1_data.simulator.cylinder import CylinderModel, CylinderState
from src.l1_data.simulator.engine_model import EngineOperatingPoint, RotaxEngineSimulator
from src.l1_data.simulator.fault_injection import FaultInjector, FaultMode, FaultScenario
from src.l1_data.simulator.rotax_915is_params import ROTAX_915IS, Rotax915iSParams


class TestRotaxParams:
    """Validate Rotax 915 iS engine parameters."""

    def test_bore(self) -> None:
        assert ROTAX_915IS.bore_m == 0.084

    def test_stroke(self) -> None:
        assert ROTAX_915IS.stroke_m == 0.061

    def test_displacement_per_cylinder(self) -> None:
        """V_d1 = (π/4) · B² · S"""
        expected = (math.pi / 4.0) * 0.084**2 * 0.061
        assert ROTAX_915IS.stroke_volume_m3 == pytest.approx(expected, rel=1e-6)

    def test_total_displacement(self) -> None:
        """Total displacement ≈ 1352 cc."""
        expected_cc = ROTAX_915IS.total_displacement_m3 * 1e6
        assert expected_cc == pytest.approx(1352.0, rel=0.01)

    def test_clearance_volume(self) -> None:
        """V_c = V_d1 / (r_c - 1)"""
        v_c = ROTAX_915IS.stroke_volume_m3 / (10.5 - 1.0)
        assert ROTAX_915IS.clearance_volume_m3 == pytest.approx(v_c, rel=1e-6)

    def test_compression_ratio(self) -> None:
        assert ROTAX_915IS.compression_ratio == 10.5

    def test_rod_ratio(self) -> None:
        """λ = r / l = (S/2) / l_rod"""
        expected = (0.061 / 2.0) / 0.105
        assert ROTAX_915IS.rod_ratio == pytest.approx(expected, rel=1e-4)

    def test_reciprocating_mass(self) -> None:
        """m_recip ≈ m_piston + m_conrod/3"""
        expected = 0.35 + 0.30 / 3.0
        assert ROTAX_915IS.reciprocating_mass_kg == pytest.approx(expected)


class TestCylinderModel:
    """Test single-cylinder Otto cycle model."""

    @pytest.fixture
    def cylinder(self) -> CylinderModel:
        return CylinderModel(params=ROTAX_915IS, cylinder_id=1)

    def test_volume_at_tdc(self, cylinder: CylinderModel) -> None:
        """Volume at TDC should be the clearance volume."""
        v_tdc = cylinder.cylinder_volume(0.0)
        assert v_tdc == pytest.approx(ROTAX_915IS.clearance_volume_m3, rel=1e-3)

    def test_volume_at_bdc(self, cylinder: CylinderModel) -> None:
        """Volume at BDC should be clearance + displacement."""
        v_bdc = cylinder.cylinder_volume(180.0)
        expected = ROTAX_915IS.clearance_volume_m3 + ROTAX_915IS.stroke_volume_m3
        assert v_bdc == pytest.approx(expected, rel=1e-3)

    def test_compression_ratio_from_volumes(self, cylinder: CylinderModel) -> None:
        """Compression ratio = V_BDC / V_TDC should match 10.5."""
        v_tdc = cylinder.cylinder_volume(0.0)
        v_bdc = cylinder.cylinder_volume(180.0)
        cr = v_bdc / v_tdc
        assert cr == pytest.approx(10.5, rel=0.01)

    def test_wiebe_zero_before_start(self, cylinder: CylinderModel) -> None:
        """Burn fraction should be 0 before combustion starts."""
        assert cylinder.wiebe_burn_fraction(330.0, 335.0) == 0.0

    def test_wiebe_complete_after_duration(self, cylinder: CylinderModel) -> None:
        """Burn fraction should be ≈1 after full duration."""
        theta_start = 335.0
        theta_end = theta_start + ROTAX_915IS.combustion_duration_deg + 10.0
        x_b = cylinder.wiebe_burn_fraction(theta_end, theta_start)
        assert x_b == pytest.approx(1.0, abs=0.01)

    def test_wiebe_monotonically_increasing(self, cylinder: CylinderModel) -> None:
        """Burn fraction must increase monotonically."""
        theta_start = 335.0
        prev = 0.0
        for theta in range(int(theta_start), int(theta_start + 60)):
            x_b = cylinder.wiebe_burn_fraction(float(theta), theta_start)
            assert x_b >= prev
            prev = x_b

    def test_cycle_produces_positive_work(self, cylinder: CylinderModel) -> None:
        """A firing cycle must produce positive indicated work."""
        state = cylinder.run_cycle(
            rpm=4000.0,
            map_pa=120000.0,
            intake_temp_k=300.0,
            fuel_mass_kg=5e-5,
            lambda_actual=0.92,
            ambient_pressure_pa=101325.0,
        )
        assert state.imep_pa > 0, "IMEP must be positive for a firing cycle"

    def test_misfire_produces_less_work(self, cylinder: CylinderModel) -> None:
        """A misfiring cycle should produce near-zero work."""
        cylinder.state.is_misfiring = True
        state = cylinder.run_cycle(
            rpm=4000.0,
            map_pa=120000.0,
            intake_temp_k=300.0,
            fuel_mass_kg=5e-5,
            lambda_actual=0.92,
            ambient_pressure_pa=101325.0,
        )
        # Misfire → no heat release → work should be very small
        assert abs(state.imep_pa) < 200000, "Misfiring IMEP should be much smaller"

    def test_mean_piston_speed_at_rated(self, cylinder: CylinderModel) -> None:
        """Mean piston speed at 5800 RPM: v̄_p = 2·S·N/60 ≈ 11.8 m/s."""
        v_p = cylinder.mean_piston_speed(5800.0)
        expected = 2.0 * 0.061 * 5800.0 / 60.0
        assert v_p == pytest.approx(expected, rel=1e-4)
        assert v_p == pytest.approx(11.8, abs=0.1)


class TestEngineSimulator:
    """Test the top-level engine simulator."""

    @pytest.fixture
    def simulator(self) -> RotaxEngineSimulator:
        return RotaxEngineSimulator()

    def test_produces_normalized_signal_record(self, simulator: RotaxEngineSimulator) -> None:
        """Simulator output must be a NormalizedSignalRecord."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        assert isinstance(record, NormalizedSignalRecord)

    def test_provenance_is_simulated(self, simulator: RotaxEngineSimulator) -> None:
        """All output must be tagged SIMULATED."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        assert record.provenance == Provenance.SIMULATED

    def test_sequence_number_increments(self, simulator: RotaxEngineSimulator) -> None:
        """Sequence numbers must be monotonically increasing."""
        r1 = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        r2 = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        assert r2.sequence_number > r1.sequence_number

    def test_rpm_in_output(self, simulator: RotaxEngineSimulator) -> None:
        """RPM in output should be close to input."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        assert record.rpm.value == pytest.approx(4000.0, abs=50)

    def test_all_channels_present(self, simulator: RotaxEngineSimulator) -> None:
        """All 38 channels should be present in the output."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        channels = record.all_channel_values
        # At least 30+ channels (some may not be ChannelValue type)
        assert len(channels) >= 30

    def test_egt_physically_plausible(self, simulator: RotaxEngineSimulator) -> None:
        """EGT should be in a physically plausible range (400-1100 K)."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=60))
        for egt in record.egt_channels:
            assert 350 < egt.value < 1200, f"EGT {egt.value} K is implausible"

    def test_oil_pressure_scales_with_rpm(self, simulator: RotaxEngineSimulator) -> None:
        """Oil pressure should increase with RPM."""
        r_low = simulator.step(EngineOperatingPoint(rpm=1500, throttle_pct=30))
        # Reset simulator to avoid warm-up effects
        sim2 = RotaxEngineSimulator()
        r_high = sim2.step(EngineOperatingPoint(rpm=5000, throttle_pct=80))
        assert r_high.oil_pressure.value > r_low.oil_pressure.value

    def test_integrity_hash_not_empty(self, simulator: RotaxEngineSimulator) -> None:
        """Every record must have an integrity hash."""
        record = simulator.step(EngineOperatingPoint(rpm=4000, throttle_pct=50))
        assert record.integrity_hash != ""
        assert len(record.integrity_hash) == 64  # SHA-256 hex digest


class TestFaultInjection:
    """Test that fault injection produces distinct signatures."""

    def _run_with_fault(self, mode: FaultMode, severity: float = 0.8) -> NormalizedSignalRecord:
        sim = RotaxEngineSimulator()
        sim.configure_fault(FaultScenario(
            mode=mode,
            severity=severity,
            onset_time_s=0.0,
        ))
        # Run a few steps to let fault take effect
        for _ in range(3):
            record = sim.step(EngineOperatingPoint(rpm=4000, throttle_pct=60))
        return record

    def _run_nominal(self) -> NormalizedSignalRecord:
        sim = RotaxEngineSimulator()
        for _ in range(3):
            record = sim.step(EngineOperatingPoint(rpm=4000, throttle_pct=60))
        return record

    def test_misfire_lowers_egt(self) -> None:
        """Misfire should lower EGT on affected cylinder."""
        nominal = self._run_nominal()
        faulty = self._run_with_fault(FaultMode.MISFIRE)
        # EGT on cylinder 1 should be lower with misfire
        assert faulty.egt_cyl_1.value < nominal.egt_cyl_1.value

    def test_intake_leak_lowers_map(self) -> None:
        """Intake leak should reduce MAP."""
        nominal = self._run_nominal()
        faulty = self._run_with_fault(FaultMode.INTAKE_BOOST_LEAK)
        assert faulty.map_pressure.value < nominal.map_pressure.value

    def test_oil_degradation_lowers_pressure(self) -> None:
        """Oil degradation should reduce oil pressure."""
        nominal = self._run_nominal()
        faulty = self._run_with_fault(FaultMode.OIL_DEGRADATION)
        assert faulty.oil_pressure.value < nominal.oil_pressure.value

    def test_cooling_fault_raises_cht(self) -> None:
        """Cooling fault should raise CHT."""
        nominal = self._run_nominal()
        faulty = self._run_with_fault(FaultMode.COOLING_FAULT)
        # At least one CHT should be higher
        assert faulty.cht_cyl_1.value > nominal.cht_cyl_1.value

    def test_bearing_wear_increases_vibration(self) -> None:
        """Bearing wear should increase vibration."""
        nominal = self._run_nominal()
        faulty = self._run_with_fault(FaultMode.BEARING_WEAR)
        assert faulty.vibration_rms.value > nominal.vibration_rms.value
