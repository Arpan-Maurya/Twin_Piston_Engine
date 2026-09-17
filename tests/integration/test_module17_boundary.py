"""
Integration and Architecture Boundary Tests for Original Module 17.

Verifies:
1. Strict Ground-Truth Boundary (AST audit ensuring no L2/L3 imports).
2. Ground Truth isolation (SimulationGroundTruth never enters RawSignalRecord).
3. RawSignalRecord emitted by Module 17 is 100% compliant with L1 telemetry pipeline.
"""

import ast
from pathlib import Path
import pytest

from src.core.provenance import Provenance
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l1_data.simulator.forward_simulator import (
    ScenarioRunner,
    SimulationGroundTruth,
)


class TestModule17ArchitectureBoundary:
    """Architectural boundary enforcement tests."""

    def test_zero_l2_l3_imports_in_simulator(self):
        """Verify that Module 17 simulator contains zero imports of L2 or L3 packages."""
        simulator_dir = Path("src/l1_data/simulator")

        forbidden_prefixes = (
            "src.l2_digital_twin",
            "src.l3_supervision",
            "src.l2",
            "src.l3",
        )

        for py_file in simulator_dir.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in forbidden_prefixes:
                            assert not alias.name.startswith(forbidden), (
                                f"Forbidden import '{alias.name}' in {py_file}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for forbidden in forbidden_prefixes:
                            assert not node.module.startswith(forbidden), (
                                f"Forbidden import from '{node.module}' in {py_file}"
                            )

    def test_ground_truth_isolation(self):
        """Verify ground truth is strictly separated from RawSignalRecord."""
        runner = ScenarioRunner(seed=42)
        records, ground_truths, _ = runner.run_scenario(duration_s=2.0, dt_s=1.0)

        raw_rec = records[0]
        gt = ground_truths[0]

        # Verify RawSignalRecord field set does NOT contain ground truth fields
        raw_dict = raw_rec.model_dump()

        assert "true_power_kw" not in raw_dict
        assert "power_kw" not in raw_dict
        assert "combustion_efficiency" not in raw_dict
        assert "active_fault" not in raw_dict
        assert "fault_severity" not in raw_dict

        # Ground truth has fields not present in RawSignalRecord
        assert hasattr(gt, "power_kw")
        assert hasattr(gt, "torque_nm")

    def test_l1_validator_accepts_simulated_record(self):
        """Verify that L1 TelemetryValidator accepts simulated RawSignalRecords."""
        runner = ScenarioRunner(seed=123)
        records, _, _ = runner.run_scenario(duration_s=1.0, dt_s=1.0)

        signer = PacketSigner()
        validator = TelemetryValidator(signer=signer)
        sim_rec = records[0]

        assert sim_rec.source_type == Provenance.SIMULATED

        # Sign record and validate with telemetry validator
        sig = signer.sign_record(sim_rec)
        val_result = validator.validate_packet(sim_rec, signature=sig)
        assert val_result.accepted is True
