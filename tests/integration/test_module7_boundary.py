"""
Architecture Boundary Test for Original Module 7 — EGT Diagnostics.

Verifies that Module 7:
    1. Operates on NormalizedSignalRecord from Module 5 and outputs DiagnosticState
    2. Does NOT import simulator internals or simulator ground truth
    3. Does NOT import ML models, Advisory, or API modules
    4. Does NOT implement Module 8+ lubrication/vibration/ML logic
    5. Leaves RawSignalRecord and NormalizedSignalRecord completely unchanged
"""

import ast
from pathlib import Path
import pytest

from src.core.provenance import Provenance
from src.core.schemas import DiagnosticState, SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l2_digital_twin.egt_diagnostics import (
    EGTDiagnosticsEngine,
    evaluate_egt_diagnostics,
)
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state


class TestModule7ArchitecturalBoundary:
    """Automated AST and boundary check for Module 7 EGT Diagnostics."""

    def test_egt_diagnostics_ast_imports(self) -> None:
        """Scan src/l2_digital_twin/egt_diagnostics.py for forbidden dependencies."""
        filepath = Path("src/l2_digital_twin/egt_diagnostics.py")
        assert filepath.exists()

        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        forbidden_modules = {
            "src.l3_ml",
            "src.l4_advisory",
            "src.l5_interface",
            "src.l1_data.simulator",
            "fastapi",
            "onnxruntime",
        }
        forbidden_names = {
            "RotaxEngineSimulator",
            "CylinderModel",
            "TurbochargerModel",
            "FaultInjector",
            "AnomalyDetector",
            "AdvisoryEngine",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(alias.name.startswith(mod) for mod in forbidden_modules), \
                        f"Forbidden import '{alias.name}' in egt_diagnostics.py"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not any(mod_name.startswith(mod) for mod in forbidden_modules), \
                    f"Forbidden import from '{mod_name}' in egt_diagnostics.py"
                for alias in node.names:
                    assert alias.name not in forbidden_names, \
                        f"Forbidden symbol '{alias.name}' in egt_diagnostics.py"

    def test_simulator_independence(self) -> None:
        """Verify that EGT Diagnostics operates purely on NormalizedSignalRecord without simulator access."""
        raw_record = RawSignalRecord(
            sequence_number=1,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=30000.0,
            egt_cyl2_hot_uv=30500.0,
            egt_cyl3_hot_uv=29800.0,
            egt_cyl4_hot_uv=30100.0,
            egt_cold_c=25.0,
            cht_hot_uv=12000.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=135.0,
            oil_p_counts=2048,
            map_counts=2048,
            adc_vref_counts=4095,
            crank_period_us=15000.0,
            fuel_pulse_hz=100.0,
            accel_counts_xyz=(10, 20, 980),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=0.98),
        )

        original_raw_dict = raw_record.model_dump()
        norm_record = convert_raw_to_engineering_state(raw_record)
        original_norm_dict = norm_record.model_dump()

        # Evaluate Module 7
        diag_state, res = evaluate_egt_diagnostics(norm_record)

        # Immutability assertions
        assert raw_record.model_dump() == original_raw_dict
        assert norm_record.model_dump() == original_norm_dict

        # Output assertions
        assert isinstance(diag_state, DiagnosticState)
        assert diag_state.provenance == Provenance.DERIVED
        assert len(diag_state.per_cylinder_egt_dev_k) == 4
        assert res.overall_status.value in ("NORMAL", "WARNING", "CRITICAL", "INVALID")
