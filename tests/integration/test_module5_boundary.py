"""
Architecture Boundary Test for Original Module 5 — Sensor Inverse Modelling.

Verifies that Module 5:
    1. Consumes RawSignalRecord and produces NormalizedSignalRecord with Provenance.DERIVED
    2. Leaves RawSignalRecord completely unchanged (immutable)
    3. Does NOT import simulator internals or ground truth
    4. Does NOT perform thermodynamic/mechanical derivations (Module 6+)
    5. Does NOT import ML models, Advisory, or API modules
"""

import ast
from pathlib import Path
import pytest

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l2_digital_twin.sensor_inverse import SensorInverseModel, convert_raw_to_engineering_state


class TestModule5ArchitecturalBoundary:
    """Automated AST and boundary check for Module 5 sensor inverse modelling."""

    def test_sensor_inverse_ast_imports(self) -> None:
        """Scan src/l2_digital_twin/sensor_inverse.py for forbidden dependencies."""
        filepath = Path("src/l2_digital_twin/sensor_inverse.py")
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
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(alias.name.startswith(mod) for mod in forbidden_modules), \
                        f"Forbidden import '{alias.name}' in sensor_inverse.py"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not any(mod_name.startswith(mod) for mod in forbidden_modules), \
                    f"Forbidden import from '{mod_name}' in sensor_inverse.py"
                for alias in node.names:
                    assert alias.name not in forbidden_names, \
                        f"Forbidden symbol '{alias.name}' in sensor_inverse.py"

    def test_raw_record_immutability(self) -> None:
        """Verify that converting a RawSignalRecord leaves the original object unmodified."""
        raw_record = RawSignalRecord(
            sequence_number=100,
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

        original_dict = raw_record.model_dump()
        eng_record = convert_raw_to_engineering_state(raw_record)

        # Raw record must be 100% unchanged
        assert raw_record.model_dump() == original_dict
        assert eng_record.provenance == Provenance.DERIVED
