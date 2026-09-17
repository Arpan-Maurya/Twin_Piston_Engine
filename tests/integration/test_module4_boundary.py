"""
Architecture Boundary Test for Original Module 4 — Raw Telemetry Persistence.

Verifies that Module 4 persistence:
    1. Accepts RawSignalRecord
    2. Stores RawSignalRecord
    3. Returns RawSignalRecord
    4. Does NOT require simulator internals or access ground truth
    5. Does NOT perform engineering-unit conversion
    6. Does NOT import L2 digital twin implementation
    7. Does NOT introduce ML or REST API dependencies
"""

import ast
from pathlib import Path
import pytest

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_repository import (
    InMemoryRawTelemetryRepository,
    RawTelemetryRepositoryProtocol,
    SQLiteRawTelemetryRepository,
)
from src.l1_data.raw_signal_record import RawSignalRecord


class TestModule4ArchitecturalBoundary:
    """Automated AST and boundary check for Module 4 persistence."""

    def test_persistence_module_ast_imports(self) -> None:
        """Scan src/l1_data/raw_repository.py for forbidden dependencies."""
        filepath = Path("src/l1_data/raw_repository.py")
        assert filepath.exists()

        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        forbidden_modules = {
            "src.l2_digital_twin",
            "src.l3_ml",
            "src.l4_advisory",
            "src.l5_interface",
            "src.l1_data.simulator",
            "fastapi",
            "onnxruntime",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(alias.name.startswith(mod) for mod in forbidden_modules), \
                        f"Forbidden import '{alias.name}' in raw_repository.py"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert not any(mod_name.startswith(mod) for mod in forbidden_modules), \
                    f"Forbidden import from '{mod_name}' in raw_repository.py"

    def test_repository_operates_purely_on_raw_signal_record(self) -> None:
        """Verify repository input and output contracts are strictly RawSignalRecord."""
        repo: RawTelemetryRepositoryProtocol = InMemoryRawTelemetryRepository()

        raw_in = RawSignalRecord(
            sequence_number=50,
            source_type=Provenance.REAL,
            egt_cyl1_hot_uv=31000.0,
            egt_cyl2_hot_uv=31100.0,
            egt_cyl3_hot_uv=30900.0,
            egt_cyl4_hot_uv=31050.0,
            egt_cold_c=25.0,
            cht_hot_uv=12500.0,
            cht_cold_c=25.0,
            oil_rtd_ohms=136.0,
            oil_p_counts=2100,
            map_counts=2048,
            adc_vref_counts=4095,
            crank_period_us=14500.0,
            fuel_pulse_hz=105.0,
            accel_counts_xyz=(0, 0, 1000),
            ambient_temp_c=20.0,
            ambient_press_pa=101325.0,
            signal_quality=SignalQuality(score=1.0),
        )

        assert repo.save(raw_in) is True

        raw_out = repo.get_by_sequence(50)
        assert raw_out is not None
        assert isinstance(raw_out, RawSignalRecord)
        assert type(raw_out) is RawSignalRecord

        # Ensure no derived quantities (like RPM or Pa or K) were injected into raw_out
        assert not hasattr(raw_out, "rpm")
        assert not hasattr(raw_out, "bmep")
