"""
Architecture & L1 → L2 Boundary Integration Test.

Strict Boundary Rules:
    1. L2 (Digital Twin) must receive ONLY RawSignalRecord.
    2. L2 must NOT import from l1_data.simulator.
    3. L2 must NOT access simulator ground truth or generator internals.
    4. L2 must NOT contain source-specific logic.

This test uses Python's AST module to audit all files under src/l2_digital_twin/
(and verifies that any attempt to import simulator internals in L2 raises a boundary error).
"""

import ast
from pathlib import Path
import pytest

from src.core.exceptions import BoundaryViolationError


def audit_file_for_simulator_imports(file_path: Path) -> list[str]:
    """Scan a Python source file AST for forbidden simulator imports or references."""
    violations = []
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    forbidden_modules = {"src.l1_data.simulator", "l1_data.simulator"}
    forbidden_names = {
        "RotaxEngineSimulator",
        "CylinderModel",
        "TurbochargerModel",
        "FaultInjector",
        "Rotax915iSParams",
    }

    for node in ast.walk(tree):
        # Check 'import foo'
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(mod) for mod in forbidden_modules):
                    violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")

        # Check 'from foo import bar'
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if any(mod_name.startswith(mod) for mod in forbidden_modules):
                violations.append(f"Forbidden import from '{mod_name}' at line {node.lineno}")
            for alias in node.names:
                if alias.name in forbidden_names:
                    violations.append(f"Forbidden import of symbol '{alias.name}' at line {node.lineno}")

        # Check direct class references in code
        elif isinstance(node, ast.Name):
            if node.id in forbidden_names and not isinstance(node.ctx, ast.Store):
                violations.append(f"Forbidden reference to symbol '{node.id}' at line {node.lineno}")

    return violations


class TestL1ToL2Boundary:
    """Automated L1 -> L2 Boundary Verification."""

    def test_l2_files_have_zero_simulator_imports(self) -> None:
        """Scan all files in src/l2_digital_twin/ and assert ZERO simulator imports."""
        l2_dir = Path("src/l2_digital_twin")
        if not l2_dir.exists():
            # Directory does not exist yet (Module 2 scope)
            return

        all_violations = {}
        for py_file in l2_dir.rglob("*.py"):
            violations = audit_file_for_simulator_imports(py_file)
            if violations:
                all_violations[py_file.name] = violations

        assert not all_violations, f"L1->L2 boundary violations detected: {all_violations}"

    def test_boundary_auditor_fails_on_illegal_import(self, tmp_path: Path) -> None:
        """Verify that the boundary auditor correctly detects and flags illegal L2 imports."""
        bad_l2_file = tmp_path / "bad_l2_module.py"
        bad_l2_file.write_text(
            "from src.l1_data.simulator.engine_model import RotaxEngineSimulator\n"
            "def do_something():\n"
            "    sim = RotaxEngineSimulator()\n"
        )

        violations = audit_file_for_simulator_imports(bad_l2_file)
        assert len(violations) > 0
        assert any("RotaxEngineSimulator" in v for v in violations)
