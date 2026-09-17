"""
L2 Digital Twin — Physics & Inverse Modelling Layer.

Module 5: Sensor Inverse Modelling & Engineering-Unit Conversion.
Converts RawSignalRecord (acquisition units) into NormalizedSignalRecord (SI engineering units).

Module 6: Thermodynamic and Mechanical Digital Twin.
Derives thermodynamic cycle metrics and mechanical engine state from NormalizedSignalRecord.

Module 7: Per-Cylinder EGT Diagnostics.
Evaluates per-cylinder EGT health, deviations, spread imbalance, and temporal rates.

Module 8: Lubrication Model.
Evaluates oil temperature, oil pressure, dynamic viscosity (Vogel model), margins, and pressure-viscosity interaction.

Module 9: Vibration Processing Subsystem.
Performs time-domain and FFT spectral analysis, windowing, and RPM order tracking on accelerometer signals.

Module 10: Misfire and Combustion Stability.
Performs multi-signal evidence fusion (EGT drops, crank speed fluctuation, vibration) to detect combustion instability and possible misfire per cylinder.

Module 11: Healthy Expectation Models and Residual Engine.
Generates operating-context-dependent healthy expectations and computes multi-signal physical residuals (observed - expected).
"""

from src.l2_digital_twin.egt_diagnostics import (
    CylinderEGTDiagnostic,
    EGTDiagnosticResult,
    EGTDiagnosticsEngine,
    evaluate_egt_diagnostics,
)
from src.l2_digital_twin.lubrication_model import (
    LubricationModel,
    LubricationModelResult,
    evaluate_lubrication_model,
)
from src.l2_digital_twin.misfire_classifier import (
    CylinderCombustionEvidence,
    MisfireDiagnosticResult,
    MisfireDetector,
    evaluate_misfire_detector,
)
from src.l2_digital_twin.residual_engine import (
    HealthyExpectationModel,
    HealthyExpectationResult,
    QuantityResidual,
    ResidualEngine,
    evaluate_residual_engine,
)
from src.l2_digital_twin.sensor_inverse import (
    SensorInverseModel,
    convert_raw_to_engineering_state,
)
from src.l2_digital_twin.thermo_mechanical_twin import (
    EngineGeometry,
    MechanicalTwin,
    ThermodynamicMechanicalTwin,
    ThermodynamicTwin,
    evaluate_digital_twin,
)
from src.l2_digital_twin.vibration_processor import (
    AxisVibrationFeatures,
    VibrationProcessor,
    VibrationSignalProcessingResult,
    evaluate_vibration_processor,
)

__all__ = [
    "SensorInverseModel",
    "convert_raw_to_engineering_state",
    "EngineGeometry",
    "ThermodynamicTwin",
    "MechanicalTwin",
    "ThermodynamicMechanicalTwin",
    "evaluate_digital_twin",
    "EGTDiagnosticsEngine",
    "evaluate_egt_diagnostics",
    "CylinderEGTDiagnostic",
    "EGTDiagnosticResult",
    "LubricationModel",
    "evaluate_lubrication_model",
    "LubricationModelResult",
    "VibrationProcessor",
    "evaluate_vibration_processor",
    "AxisVibrationFeatures",
    "VibrationSignalProcessingResult",
    "MisfireDetector",
    "evaluate_misfire_detector",
    "CylinderCombustionEvidence",
    "MisfireDiagnosticResult",
    "HealthyExpectationModel",
    "HealthyExpectationResult",
    "QuantityResidual",
    "ResidualEngine",
    "evaluate_residual_engine",
]






