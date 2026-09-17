# AI-Enabled Real-Time Digital Twin for Aero Piston Engine Health Monitoring, Fault Prediction & Mission Reliability in MALE UAVs

**Reference Engine**: Rotax 915 iS (1352 cc, 4-cylinder turbocharged boxer, 105 kW)  
**Application**: MALE UAV health monitoring, fault prediction, and mission reliability  
**Backend Modules**: 0–26 (All Complete)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Configuration System](#4-configuration-system)
5. [Module 0 — Physical Constants & Units](#5-module-0--physical-constants--units)
6. [Module 1 — Core Schemas & Provenance](#6-module-1--core-schemas--provenance)
7. [Module 2 — Exception Hierarchy](#7-module-2--exception-hierarchy)
8. [Module 3 — Telemetry Validation & Security](#8-module-3--telemetry-validation--security)
9. [Module 4 — Sensor Inverse Model](#9-module-4--sensor-inverse-model)
10. [Module 5 — Thermo-Mechanical Digital Twin](#10-module-5--thermo-mechanical-digital-twin)
11. [Module 6 — Lubrication Model](#11-module-6--lubrication-model)
12. [Module 7 — EGT Diagnostics](#12-module-7--egt-diagnostics)
13. [Module 8 — Vibration Processor](#13-module-8--vibration-processor)
14. [Module 9 — Combustion Stability & Misfire Classifier](#14-module-9--combustion-stability--misfire-classifier)
15. [Module 10 — Residual Engine](#15-module-10--residual-engine)
16. [Module 11 — Forward Simulator](#16-module-11--forward-simulator)
17. [Module 12 — ML Inference Infrastructure](#17-module-12--ml-inference-infrastructure)
18. [Module 13 — Anomaly Detection & Fault Classifier](#18-module-13--anomaly-detection--fault-classifier)
19. [Module 14 — Health Index & Degradation Supervision](#19-module-14--health-index--degradation-supervision)
20. [Module 15 — RUL Estimation with Uncertainty](#20-module-15--rul-estimation-with-uncertainty)
21. [Module 16 — Mission Phase & Mission Risk](#21-module-16--mission-phase--mission-risk)
22. [Module 17 — Signal Forwarding & Lineage](#22-module-17--signal-forwarding--lineage)
23. [Module 18 — Scenario Replay & What-If](#23-module-18--scenario-replay--what-if)
24. [Module 19 — Advisory & Explainability Engine](#24-module-19--advisory--explainability-engine)
25. [Module 20 — REST API & Real-Time Backend](#25-module-20--rest-api--real-time-backend)
26. [Module 21 — Edge/Ground-Station Partition](#26-module-21--edgeground-station-partition)
27. [Module 22 — Dataset Validation Harness](#27-module-22--dataset-validation-harness)
28. [Module 23 — Performance & Resilience](#28-module-23--performance--resilience)
29. [Module 24 — Scientific Acceptance Suite](#29-module-24--scientific-acceptance-suite)
30. [Module 25 — Docker, CI & Production Package](#30-module-25--docker-ci--production-package)
31. [Module 26 — Final Backend Audit](#31-module-26--final-backend-audit)
32. [Data Flow & Boundaries](#32-data-flow--boundaries)
33. [Nine-Class Fault Taxonomy](#33-nine-class-fault-taxonomy)
34. [API Reference](#34-api-reference)
35. [Database & Migrations](#35-database--migrations)
36. [Testing](#36-testing)
37. [Docker Deployment](#37-docker-deployment)
38. [Environment Variables](#38-environment-variables)
39. [Troubleshooting](#39-troubleshooting)
40. [Reference Engine — Rotax 915 iS](#40-reference-engine--rotax-915-is)
41. [License](#41-license)

---

## 1. Architecture Overview

The backend is organized into five strict layers with a unidirectional data flow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          L5 — Interface                                  │
│              REST API · WebSocket · Replay · What-If                     │
│              Modules: 20                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                          L4 — Advisory                                   │
│       Explanations · Evidence · Maintenance · Mission Reports            │
│       Modules: 19                                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                      L3 — ML / Supervision                               │
│     Anomaly Detection · 9-Class Fault · Health Index · RUL · Risk        │
│     Modules: 12, 13, 14, 15, 16                                         │
├──────────────────────────────────────────────────────────────────────────┤
│                      L2 — Digital Twin                                   │
│    Thermo · Mechanical · Lubrication · EGT · Vibration · Residuals       │
│    Modules: 4, 5, 6, 7, 8, 9, 10                                        │
├═══════════════════  STRICT RAW-SIGNAL BOUNDARY  ════════════════════════╡
│                       L1 — Data Ingestion                                │
│         Simulator · CSV Replay · Live Telemetry · Adapters               │
│         Modules: 3, 11, 17, 18                                           │
└──────────────────────────────────────────────────────────────────────────┘
Cross-cutting: Modules 0, 1, 2 (constants, schemas, exceptions)
Deployment:    Module 21 (Edge/Ground partition)
QA & Ops:      Modules 22, 23, 24, 25, 26
```

### Key Architectural Rules

1. **Advisory only** — the system never sends engine control/actuation commands. `ActuationAttemptError` is raised on any violation.
2. **Strict L1→L2 boundary** — only `RawSignalRecord` crosses from L1 to L2. L2 must never access simulator internals or `SimulationGroundTruth`.
3. **ML consumes residuals** — L3 ML models receive residual features from L2, not raw sensor channels.
4. **Provenance on every value** — `REAL`, `SIMULATED`, `CSV_REPLAY`, `DERIVED`, `MODEL_OUTPUT` via `ProvenanceTaggedValue`.
5. **Offline-capable** — no internet access required at runtime.
6. **ML graceful degradation** — missing ONNX model artifacts trigger `MODEL_UNAVAILABLE` status; deterministic heuristic fallback rules activate automatically.

---

## 2. Repository Structure

```
Piston_Engine/
├── .env.example                    # Environment variable template
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD pipeline
├── Dockerfile                      # Multi-stage production container
├── Makefile                        # Developer convenience targets
├── alembic/                        # Database migration scripts
├── alembic.ini                     # Alembic configuration
├── config/
│   ├── default.yaml                # Externalized YAML configuration
│   └── logging.yaml                # Structured logging configuration
├── docker-compose.yml              # Production compose stack
├── pyproject.toml                  # Project metadata & dependencies
├── src/
│   ├── __init__.py
│   ├── api/                        # L5: FastAPI application & dependencies
│   │   ├── app.py                  # Application factory & exception handlers
│   │   ├── dependencies.py         # FastAPI dependency injection providers
│   │   └── v1/                     # Versioned REST & WebSocket routes
│   │       ├── __init__.py         # Master v1 router aggregation
│   │       ├── advisories.py       # Advisory & Q&A endpoints
│   │       ├── diagnostics.py      # Diagnostic query endpoints
│   │       ├── engine_health.py    # Physical engine health endpoints
│   │       ├── fault_anomaly.py    # Anomaly & fault classification
│   │       ├── history.py          # Historical data pagination
│   │       ├── mission_risk.py     # Mission risk & phase endpoints
│   │       ├── replay.py           # Scenario replay control
│   │       ├── rul.py              # RUL prediction endpoints
│   │       ├── schemas.py          # API-layer Pydantic schemas
│   │       ├── system.py           # Service health & status
│   │       ├── telemetry.py        # Raw telemetry ingestion & query
│   │       ├── websocket.py        # Real-time WebSocket streaming
│   │       └── what_if.py          # What-if scenario endpoints
│   ├── core/                       # Cross-cutting foundation
│   │   ├── config.py               # Pydantic Settings + YAML loader
│   │   ├── constants.py            # Physical constants (NIST-sourced)
│   │   ├── exceptions.py           # Exception hierarchy
│   │   ├── interfaces.py           # Abstract protocol definitions
│   │   ├── logging.py              # Structured JSON logger
│   │   ├── provenance.py           # Provenance & enumeration types
│   │   ├── schemas.py              # Base Pydantic schemas
│   │   └── units.py                # Unit conversion functions
│   ├── dataset_validation/         # Module 22: Dataset/source validation
│   │   ├── allocation.py           # Train/val/test split allocation
│   │   ├── harness.py              # Validation harness runner
│   │   ├── manifest.py             # Dataset manifest tracking
│   │   ├── metrics.py              # Statistical quality metrics
│   │   ├── quality_schema.py       # Quality gate schemas
│   │   ├── report.py               # Validation report generation
│   │   └── taxonomy.py             # Source type classification
│   ├── edge_ground/                # Module 21: Edge/Ground partition
│   │   ├── buffer.py               # Bounded telemetry buffer
│   │   ├── edge_node.py            # Edge onboard processing node
│   │   ├── envelope.py             # Transport envelope schema
│   │   ├── ground_gateway.py       # Ground station gateway receiver
│   │   ├── role_enforcement.py     # Deployment role access control
│   │   └── transport.py            # Pluggable transport layer
│   ├── l1_data/                    # L1: Data ingestion layer
│   │   ├── adapters/               # Telemetry source adapters
│   │   │   ├── base.py             # Abstract adapter protocol
│   │   │   ├── can_transport.py    # CAN bus adapter
│   │   │   ├── csv_replay_adapter.py # CSV file replay
│   │   │   ├── live_telemetry_adapter.py # Live hardware telemetry
│   │   │   └── simulator_adapter.py # Simulator-to-L1 adapter
│   │   ├── lineage.py              # Data lineage tracking
│   │   ├── raw_repository.py       # Raw telemetry repository
│   │   ├── raw_signal_record.py    # Canonical RawSignalRecord schema
│   │   ├── sensor_forward.py       # Module 17 sensor forwarding
│   │   ├── signal_record.py        # NormalizedSignalRecord schema
│   │   ├── simulator/              # Physics forward simulator
│   │   │   ├── cooling.py          # Cooling subsystem model
│   │   │   ├── cylinder.py         # Per-cylinder combustion model
│   │   │   ├── engine_model.py     # Complete engine model
│   │   │   ├── fault_injection.py  # Fault scenario injection
│   │   │   ├── forward_simulator.py # ScenarioRunner orchestrator
│   │   │   ├── lubrication_sim.py  # Lubrication subsystem model
│   │   │   ├── replay_and_whatif.py # Replay/What-If engines
│   │   │   ├── rotax_915is_params.py # Rotax 915 iS parameters
│   │   │   ├── sensors.py          # Sensor noise models
│   │   │   └── turbocharger.py     # Turbocharger model
│   │   ├── telemetry_security.py   # HMAC signing & verification
│   │   ├── telemetry_validator.py  # Packet validation pipeline
│   │   └── validation.py           # Channel-level validation rules
│   ├── l2_digital_twin/            # L2: Physics digital twin layer
│   │   ├── egt_diagnostics.py      # Per-cylinder EGT analysis
│   │   ├── lubrication_model.py    # Lubrication state derivation
│   │   ├── misfire_classifier.py   # Combustion stability analysis
│   │   ├── residual_engine.py      # Actual-vs-expected residuals
│   │   ├── sensor_inverse.py       # Raw → engineering conversion
│   │   ├── thermo_mechanical_twin.py # Core thermodynamic model
│   │   └── vibration_processor.py  # Vibration feature extraction
│   ├── l3_ml/                      # L3: ML & supervision layer
│   │   ├── advisory_explainability.py # Module 19 advisory engine
│   │   ├── anomaly_fault.py        # Module 13 anomaly & fault
│   │   ├── health_supervision.py   # Module 14 health index
│   │   ├── mission_risk.py         # Module 16 mission risk
│   │   ├── ml_infrastructure.py    # Module 12 ML infrastructure
│   │   └── rul_estimation.py       # Module 15 RUL estimation
│   ├── performance_resilience/     # Module 23: Performance optimization
│   │   ├── benchmark.py            # Performance benchmarking
│   │   ├── cache.py                # Computation cache
│   │   ├── limits.py               # Resource bounds enforcement
│   │   ├── resilience.py           # Failure isolation & recovery
│   │   └── tracker.py              # Runtime metrics tracker
│   └── scientific_acceptance/      # Module 24: Scientific acceptance
│       ├── acceptance_models.py    # Acceptance criteria models
│       └── acceptance_report.py    # Report generator
├── tests/
│   ├── conftest.py                 # Shared test fixtures
│   ├── unit/                       # 27 unit test files (Modules 0–23)
│   ├── integration/                # 33 integration/boundary test files
│   └── scientific/                 # 8 scientific acceptance test files
└── docs/                           # Architecture documentation
```

---

## 3. Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| [uv](https://docs.astral.sh/uv/) | Latest |

### Local Development Setup

```bash
# Clone the repository
git clone <repo-url> Piston_Engine
cd Piston_Engine

# Install all dependencies (including dev tools)
uv sync --extra dev

# Copy environment template
cp .env.example .env

# Run the full test suite
uv run pytest tests/ -v

# Start the API server (development mode with auto-reload)
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify

Open http://localhost:8000/docs for interactive Swagger UI, or http://localhost:8000/redoc for ReDoc.

Check service health:

```bash
curl http://localhost:8000/api/v1/system/health
```

### Docker

```bash
# Build and start
docker-compose up -d

# Verify
curl http://localhost:8000/api/v1/system/health

# Stop
docker-compose down
```

### Makefile Targets

| Target | Command | Description |
|---|---|---|
| `make install` | `uv sync` | Install production dependencies |
| `make dev` | `uv sync --extra dev` | Install with dev tools |
| `make lint` | `ruff check + format --check` | Lint & format check |
| `make lint-fix` | `ruff check --fix + format` | Auto-fix lint issues |
| `make typecheck` | `mypy src/` | Static type checking |
| `make test` | `pytest tests/unit/ -v` | Unit tests only |
| `make test-integration` | `pytest tests/integration/ -v` | Integration tests |
| `make test-all` | `pytest tests/ -v` | Complete test suite |
| `make test-cov` | `pytest --cov=src` | Coverage report |
| `make run` | `uvicorn src.api.app:app` | Start API server |
| `make docker-build` | `docker build` | Build Docker image |
| `make docker-up` | `docker-compose up -d` | Start Docker stack |
| `make docker-down` | `docker-compose down` | Stop Docker stack |
| `make clean` | Remove caches | Clean build artifacts |

---

## 4. Configuration System

Configuration is centralized in `src/core/config.py` using Pydantic Settings. No physical constant, threshold, or tuning parameter is hardcoded in business logic.

### Load Priority (Highest Wins)

1. **Environment variables** (prefixed `APP_`)
2. **`.env` file**
3. **`config/default.yaml`**

### Key Configuration Sections

| Section | File Path | Description |
|---|---|---|
| `engine` | `config/default.yaml` | Rotax 915 iS geometric/operational parameters |
| `turbocharger` | `config/default.yaml` | Turbo boost, wastegate PI controller |
| `lubrication` | `config/default.yaml` | Oil system pressures, temps, Vogel viscosity |
| `cooling` | `config/default.yaml` | Coolant system thermostat & radiator |
| `telemetry` | `config/default.yaml` | Sample rate, channel count |
| `simulator` | `config/default.yaml` | Time step, noise models, dropout |
| `pipeline` | `config/default.yaml` | Cycle timeout, residual window |
| `ml` | `config/default.yaml` | Anomaly/fault thresholds, health weights |
| `advisory` | `config/default.yaml` | Advisory severity health-index thresholds |
| `security` | `config/default.yaml` | HMAC algorithm, stale timeouts |
| `persistence` | `config/default.yaml` | Storage backend, max records |
| `channel_ranges` | `config/default.yaml` | Engineering-unit valid ranges |
| `raw_channel_ranges` | `config/default.yaml` | Raw acquisition-unit valid ranges |
| `deployment` | `config/default.yaml` | Edge/Ground role, transport settings |

### Usage

```python
from src.core.config import get_settings

settings = get_settings()
bore = settings.engine.bore_mm          # 84.0
threshold = settings.ml.anomaly_threshold  # 0.85
```

---

## 5. Module 0 — Physical Constants & Units

**Files**: `src/core/constants.py`, `src/core/units.py`

Every constant carries a source citation in its docstring, ensuring traceability.

| Constant | Value | Source |
|---|---|---|
| `R_UNIVERSAL` | 8.314462618 J/(mol·K) | NIST CODATA 2018 |
| `GAMMA_AIR` | 1.4 | Heywood (2018), Table 3.3 |
| `LHV_AVGAS` | 43.5 MJ/kg | CRC Report No. 530 |
| `STOICHIOMETRIC_AFR` | 14.7 | Heywood (2018), Table 3.4 |
| `WIEBE_A` | 6.908 | Heywood (2018), §9.2 |
| `WIEBE_M` | 2.0 | Heywood (2018), §9.2 |
| `WOSCHNI_C1` | 3.26 | Woschni, SAE 670931 (1967) |
| `CHEN_FLYNN_C0` | 0.4×10⁵ Pa | Chen & Flynn, SAE 650733 |

All values in SI units. Conversion utilities in `units.py`.

---

## 6. Module 1 — Core Schemas & Provenance

**Files**: `src/core/schemas.py`, `src/core/provenance.py`

### Provenance Tags

Every output value in the system is wrapped in `ProvenanceTaggedValue[T]` carrying:

| Tag | Meaning |
|---|---|
| `REAL` | Live engine telemetry hardware |
| `SIMULATED` | Physics simulator output |
| `CSV_REPLAY` | Recorded CSV file playback |
| `DERIVED` | L2 physics derivation |
| `MODEL_OUTPUT` | L3 ML model prediction |

### Key Domain Schemas

| Schema | Layer | Description |
|---|---|---|
| `RawSignalRecord` | L1 | Canonical acquisition-level record (µV, Ω, ADC counts) |
| `OperatingPoint` | L2 | Engine operating state (RPM, MAP, altitude) |
| `DerivedEngineState` | L2 | Thermodynamic parameters (BMEP, η, torque, power) |
| `DiagnosticState` | L2 | Per-cylinder diagnostic status |
| `LubricationState` | L2 | Oil temperature, pressure, viscosity |
| `VibrationState` | L2 | Vibration features (RMS, crest factor, dominant frequency) |
| `CombustionStabilityState` | L2 | Misfire detection and combustion quality |
| `ResidualState` | L2 | Actual-vs-baseline residuals |
| `AnomalyResult` | L3 | Anomaly detection output |
| `FaultClassificationResult` | L3 | Nine-class fault classification |
| `HealthState` | L3 | Aggregate health index and degradation state |
| `RULState` | L3 | Remaining useful life with uncertainty bounds |
| `MissionState` | L3 | Flight phase, risk score, go/no-go |
| `Alert` | L4 | Advisory alert with recommended action |

---

## 7. Module 2 — Exception Hierarchy

**File**: `src/core/exceptions.py`

All exceptions inherit from `PistonEngineError`, enabling centralized API error handling.

```
PistonEngineError
├── ResourceNotFoundException       → 404
├── ValidationException             → 422
├── DigitalTwinError                → 500
├── InvalidChannelError             → 400 (per-channel validation failure)
├── PacketIntegrityError            → 400 (integrity check failure)
├── AdapterConnectionError          → 500 (data source connection failure)
├── BoundaryViolationError          → 400 (L1↔L2 boundary violation)
│   └── RoleViolationError          → 400 (wrong deployment role)
├── ConfigurationError              → 500 (invalid configuration)
├── ModelLoadError                  → 500 (ONNX model load failure)
├── ModelInferenceError             → 500 (ML inference failure)
└── ActuationAttemptError           → SAFETY (advisory-only violation)
```

**Critical**: `ActuationAttemptError` must never be caught and silenced. Any attempt to send engine control commands is a severe architectural violation.

---

## 8. Module 3 — Telemetry Validation & Security

**Files**: `src/l1_data/telemetry_validator.py`, `src/l1_data/telemetry_security.py`, `src/l1_data/validation.py`

### Validation Pipeline

1. **HMAC Signature Verification** — SHA-256 packet authentication
2. **Sequence Monotonicity** — detects dropped or replayed packets
3. **Per-Channel Range Checks** — rejects physically impossible raw values
4. **Staleness Detection** — per-channel configurable timeout windows
5. **Signal Quality Scoring** — aggregates per-channel quality into a packet-level score

### Security Rules

- Tampered packets are **rejected** (not silently corrected)
- Invalid channels are **flagged** — the system continues computing unaffected parameters
- No silent value replacement or imputation

---

## 9. Module 4 — Sensor Inverse Model

**File**: `src/l2_digital_twin/sensor_inverse.py`

Converts raw acquisition-unit signals to engineering-unit values:

| Raw Signal | Engineering Output |
|---|---|
| Type-K thermocouple µV + cold-junction °C | Temperature [K] |
| PT100 RTD Ω | Oil temperature [K] |
| ADC counts + Vref counts | Pressure [Pa] |
| Crank period µs | RPM [rev/min] |
| Fuel pulse Hz | Fuel flow [kg/s] |
| Accelerometer ADC counts | Acceleration [m/s²] |

Calibration coefficients are externalized in `config/default.yaml` → `sensor_calibration` section.

---

## 10. Module 5 — Thermo-Mechanical Digital Twin

**File**: `src/l2_digital_twin/thermo_mechanical_twin.py`

Core thermodynamic engine model producing `DerivedEngineState`:

| Output | Formula Basis |
|---|---|
| BMEP [Pa] | `P_brake / (V_d × N / n_R)` — Heywood (2018), §2.6 |
| Thermal Efficiency | `P_brake / (ṁ_f × LHV)` — First Law |
| Volumetric Efficiency | `ṁ_air / (ρ_ref × V_d × N/2)` — Heywood (2018), §6.2 |
| AFR | `ṁ_air / ṁ_f` |
| Brake Torque [Nm] | `BMEP × V_d / (2π × n_R)` |
| Brake Power [kW] | `Torque × 2π × N / 60` |
| FMEP [Pa] | Chen-Flynn: `C₀ + C₁·P_max + C₂·v̄_p + C₃·v̄_p²` |
| Mechanical Efficiency | `BMEP / (BMEP + FMEP)` |
| Mean Piston Speed | `2 × S × N / 60` |

---

## 11. Module 6 — Lubrication Model

**File**: `src/l2_digital_twin/lubrication_model.py`

Produces `LubricationState` with:
- Oil temperature derivation from RTD
- Oil pressure conversion from ADC
- Dynamic viscosity via **Vogel equation**: `μ = A × exp(B / (T − C))`
- Pressure and temperature margin calculations
- Multi-threshold diagnostic status (`NORMAL` / `WARNING` / `CRITICAL`)

---

## 12. Module 7 — EGT Diagnostics

**File**: `src/l2_digital_twin/egt_diagnostics.py`

Per-cylinder exhaust gas temperature analysis:
- Absolute temperature threshold monitoring
- Inter-cylinder spread analysis (max – min)
- Per-cylinder deviation from mean
- Rate-of-change detection (thermal transients)
- Cylinder-level status classification

Thresholds externalized in `config/default.yaml` → `egt_diagnostics`.

---

## 13. Module 8 — Vibration Processor

**File**: `src/l2_digital_twin/vibration_processor.py`

Extracts vibration features for ML consumption:
- 3-axis RMS acceleration
- Overall RMS and peak amplitude
- Crest factor (peak/RMS ratio)
- Dominant frequency via FFT
- Dominant order (frequency normalized to engine RPM)
- Configurable windowing (Hanning) and frequency bands

---

## 14. Module 9 — Combustion Stability & Misfire Classifier

**File**: `src/l2_digital_twin/misfire_classifier.py`

Evidence-fusion combustion stability analysis:

| Evidence Source | Weight |
|---|---|
| EGT drop/deviation | 0.40 |
| Crank irregularity | 0.35 |
| Vibration anomaly | 0.25 |

- Per-cylinder misfire detection
- Confidence scoring per cylinder
- Overall combustion status classification

---

## 15. Module 10 — Residual Engine

**File**: `src/l2_digital_twin/residual_engine.py`

Computes **actual − baseline expected** residuals at each operating point:
- Produces `ResidualState` containing per-channel residual `ProvenanceTaggedValue`
- Baseline expectations derived from `HealthyBaselineConfig`
- Normalization scales prevent dimensional bias
- Rolling window statistics for trend detection
- Residuals become the **input feature vector for L3 ML models**

---

## 16. Module 11 — Forward Simulator

**Files**: `src/l1_data/simulator/`

High-fidelity Rotax 915 iS physics simulator:
- Crank-angle resolved cylinder model with Wiebe combustion
- Woschni heat transfer correlation
- Turbocharger with PI-controlled wastegate
- Lubrication system (Vogel viscosity model)
- Cooling system with thermostat hysteresis
- Sensor noise injection (configurable σ per channel)
- Sensor dropout simulation
- **Fault injection** via `FaultScenarioConfig` (all 9 fault classes)
- `ScenarioRunner` orchestrator with deterministic seeding
- Produces `RawSignalRecord` + `SimulationGroundTruth` (ground truth is **isolated** and never exposed through the L1→L2 boundary)

---

## 17. Module 12 — ML Inference Infrastructure

**File**: `src/l3_ml/ml_infrastructure.py`

| Component | Purpose |
|---|---|
| `MLModelLoader` | Loads ONNX model artifacts from `models/` directory |
| `BaseMLModel` | Abstract base with lifecycle management |
| `MLFeatureVector` | Typed feature vector with schema versioning |
| `MLFeatureVectorBuilder` | Constructs feature vectors from `ResidualState` |
| `MLInferenceService` | Orchestrates load → validate → infer → package |
| `InferenceResult` | Standardized inference output container |

**Graceful degradation**: If no ONNX model file is found, the system enters `MODEL_UNAVAILABLE` state and falls back to deterministic heuristic rules. No crash, no silent failure.

---

## 18. Module 13 — Anomaly Detection & Fault Classifier

**File**: `src/l3_ml/anomaly_fault.py`

| Component | Output |
|---|---|
| `AnomalyDetector` | `AnomalyResult` — binary anomaly flag + score + threshold |
| `NineClassFaultClassifier` | `FaultClassificationResult` — predicted class, confidence, per-class probabilities |

Both support ONNX model backends with heuristic fallback when models are unavailable.

---

## 19. Module 14 — Health Index & Degradation Supervision

**File**: `src/l3_ml/health_supervision.py`

`HealthSupervisionEngine` produces `HealthState`:

| Component Weight | Default |
|---|---|
| Thermal | 0.20 |
| Lubrication | 0.20 |
| Vibration | 0.20 |
| Combustion | 0.15 |
| Performance | 0.10 |
| Anomaly/Fault | 0.15 |

**Degradation states** with hysteresis (±0.03):
`HEALTHY` → `WATCH` → `CAUTION` → `WARNING` → `CRITICAL`

---

## 20. Module 15 — RUL Estimation with Uncertainty

**File**: `src/l3_ml/rul_estimation.py`

| Component | Purpose |
|---|---|
| `HealthHistoryTracker` | Maintains health index time series |
| `RULEstimator` | Extrapolates time to degradation target threshold |

Produces `RULState` with:
- `hours_remaining` — point estimate
- `lower_bound_hours` / `upper_bound_hours` — confidence interval
- `confidence` — prediction quality score
- `uncertainty_available` flag

Minimum history: 3 samples. Maximum horizon: 2000 hours.

---

## 21. Module 16 — Mission Phase & Mission Risk

**File**: `src/l3_ml/mission_risk.py`

### Mission Phase Classifier

Detects flight phase from operating parameters:

| Phase | Detection Rule |
|---|---|
| `GROUND` | RPM < 1200 AND throttle < 15% |
| `TAKEOFF` | RPM ≥ 5000 AND MAP ≥ 110 kPa |
| `CLIMB` | RPM ≥ 4500 AND vertical rate > 1 m/s |
| `CRUISE` | Default if airborne, not climbing/descending |
| `DESCENT` | RPM < 3800 AND vertical rate < −1 m/s |
| `LANDING` | Altitude < 200 m during descent |

Phase transition debounce: 2 samples minimum.

### Mission Risk Engine

Risk score formula: `risk = (1 − health_index) × phase_weight`

Phase risk multipliers: TAKEOFF=1.5, CLIMB=1.3, LANDING=1.2, CRUISE/DESCENT=1.0, GROUND=0.5

Go/No-Go: `risk ≥ 0.75 → NO_GO`

---

## 22. Module 17 — Signal Forwarding & Lineage

**Files**: `src/l1_data/sensor_forward.py`, `src/l1_data/lineage.py`

- Forward-direction sensor signal routing and multiplexing
- Data lineage tracking for audit trail
- Source-to-destination traceability

---

## 23. Module 18 — Scenario Replay & What-If

**File**: `src/l1_data/simulator/replay_and_whatif.py`

| Component | Purpose |
|---|---|
| `ReplayEngine` | Scenario playback with play/pause/resume/stop/seek |
| `WhatIfEngine` | Clone baseline → modify parameters → execute |
| `PipelineReplayAdapter` | Routes records through L1→L2→L3→L4 pipeline |
| `ScenarioComparator` | Diff baseline vs. what-if outputs |

Replay statuses: `IDLE`, `PLAYING`, `PAUSED`, `STOPPED`, `COMPLETE`

---

## 24. Module 19 — Advisory & Explainability Engine

**File**: `src/l3_ml/advisory_explainability.py`

### Advisory Engine

Generates maintenance advisories based on health state:

| Health Index Range | Advisory Priority | Category |
|---|---|---|
| ≥ 0.7 | `MONITOR` | Routine monitoring |
| 0.5 – 0.7 | `SCHEDULE_MAINTENANCE` | Scheduled inspection |
| 0.3 – 0.5 | `IMMEDIATE_INSPECTION` | Urgent inspection required |
| < 0.3 | `GROUND_AIRCRAFT` | Do not fly |

### Explainability Engine

Produces `Explanation` objects containing:
- Natural language fault description
- Contributing factor analysis
- Evidence chain with confidence scores
- Diagnostic quality context

### Diagnostic Query Service

Interactive Q&A via `QuestionType`:
- `WHAT_IS_WRONG` — current fault identification
- `WHY_IS_THIS_HAPPENING` — root cause analysis
- `WHAT_SHOULD_I_DO` — recommended action
- `HOW_CONFIDENT` — prediction certainty
- `WHAT_IS_THE_TREND` — degradation trajectory

---

## 25. Module 20 — REST API & Real-Time Backend

**Files**: `src/api/`

### Application Factory

`create_app()` in `src/api/app.py` configures:
- Versioned router namespace at `/api/v1`
- CORS middleware (localhost:3000, localhost:8000)
- Centralized exception handlers (clean JSON error envelopes, zero stack trace leakage)
- OpenAPI docs at `/docs` (Swagger UI) and `/redoc`

### REST Endpoints

All endpoints are prefixed with `/api/v1`.

| Endpoint | Method | Description |
|---|---|---|
| `/system/health` | GET | API service availability (NOT engine health) |
| `/system/status` | GET | Subsystem operational readiness |
| `/telemetry` | POST | Ingest raw telemetry packet |
| `/telemetry/latest` | GET | Latest ingested raw record |
| `/telemetry/history` | GET | Paginated historical raw records |
| `/engine/health` | GET | Physical engine health index & degradation state |
| `/diagnostics/...` | GET | Thermodynamic & subsystem diagnostics |
| `/diagnostics/faults` | GET | Anomaly detection & fault classification |
| `/diagnostics/faults/anomaly` | GET | Latest anomaly result |
| `/engine/rul` | GET | RUL prediction with uncertainty |
| `/mission/risk` | GET | Mission risk & phase assessment |
| `/advisories/latest` | GET | Latest maintenance advisory |
| `/advisories/explain` | GET | Explainability for current state |
| `/advisories/query` | POST | Interactive diagnostic Q&A |
| `/history/...` | GET | Historical analysis records |
| `/replay/state` | GET | Current replay engine state |
| `/replay/start` | POST | Start scenario playback |
| `/replay/pause` | POST | Pause playback |
| `/replay/resume` | POST | Resume playback |
| `/replay/stop` | POST | Stop and reset playback |
| `/replay/seek` | POST | Seek to position/timestamp |
| `/scenarios/what-if` | POST | Create & execute what-if scenario |
| `/scenarios/{id}` | GET | Get scenario metadata |
| `/scenarios/compare` | POST | Compare baseline vs. what-if |

### WebSocket

| Endpoint | Protocol | Description |
|---|---|---|
| `/api/v1/ws/engine` | WebSocket | Real-time engine diagnostic event stream |

**Event envelope types**: `system_status`, `telemetry_update`, `health_update`

**Resilience features**:
- Maximum 100 concurrent clients
- 2-second per-client send timeout
- Slow-client isolation and disconnection
- Connection capacity enforcement

### Error Response Format

All errors return a structured JSON envelope:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Channel 'rpm' invalid: value=-100, reason=below_minimum",
  "details": { "channel": "rpm", "value": -100 }
}
```

---

## 26. Module 21 — Edge/Ground-Station Partition

**Files**: `src/edge_ground/`

Defines the deployment boundary between onboard (Edge) and server (Ground) processing:

```
EDGE (Onboard)                      GROUND (Server)
┌─────────────────┐                 ┌─────────────────────────────┐
│ L1 Acquisition   │                │ L2 Digital Twin              │
│ Validation        │  ─────────►   │ L3 ML Supervision            │
│ Buffering         │  Transport    │ L4 Advisory                  │
│ Telemetry Security│               │ L5 REST API + WebSocket      │
└─────────────────┘                 └─────────────────────────────┘
```

### Deployment Roles

| Role | Allowed Operations |
|---|---|
| `EDGE` | L1 acquisition, validation, buffering, secure transport |
| `GROUND` | L2–L5 full processing stack |
| `SIMULATION` | Full stack (development/testing) |

### Transport Features

- Pluggable transport backend (`in_memory` for dev, extensible for production)
- Bounded telemetry buffer with configurable overflow policy (`DISCARD_OLDEST` / `DISCARD_NEWEST`)
- Automatic reconnection with backoff
- Communication state tracking: `CONNECTED` → `DEGRADED` → `DISCONNECTED` → `BUFFERING` → `RECOVERING`
- Role enforcement via `RoleViolationError`

---

## 27. Module 22 — Dataset Validation Harness

**Files**: `src/dataset_validation/`

| Component | Purpose |
|---|---|
| `DatasetManifest` | Tracks dataset metadata, source, schema version |
| `DatasetAllocation` | Train/validation/test split management |
| `QualitySchema` | Statistical quality gate definitions |
| `ValidationHarness` | Executes quality checks against datasets |
| `ValidationMetrics` | Computes distributional statistics |
| `ValidationReport` | Generates structured pass/fail report |
| `SourceTaxonomy` | Classifies data sources (simulator, CSV, live, ground-truth) |

Core principle: **source → manifest → allocate → validate → report**.

---

## 28. Module 23 — Performance & Resilience

**Files**: `src/performance_resilience/`

| Component | Purpose |
|---|---|
| `benchmark.py` | Latency/throughput benchmarking framework |
| `cache.py` | Computation result caching |
| `limits.py` | Resource bounds enforcement (memory, connections) |
| `resilience.py` | Failure isolation, circuit breaker patterns |
| `tracker.py` | Runtime metrics (active WS clients, dropped events, latency) |

**Invariant**: Same input + same config = same scientific output. Optimization never changes results.

---

## 29. Module 24 — Scientific Acceptance Suite

**Files**: `src/scientific_acceptance/`, `tests/scientific/`

Validates scientific correctness across the full L1→L5 stack:

| Test Category | Scope |
|---|---|
| Sensor conversion accuracy | L1 → L2 inverse models |
| Physics equation correctness | L2 thermodynamic derivations |
| Diagnostic threshold integrity | L2 EGT, vibration, misfire |
| ML pipeline boundary compliance | L2 → L3 feature contract |
| Health index composition | L3 weighted aggregation |
| RUL prediction validity | L3 extrapolation bounds |
| End-to-end pipeline | L1 → L4 full stack |
| Deterministic reproducibility | Same seed → same output |

Report generation: `generate_scientific_acceptance_report()`

---

## 30. Module 25 — Docker, CI & Production Package

### Dockerfile

Multi-stage build (`python:3.11-slim`):
- **Stage 1 (Builder)**: `uv sync --no-dev` for dependency resolution
- **Stage 2 (Runtime)**: Non-root `appuser` (UID 1000), minimal attack surface

Health check against `/api/v1/system/health` every 30 seconds.

### GitHub Actions CI

**File**: `.github/workflows/ci.yml`

| Job | Steps |
|---|---|
| `test` | Checkout → Python 3.11 → uv install → Ruff lint → Pytest (unit + integration + scientific) → Scientific acceptance report |
| `package` | Checkout → Python build → Docker build → Container health smoke test |

Triggers: push to `main`/`master`/`feature/*`, PRs to `main`/`master`.

### Docker Compose

Single-service stack with:
- Port mapping: `8000:8000`
- Read-only config and model mounts
- Persistent data volume
- Named log volume
- Auto-restart: `unless-stopped`

---

## 31. Module 26 — Final Backend Audit

Non-code module. Comprehensive audit of Modules 0–25 covering:
- Architecture integrity verification
- L1↔L2 boundary compliance
- Scientific equation traceability
- API contract consistency
- ML graceful degradation
- Security (HMAC, no ground-truth leakage)
- Docker/CI packaging correctness
- Test completeness assessment

See `FINAL_BACKEND_AUDIT.md` for the complete audit report.

---

## 32. Data Flow & Boundaries

```
Hardware / Simulator / CSV
        │
        ▼
┌─────────────────────┐
│  RawSignalRecord    │  L1: Raw µV, Ω, ADC counts, µs, Hz
│  (sequence_number,  │
│   source_type,      │
│   integrity_hash,   │
│   signal_quality)   │
└─────────┬───────────┘
          │ ← STRICT BOUNDARY (only RawSignalRecord crosses)
          ▼
┌─────────────────────┐
│  Sensor Inverse     │  L2: Raw → Engineering units (K, Pa, RPM, kg/s)
│       ↓             │
│  Thermo-Mechanical  │  L2: BMEP, η, torque, power
│       ↓             │
│  Diagnostics        │  L2: EGT, lubrication, vibration, misfire
│       ↓             │
│  Residual Engine    │  L2: Actual − baseline residuals
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feature Vector     │  L3: Residuals → ML features
│       ↓             │
│  Anomaly + Fault    │  L3: Binary anomaly + 9-class fault
│       ↓             │
│  Health Supervision │  L3: Weighted health index + degradation
│       ↓             │
│  RUL Estimation     │  L3: Time-to-threshold with uncertainty
│       ↓             │
│  Mission Risk       │  L3: Phase detection + risk scoring
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Advisory Engine    │  L4: Maintenance recommendations
│  Explainability     │  L4: Natural language explanations
│  Diagnostic Q&A     │  L4: Interactive query service
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  REST API (v1)      │  L5: JSON endpoints
│  WebSocket          │  L5: Real-time event stream
│  Replay / What-If   │  L5: Scenario simulation control
└─────────────────────┘
```

### Boundary Rules

| Rule | Enforced By |
|---|---|
| Only `RawSignalRecord` crosses L1→L2 | `BoundaryViolationError` + integration tests |
| L2/L3 never access simulator internals | Boundary tests, no imports verified |
| Ground truth used only for validation | `SimulationGroundTruth` isolation |
| No actuation commands | `ActuationAttemptError` |
| Provenance on every value | `ProvenanceTaggedValue` wrapper |

---

## 33. Nine-Class Fault Taxonomy

| Class ID | Name | Description |
|---|---|---|
| 0 | `NOMINAL` | Normal operation |
| 1 | `MISFIRE` | Cylinder misfire event |
| 2 | `DETONATION_KNOCK` | Abnormal combustion detonation |
| 3 | `EXHAUST_VALVE_LEAK` | Exhaust valve seal failure |
| 4 | `INTAKE_BOOST_LEAK` | Intake/boost system leak |
| 5 | `OIL_DEGRADATION` | Oil quality degradation |
| 6 | `COOLING_FAULT` | Cooling system malfunction |
| 7 | `BEARING_WEAR` | Bearing mechanical wear |
| 8 | `SENSOR_FAULT` | Sensor malfunction/failure |

Defined in `src/core/provenance.py` → `FaultClass(int, Enum)`. Changing these requires updating the entire ML pipeline, training data, and explanation templates.

---

## 34. API Reference

### Interactive Documentation

| URL | Format |
|---|---|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc (readable) |
| `http://localhost:8000/openapi.json` | OpenAPI 3.x JSON schema |

### Authentication

Currently no authentication layer (prototype). The security module provides HMAC packet-level integrity verification for telemetry ingestion.

### CORS

Allowed origins (configurable):
- `http://localhost:3000`
- `http://localhost:8000`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:8000`

---

## 35. Database & Migrations

| Component | Technology |
|---|---|
| ORM | SQLAlchemy 2.x (async) |
| Driver | `aiosqlite` (dev), `asyncpg` (production PostgreSQL) |
| Migrations | Alembic |
| Default URL | `sqlite+aiosqlite:///./digital_twin.db` |

### Migration Commands

```bash
# Generate a new migration
uv run alembic revision --autogenerate -m "description"

# Apply all pending migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1
```

---

## 36. Testing

### Test Organization

| Directory | Count | Scope |
|---|---|---|
| `tests/unit/` | 27 files | Per-module unit tests (Modules 0–23) |
| `tests/integration/` | 33 files | Boundary enforcement, API integration, transport |
| `tests/scientific/` | 8 files | Module 24 scientific acceptance |

**Total**: 473+ tests across 68 test files.

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# Scientific acceptance only
uv run pytest tests/scientific/ -v

# With coverage report
uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Specific module
uv run pytest tests/unit/test_module5.py -v

# Exclude slow tests
uv run pytest tests/ -v -m "not slow"
```

### Test Markers

| Marker | Description |
|---|---|
| `slow` | Long-running tests |
| `integration` | Cross-module integration |
| `property` | Hypothesis property-based tests |
| `boundary` | L1-L2 boundary enforcement |

### Coverage

- Target: **≥ 80%** (enforced in `pyproject.toml`)
- Simulator code (`src/l1_data/simulator/`) is excluded from coverage requirements

---

## 37. Docker Deployment

### Build

```bash
# Build production image
docker build -t piston-engine-twin .

# Or via Makefile
make docker-build
```

### Run

```bash
# Full stack via Compose
docker-compose up -d

# Verify
curl http://localhost:8000/api/v1/system/health

# View logs
docker logs piston-engine-twin

# Stop
docker-compose down
```

### Container Details

| Property | Value |
|---|---|
| Base image | `python:3.11-slim` |
| User | `appuser` (UID 1000, non-root) |
| Port | 8000 |
| ASGI server | Uvicorn |
| Entrypoint | `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` |
| Health check | `/api/v1/system/health` every 30s |
| Data volume | `/app/data` |
| Log volume | `/app/logs` |
| Config mount | `/app/config` (read-only) |
| Models mount | `/app/models` (read-only) |

### Environment Overrides in Production

```yaml
environment:
  - APP_ENV=production
  - LOG_LEVEL=INFO
  - DATABASE_URL=sqlite+aiosqlite:///./data/digital_twin.db
  - TELEMETRY_SOURCE=simulator
  - DEPLOYMENT_ROLE=GROUND_STATION
```

---

## 38. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CONFIG_PATH` | `config/default.yaml` | YAML config file path |
| `DATABASE_URL` | `sqlite+aiosqlite:///./digital_twin.db` | Database connection URL |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_WORKERS` | `1` | Uvicorn worker count |
| `TELEMETRY_SOURCE` | `simulator` | Data source: `simulator`, `csv_replay`, `live` |
| `CSV_REPLAY_PATH` | `data/sample_flights/` | Path to CSV replay files |
| `MODEL_DIR` | `models/` | ONNX model artifacts directory |
| `TELEMETRY_SECRET_KEY` | (dev key) | HMAC signing key — **change in production** |
| `DEPLOYMENT_ROLE` | `SIMULATION` | Deployment role: `EDGE`, `GROUND`, `SIMULATION` |

---

## 39. Troubleshooting

### Common Issues

| Symptom | Cause | Resolution |
|---|---|---|
| `MODEL_UNAVAILABLE` in health/fault/RUL | No ONNX models in `models/` | Expected in dev — system uses heuristic fallback. Train models via `uv sync --extra train` + training scripts. |
| `ConfigurationError` on startup | Invalid YAML or missing config | Verify `config/default.yaml` syntax. Check `CONFIG_PATH` env var. |
| Telemetry packets rejected | HMAC signature mismatch | Ensure `TELEMETRY_SECRET_KEY` matches between sender and receiver. |
| `ActuationAttemptError` | Code path attempting control output | Architecture violation — review and remove actuation code path. |
| `RoleViolationError` | Wrong `DEPLOYMENT_ROLE` for operation | Set `DEPLOYMENT_ROLE=SIMULATION` for full-stack dev, `GROUND` for server deployment. |
| Database locked (SQLite) | Concurrent write contention | Use PostgreSQL (`asyncpg`) for production multi-worker deployment. |
| WebSocket connection refused | Max clients (100) reached | Check `ConnectionManager.max_clients`. Scale horizontally. |
| Slow WebSocket clients disconnected | 2s send timeout exceeded | Expected resilience behavior. Client should reconnect. |
| Test failures in `tests/scientific/` | Scientific invariant violation | Check for unintended changes to equations, constants, or thresholds. |
| Docker health check failing | App not ready within 10s | Increase `start_period` in `docker-compose.yml`. Check container logs. |

### Logging

Structured JSON logging via `config/logging.yaml`:
- Console: simple text format
- File: JSON format to `logs/digital_twin.log` (10 MB rotation, 5 backups)
- Per-module log levels configurable

```bash
# View real-time logs
tail -f logs/digital_twin.log | python -m json.tool
```

---

## 40. Reference Engine — Rotax 915 iS

| Parameter | Value |
|---|---|
| Displacement | 1352 cc |
| Configuration | 4-cylinder turbocharged boxer |
| Bore × Stroke | 84 mm × 61 mm |
| Compression Ratio | 10.5:1 |
| Rated Power | 105 kW @ 5800 RPM |
| Connecting Rod | 105 mm |
| Firing Order | 1-3-2-4 |
| Fuel | Avgas 100LL (LHV = 43.5 MJ/kg) |
| Stoichiometric AFR | 14.7:1 |
| Turbocharger | Max boost 1.45 bar, wastegate target 1.35 bar |
| Oil | AeroShell Sport Plus 4, 4.0 bar nominal |
| Coolant | 50/50 ethylene glycol, thermostat opens at 85°C |

---

## 41. License

Proprietary — All rights reserved.
