# Edge / Ground-Station Partition Architecture (Module 21)

## 1. System Overview & Deployment Model

The Aero Piston Engine Digital Twin architecture defines a strict operational boundary between **Edge/Onboard** telemetry acquisition nodes and **Ground-Station/Server** analytics engines.

```
+-----------------------------------------------------------------------+
|                             EDGE / ONBOARD                            |
|                                                                       |
|  [Hardware Sensors / Adapter]                                         |
|              |                                                        |
|              v                                                        |
|  [L1 Telemetry Acquisition & Structural Check]                        |
|              |                                                        |
|              v                                                        |
|  [L1 Telemetry Security & Validation]                                 |
|  (HMAC Integrity, Monotonic Sequence, Stale Signal Check)             |
|              |                                                        |
|              v                                                        |
|  [Edge Store-and-Forward Bounded Buffer]                              |
|  (FIFO, Max Capacity, Discard Policy, Freshness Tracker)              |
+-----------------------------------------------------------------------+
                               |
                               | Transport (EdgeTelemetryEnvelope / RawSignalRecord)
                               v
+-----------------------------------------------------------------------+
|                         GROUND STATION / SERVER                       |
|                                                                       |
|  [Ground Telemetry Gateway & Persistence Repository]                  |
|              |                                                        |
|              v                                                        |
|  [L2 Digital Twin Physics]                                            |
|  (Thermo-mechanical derivation, EGT, Lubrication, Vib, Misfire)       |
|              |                                                        |
|              v                                                        |
|  [L3 ML Supervision]                                                  |
|  (Anomaly Detection, Nine-Class Fault Classifier)                     |
|              |                                                        |
|              v                                                        |
|  [L4 Supervisory Analytics]                                           |
|  (Health Index, RUL & Uncertainty, Mission Risk)                      |
|              |                                                        |
|              v                                                        |
|  [L5 Decision Support & User Interfaces]                              |
|  (Advisory Engine, Explainability, REST API v1, WebSocket Stream)     |
+-----------------------------------------------------------------------+
```

---

## 2. Partitioned Responsibilities

### Edge / Onboard Node Responsibilities
1. **Telemetry Acquisition**: Reads raw sensor channel counts and voltage signals from hardware or serial/UDP adapters.
2. **Source Adapters**: Provides standardized `RawSignalRecord` instances.
3. **Raw Telemetry Validation**: Executes Module 3 `TelemetryValidator` checking physical unit boundaries and non-null constraints.
4. **Packet Integrity & Security Checks**: Signs packets using `PacketSigner` and verifies signatures using `PacketVerifier`.
5. **Timestamp & Sequence Validation**: Enforces sequence monotonicity and anti-replay protection (`ReplayProtector`), flagging stuck sensors (`StaleSignalDetector`).
6. **Data Quality Assessment**: Attaches L1 `SignalQuality` flags to each packet.
7. **Store-and-Forward Buffering**: Enqueues valid packets into `EdgeStoreAndForwardBuffer` (bounded FIFO queue with configurable overflow drop policies).
8. **Telemetry Transport**: Transmits normalized `EdgeTelemetryEnvelope` objects over `ITelemetryTransport`.
9. **Communication Link Health Monitoring**: Tracks and exposes link state (`CONNECTED`, `DEGRADED`, `DISCONNECTED`, `BUFFERING`, `RECOVERING`).

### Ground Station / Server Responsibilities
1. **Historical Telemetry Persistence**: Stores all incoming `RawSignalRecord` payloads in persistent storage (`RawTelemetryRepositoryProtocol`).
2. **Engineering Unit Conversion**: Module 5 sensor inverse modelling (EGT, CHT, MAP, RTD, pulse conversion).
3. **Digital Twin Physics (L2)**: Derived thermo-mechanical states (Module 6), per-cylinder EGT diagnostics (Module 7), lubrication dynamics (Module 8), vibration order tracking (Module 9), and combustion misfire fusion (Module 10).
4. **Healthy Expectation & Residuals**: Module 11 physics-based residual generation.
5. **ML Supervision (L3)**: Feature vector assembly (Module 12), normalized residual anomaly detection, and Nine-Class fault taxonomy classification (Module 13).
6. **Health & Predictive Supervision (L4)**: Health Index aggregation (Module 14), Remaining Useful Life (RUL) estimation with uncertainty bounds (Module 15), and Mission Phase classification and risk scoring (Module 16).
7. **Decision Support & Explainability (L5)**: Decision-support advisory generation and human-readable evidence explanations (Module 19).
8. **Simulation Replay & What-If Sandbox**: Scenario replay, parameter modification, and run comparisons (Modules 17, 18).
9. **REST API & WebSocket Interfaces**: Module 20 API endpoints (`/api/v1`) and real-time WebSocket event streaming (`/api/v1/ws/engine`).

---

## 3. Strict Data Boundary & Telemetry Contract

- **Data Boundary Invariant**: ONLY canonical `RawSignalRecord` payloads wrapped in `EdgeTelemetryEnvelope` cross the Edge $\rightarrow$ Ground boundary.
- **Shielded Artifacts**: Ground truth physics states (`SimulationGroundTruth`), Wiebe combustion parameters, ML model weights, intermediate residual states, and database internal schemas are **NEVER** transmitted to Edge or received from Edge.
- **Ground State Reconstruction**: The Ground station reconstructs all higher-level engineering units, physics derived states, diagnostic states, ML predictions, and advisories purely from the canonical telemetry stream.

### Edge Telemetry Envelope Schema (`EdgeTelemetryEnvelope`)
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `record` | `RawSignalRecord` | Canonical raw telemetry payload. |
| `timestamp` | `datetime` | Packet timestamp (ISO 8601 UTC). |
| `sequence_number` | `int` | Monotonic sequence tracking number. |
| `source_type` | `Provenance` | Acquisition source (`REAL`, `SIMULATED`, `CSV_REPLAY`). |
| `quality` | `SignalQuality` | L1 signal quality score and validity flags. |
| `signature` | `str \| None` | HMAC SHA-256 integrity signature. |
| `schema_version` | `str` | Envelope schema version (default `"1.0.0"`). |
| `node_id` | `str` | Originating onboard node identifier. |

---

## 4. Edge Store-and-Forward Buffering & Offline Behavior

- **Queue Architecture**: Thread-safe, bounded FIFO queue (`EdgeStoreAndForwardBuffer`).
- **Configurable Overflow Policies**:
  - `DISCARD_OLDEST` (Default): When full capacity is reached, drops the oldest head envelope and enqueues the new envelope.
  - `DISCARD_NEWEST`: Drops incoming envelope when buffer is full.
- **Offline Link Loss Behavior**:
  - When connection is lost (`DISCONNECTED`), packets accumulate in strict FIFO order in `EdgeStoreAndForwardBuffer`.
  - When connection is restored (`RECOVERING`), the edge node automatically flushes buffered envelopes sequentially to Ground.
  - Non-blocking flush preserves sequence ordering without dropping packets.
- **Telemetry Freshness Tracking**:
  - Exposes `queue_depth`, `max_capacity`, `overflow_policy`, `dropped_packet_count`, `oldest_buffered_timestamp`, `newest_buffered_timestamp`, and `last_successful_transmission`.

---

## 5. Deployment Roles & Role Enforcement

System configuration (`AppSettings.deployment.role`) specifies one of three deployment roles:

1. `EDGE`: Onboard edge node role. Initializes only L1 acquisition, validation, buffering, and transport. Rejects attempts to initialize ML supervision, RUL, advisory, or ground API servers with `RoleViolationError`.
2. `GROUND`: Ground station server role. Initializes historical persistence, L2 Digital Twin, L3 ML models, L4 health/RUL/risk, L5 advisory engine, and Module 20 REST/WebSocket API services.
3. `SIMULATION`: Local development and end-to-end integration role. Allows full pipeline execution in a single process for automated testing and simulator validation.

---

## 6. What is Intentionally NOT on Edge
- **No Heavy ML Inference**: ML models (anomaly detection, nine-class fault classifier) run strictly on Ground server.
- **No RUL Estimation**: RUL curve fitting and uncertainty bounds are computed on Ground.
- **No Advisory Engine**: Human-readable advisory recommendations are generated on Ground.
- **No What-If Sandbox**: Scenario cloning and comparative physics simulations run on Ground.

## 7. What is Intentionally NOT Controllable from Ground
- **Advisory-Only Invariant**: Ground $\rightarrow$ Edge direction contains **NO** actuator commands, throttle commands, fuel mixture controls, or UAV flight commands.
- **Safety Violation Protection**: Any attempt to issue actuation commands raises an uncatchable `ActuationAttemptError`.

---

## 8. Simulator Compatibility
The physics forward simulator (Module 17) and replay engine (Module 18) remain fully compatible with the Edge/Ground partition. In `SIMULATION` deployment role, `ScenarioRunner` generates synthetic telemetry, passes it through `EdgeTelemetryNode`, transmits via `ITelemetryTransport`, and ingests into `GroundTelemetryGateway` without exposing `SimulationGroundTruth` to the ground inference pipeline.
