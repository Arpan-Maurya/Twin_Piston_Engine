# Frontend & Backend API Integration Document

**Project**: Aero Piston Engine Digital Twin for UAV Health Monitoring  
**API Version**: `v1`  
**Base URL**: `http://localhost:8000/api/v1`  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)  

---

## 1. Overview & Architecture

The REST & Real-Time API layer (`Module 20`) orchestrates interactions between ground-station / frontend clients and the underlying Digital Twin domain modules (`Modules 0–19`).

### Key Architectural Constraints
1. **Advisory-Only Safety**: The backend contains **ZERO** actuation or flight control endpoints. All recommendations are decision-support guidance only.
2. **No Business Logic in Routes**: API routes do NOT compute thermodynamics, ML inferences, or Health Index calculations; routes delegate strictly to domain application services.
3. **Ground-Truth Isolation**: Simulated ground-truth physical state (`SimulationGroundTruth`) is NEVER exposed over public diagnostic API endpoints.
4. **Data Ingestion Security**: Telemetry ingestion enforces HMAC signature verification and range checks via `Module 3` without bypass.

---

## 2. Authentication & Security Status

- **Current Status**: Prototype API interface layer. Authentication hooks and middleware stubs are prepared for future JWT/OAuth2 integration.
- **CORS**: Configurable allowed origins (`http://localhost:3000`, `http://localhost:8000`). Wildcard `*` is disabled in production settings.
- **Error Protection**: Stack traces, internal file paths, and environment secrets are sanitized and hidden from client error responses.

---

## 3. Data Semantics & Units

### Timestamp Format
All timestamps are returned as ISO 8601 UTC strings:
`YYYY-MM-DDTHH:MM:SS.ffffffZ` or `YYYY-MM-DDTHH:MM:SS+00:00`

### Provenance Semantics (`source_type` / `provenance`)
- `REAL`: Telemetry acquired from physical hardware sensors.
- `SIMULATED`: Data emitted by `Module 17` forward physics simulator.
- `CSV_REPLAY`: Data replayed from stored flight test logs.
- `DERIVED`: Values calculated by `Module 5–11` digital twin physics engines.
- `MODEL_OUTPUT`: Predictions produced by `Module 12–16` ML inference models.

### Signal Quality Semantics
Quality scores range from `0.0` (invalid/dropped) to `1.0` (perfect integrity).

### Enums
- **`FaultClass`**: `0: NOMINAL`, `1: MISFIRE`, `2: DETONATION_KNOCK`, `3: EXHAUST_VALVE_LEAK`, `4: INTAKE_BOOST_LEAK`, `5: OIL_DEGRADATION`, `6: COOLING_FAULT`, `7: BEARING_WEAR`, `8: SENSOR_FAULT`
- **`FlightPhase`**: `GROUND`, `TAKEOFF`, `CLIMB`, `CRUISE`, `DESCENT`, `LANDING`
- **`DegradationState`**: `HEALTHY`, `WATCH`, `CAUTION`, `WARNING`, `CRITICAL`
- **`AdvisoryCategory`**: `MONITOR`, `INSPECT`, `INVESTIGATE`, `MAINTENANCE_REVIEW`, `DATA_QUALITY_CHECK`, `CONTINUE_MONITORING`
- **`AdvisoryPriority`**: `INFORMATION`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

---

## 4. REST API Endpoint Reference

### A. System Endpoints
- `GET /api/v1/system/health`: REST API service operational health (distinguishes service health from physical engine health).
- `GET /api/v1/system/status`: Overall system subsystem readiness.

### B. Telemetry Endpoints
- `POST /api/v1/telemetry`: Ingest raw acquisition telemetry packet.
- `GET /api/v1/telemetry/latest`: Get latest canonical `RawSignalRecord`.
- `GET /api/v1/telemetry/history?page=1&page_size=50`: Get paginated raw telemetry history.

### C. Engine Health & Diagnostics Endpoints
- `GET /api/v1/engine/health`: Module 14 Health Index & degradation classification.
- `GET /api/v1/diagnostics`: Summary of all L2 physics diagnostics.
- `GET /api/v1/diagnostics/egt`: Module 7 per-cylinder EGT diagnostics.
- `GET /api/v1/diagnostics/lubrication`: Module 8 lubrication diagnostics.
- `GET /api/v1/diagnostics/vibration`: Module 9 vibration processing features.
- `GET /api/v1/diagnostics/combustion`: Module 10 misfire & stability diagnostics.

### D. ML Supervision & Risk Endpoints
- `GET /api/v1/diagnostics/anomaly`: Module 13 anomaly detection status & score.
- `GET /api/v1/diagnostics/fault`: Module 13 nine-class fault classification prediction.
- `GET /api/v1/engine/rul`: Module 15 Remaining Useful Life estimate & uncertainty bounds.
- `GET /api/v1/mission/risk`: Module 16 Flight Phase & analytical Mission Risk score.

### E. Advisories & Q&A Endpoints
- `GET /api/v1/advisories`: Active decision-support advisories.
- `GET /api/v1/advisories/{advisory_id}`: Detailed advisory by ID.
- `GET /api/v1/explanations`: Human-readable evidence-backed explanations.
- `POST /api/v1/diagnostic/query`: Operator Q&A query interface.

### F. History Endpoints
- `GET /api/v1/history/health`: Paginated Health Index history.
- `GET /api/v1/history/diagnostics`: Paginated diagnostic state history.
- `GET /api/v1/history/advisories`: Paginated decision-support advisory history.

### G. Replay & What-If Endpoints
- `POST /api/v1/replay/start`: Start scenario telemetry playback.
- `POST /api/v1/replay/pause`: Pause scenario replay.
- `POST /api/v1/replay/resume`: Resume scenario replay.
- `POST /api/v1/replay/stop`: Stop scenario replay.
- `POST /api/v1/replay/seek`: Seek replay by sample position or timestamp.
- `GET /api/v1/replay/state`: Get current replay status.
- `POST /api/v1/scenarios/what-if`: Execute what-if scenario cloning & simulation.
- `POST /api/v1/scenarios/compare`: Compare baseline vs what-if pipeline outputs.

---

## 5. Real-Time Streaming (WebSocket)

### Endpoint: `ws://localhost:8000/api/v1/ws/engine`

Read-only, exception-isolated, disconnect-safe WebSocket event stream.

### Event Envelope Format
```json
{
  "event_type": "health_update",
  "timestamp": "2026-09-17T12:00:00Z",
  "sequence_number": 42,
  "payload": {
    "health_index": 0.92,
    "is_anomaly": false,
    "anomaly_score": 0.05,
    "predicted_fault_class": "NOMINAL",
    "rul_hours": 1200.0,
    "mission_risk_score": 0.03
  },
  "quality": 1.0,
  "provenance": "DERIVED"
}
```

---

## 6. Standard Error Format

All API errors return consistent JSON responses:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Unsupported what-if parameter 'illegal_param'.",
  "details": {},
  "timestamp": "2026-09-17T12:00:00Z"
}
```

---

## 7. Edge / Ground Deployment Partition (Module 21)

- **Frontend Integration Point**: The Ground Station REST API (`/api/v1`) and WebSocket stream (`/api/v1/ws/engine`) remain the single integration interface for frontend applications and ground operators.
- **Edge Onboard Role**: The Edge node handles telemetry acquisition, L1 security/validation, store-and-forward buffering, and transport to Ground.
- **Communication Link Health States**: `CONNECTED`, `DEGRADED`, `DISCONNECTED`, `BUFFERING`, `RECOVERING`.
- **Store-and-Forward Buffering**: When Ground connectivity is lost, valid telemetry packets accumulate in a bounded FIFO queue and flush sequentially upon link restoration.
- **Data Freshness Distinction**: Telemetry freshness (time since last acquisition/transmission) is tracked independently from physical engine health index freshness.
