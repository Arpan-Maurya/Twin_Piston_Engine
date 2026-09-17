# Module 23: Performance Optimization and Resilience

## Overview
This document details the architectural optimizations, resource controls, concurrency protections, and failure-isolation mechanisms implemented in Original Module 23 for the Aero Piston Engine Digital Twin system.

All performance optimizations strictly adhere to the core invariant:
$$\text{SAME INPUT} + \text{SAME CONFIG} \implies \text{SAME SCIENTIFIC OUTPUT}$$

---

## 1. Measured Baseline Benchmarks

Measurements obtained using `BenchmarkHarness` over 50 synthetic records:

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| Telemetry Ingestion Latency | **0.004** | ms / record |
| Digital Twin Processing Latency | **0.502** | ms / record |
| ML Inference Status | **AVAILABLE** | - |
| ML Inference Latency | **0.377** | ms / record |
| End-to-End Pipeline Latency | **1.255** | ms / record |
| Replay Throughput | **326,370.8** | records / sec |
| What-If Execution Time (10s scenario) | **0.514** | ms |
| Tested Sample Count | **50** | records |

---

## 2. Implemented Optimizations & Safety Rationale

### CPU & Calculation Optimizations
1. **LRU Dynamic Caching (`FastLRUCache` & `@cache_deterministic`)**:
   - Dynamic caching of pure, deterministic functions (e.g., static geometric calculations, constant sensor calibrations, reference model metadata).
   - **Safety Rationale**: Only pure deterministic functions with hashable arguments are cached. Live telemetry, time-dependent diagnostics, RUL, Health Index, and advisories are never cached.

2. **Efficient Serialization & Batch Handling**:
   - Direct record dictionary transformation without intermediate JSON round-tripping during bulk sequence processing.
   - **Safety Rationale**: Preserves exact field types, units, and validation schema requirements.

### Memory Management & Resource Limits
1. **Bounded Collections (`ResourceLimitsConfig`)**:
   - `max_telemetry_buffer_size`: 1,000 records per edge buffer.
   - `max_history_query_window`: 10,000 records max per query.
   - `max_websocket_clients`: 100 simultaneous client connections.
   - `max_ws_queue_per_client`: 500 events per client queue.
   - `max_replay_records`: 50,000 records per scenario.
   - `max_what_if_scenarios`: 20 active scenario models.
   - `request_timeout_s`: 5.0 seconds default timeout.

2. **WebSocket Slow-Client Isolation (`ConnectionManager`)**:
   - Non-blocking asynchronous broadcast (`asyncio.wait_for(client.send_text(...), timeout=0.5)`).
   - Slow or unresponsive clients trigger explicit queue eviction without blocking the main engine broadcast loop or other connected clients.

---

## 3. Resilience, Retries & Failure Isolation

### Subsystem Failure Isolation (`safe_subsystem_call`)
- Subsystem calls (ML inference, diagnostic execution, replay simulation, advisory generation) are wrapped in exception isolation boundaries.
- If a subsystem fails (e.g. `FileNotFoundError` for missing ML model weight, timeout, or runtime error), it degrades gracefully by returning a structured degraded fallback state (`AVAILABLE`, `DEGRADED`, or `UNAVAILABLE`) instead of crashing the process or whole pipeline.

### Retry Logic (`retry_with_backoff`)
- Configurable retries with exponential backoff (`max_retries=3`, `initial_delay_s=0.1`, `backoff_factor=2.0`).
- Applied strictly to idempotent read/transient network/DB query operations. Non-idempotent operations (telemetry ingestion, state mutation) do not retry automatically to prevent duplicate records or retry storms.

---

## 4. Concurrency Model & State Safety
- **Thread-Safe Metrics (`PerformanceTracker`)**: Uses `threading.Lock` to protect internal counter dictionaries, latency accumulators, and active client gauges.
- **WebSocket Connection Safety**: Uses `asyncio.Lock` to serialize updates to active client connection sets and buffer eviction.
- **Simulation Replay Safety**: What-If scenario cloning uses deep isolation copies (`copy.deepcopy`) to guarantee ground-truth isolation and zero baseline mutation during scenario exploration.

---

## 5. Benchmark Methodology & Reproducibility
- Profiling is executed via `BenchmarkHarness(settings)` in `src/performance_resilience/benchmark.py`.
- Seed-controlled scenarios (`seed=42`) ensure consistent synthetic input state generation.
- Timings are recorded using high-precision OS monotonic timers (`time.perf_counter()`).

---

## 6. Limitations
- Benchmarks are executed in single-host local environment; distributed cluster throughput will depend on network I/O and database latency.
- Hard real-time guarantees are non-claimable; performance targets serve as engineering goals.
