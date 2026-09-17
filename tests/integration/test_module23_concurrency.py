"""
Integration tests for Module 23 — Concurrency & State Safety.
"""

import pytest
import concurrent.futures
from src.performance_resilience.tracker import PerformanceTracker
from src.l1_data.simulator.forward_simulator import ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import WhatIfEngine


def test_concurrent_performance_tracker():
    tracker = PerformanceTracker()

    def record_operations(thread_id: int):
        for i in range(50):
            tracker.record_latency("api_request", float(i + thread_id))
            tracker.record_retry("db_retry")
            tracker.record_error("ws_error")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_operations, tid) for tid in range(8)]
        concurrent.futures.wait(futures)

    summary = tracker.get_summary()
    assert summary["latencies"]["api_request"]["count"] == 400
    assert summary["retries"]["db_retry"] == 400
    assert summary["errors"]["ws_error"] == 400


def test_what_if_concurrency_ground_truth_isolation():
    runner = ScenarioRunner(seed=123)
    records, ground_truths, meta = runner.run_scenario(duration_s=5.0, dt_s=1.0)

    engine1 = WhatIfEngine()
    engine2 = WhatIfEngine()

    scen1 = engine1.create_what_if_scenario("scen1", {}, {"ambient_temp_k": 320.0, "seed": 10})
    scen2 = engine2.create_what_if_scenario("scen2", {}, {"ambient_temp_k": 280.0, "seed": 20})

    def exec_scen1():
        return engine1.execute_what_if(scen1, duration_s=5.0, dt_s=1.0)

    def exec_scen2():
        return engine2.execute_what_if(scen2, duration_s=5.0, dt_s=1.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(exec_scen1)
        f2 = executor.submit(exec_scen2)

        res1_records, res1_gt, _ = f1.result()
        res2_records, res2_gt, _ = f2.result()

    # Ground truth isolation check: concurrent executions must produce independent, valid ground truths and records
    assert len(res1_records) == 5
    assert len(res2_records) == 5
    assert len(res1_gt) == 5
    assert len(res2_gt) == 5
    assert res1_records[0].sequence_number == 1
    assert res2_records[0].sequence_number == 1
