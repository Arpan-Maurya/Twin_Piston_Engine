"""
Integration tests for Data Leakage Detection (Module 22).
"""

import pytest
from datetime import datetime, timezone

from src.core.provenance import Provenance
from src.dataset_validation import (
    DatasetManifest,
    DatasetRecordContainer,
    DatasetSplitter,
    IntendedUse,
    LeakageDetector,
    SourceType,
)
from src.l1_data.simulator.forward_simulator import ScenarioRunner, SimulationGroundTruth


class TestLeakageDetection:

    def test_detect_temporal_leakage_on_overlapping_time_ranges(self):
        runner = ScenarioRunner(seed=1)
        records, gts, _ = runner.run_scenario(duration_s=10.0, dt_s=1.0)
        manifest = DatasetManifest(dataset_id="ds1", source_type=SourceType.SIMULATED, intended_use=IntendedUse.DEVELOPMENT)
        container = DatasetRecordContainer(manifest=manifest, records=records, ground_truths=gts)

        part1, part2, part3 = DatasetSplitter.split_by_time(container, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

        # Non-overlapping partitions should report no temporal leakage
        assert LeakageDetector.check_temporal_leakage(part1, part2) is False

        # Overlapping partitions should report temporal leakage
        overlapping_part = DatasetRecordContainer(manifest=manifest, records=records[2:7])
        assert LeakageDetector.check_temporal_leakage(part1, overlapping_part) is True

    def test_detect_duplicate_record_leakage(self):
        runner = ScenarioRunner(seed=1)
        records, gts, _ = runner.run_scenario(duration_s=5.0, dt_s=1.0)
        manifest = DatasetManifest(dataset_id="ds1", source_type=SourceType.SIMULATED, intended_use=IntendedUse.DEVELOPMENT)

        c1 = DatasetRecordContainer(manifest=manifest, records=records[:3])
        c2 = DatasetRecordContainer(manifest=manifest, records=records[2:5])  # record at index 2 is duplicate

        dup_cnt = LeakageDetector.check_duplicate_leakage(c1, c2)
        assert dup_cnt == 1

    def test_detect_ground_truth_leakage_in_record(self):
        runner = ScenarioRunner(seed=1)
        records, gts, _ = runner.run_scenario(duration_s=1.0, dt_s=1.0)
        raw_rec = records[0]

        # Normal RawSignalRecord has no ground truth leakage
        assert LeakageDetector.check_ground_truth_leakage(raw_rec) is False

        # If SimulationGroundTruth object or Wiebe text is attached to input:
        gt_obj = gts[0]
        assert LeakageDetector.check_ground_truth_leakage(gt_obj) is True
