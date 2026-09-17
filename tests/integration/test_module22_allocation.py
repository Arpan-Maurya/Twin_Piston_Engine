"""
Integration tests for Data Allocation and Dataset Splitting (Module 22).
"""

import pytest
from datetime import datetime, timezone

from src.core.exceptions import ValidationException
from src.core.provenance import Provenance
from src.dataset_validation import (
    DataAllocator,
    DatasetManifest,
    DatasetRecordContainer,
    DatasetSplitter,
    IntendedUse,
    SourceType,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.simulator.forward_simulator import ScenarioRunner


@pytest.fixture
def scenario_container():
    runner = ScenarioRunner(seed=10)
    records, gts, _ = runner.run_scenario(duration_s=10.0, dt_s=1.0)
    manifest = DatasetManifest(
        dataset_id="ds_scenario_10",
        source_type=SourceType.SIMULATED,
        intended_use=IntendedUse.DEVELOPMENT,
    )
    return DatasetRecordContainer(manifest=manifest, records=records, ground_truths=gts)


class TestDataAllocationAndSplitting:

    def test_data_allocator_permits_valid_allocation(self):
        m1 = DatasetManifest(dataset_id="ds1", source_type=SourceType.SIMULATED, intended_use=IntendedUse.TRAINING)
        assert DataAllocator.validate_allocation(m1) is True

        m2 = DatasetManifest(dataset_id="ds2", source_type=SourceType.REPLAY, intended_use=IntendedUse.REPLAY)
        assert DataAllocator.validate_allocation(m2) is True

    def test_data_allocator_rejects_unpermitted_live_training(self):
        m_live = DatasetManifest(dataset_id="ds_live", source_type=SourceType.LIVE, intended_use=IntendedUse.TRAINING)
        with pytest.raises(ValidationException):
            DataAllocator.validate_allocation(m_live)

    def test_chronological_time_splitting(self, scenario_container):
        train_c, val_c, test_c = DatasetSplitter.split_by_time(
            scenario_container, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        assert len(train_c.records) == 6
        assert len(val_c.records) == 2
        assert len(test_c.records) == 2

        # Verify strict chronological order
        train_max_ts = max(r.timestamp for r in train_c.records)
        val_min_ts = min(r.timestamp for r in val_c.records)
        val_max_ts = max(r.timestamp for r in val_c.records)
        test_min_ts = min(r.timestamp for r in test_c.records)

        assert train_max_ts <= val_min_ts
        assert val_max_ts <= test_min_ts

    def test_scenario_group_splitting(self, scenario_container):
        containers = [scenario_container.model_copy() for _ in range(10)]
        train_list, val_list, test_list = DatasetSplitter.split_by_scenario(
            containers, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
        )

        assert len(train_list) == 7
        assert len(val_list) == 2
        assert len(test_list) == 1
