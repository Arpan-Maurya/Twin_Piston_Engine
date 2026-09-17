"""
Data Allocation, Splitting, and Data Leakage Detection — Module 22.

Provides deterministic data allocation rules, time-aware splitting for time-series,
scenario/group splitting, and strict data leakage detection (temporal, duplicate, ground-truth).
"""

from __future__ import annotations

from typing import Sequence
from src.core.exceptions import ValidationException
from src.dataset_validation.manifest import DatasetManifest, DatasetRecordContainer
from src.dataset_validation.taxonomy import IntendedUse, SourceType


class DataAllocator:
    """Enforces deterministic dataset allocation to intended-use roles."""

    PERMITTED_ALLOCATIONS: dict[SourceType, set[IntendedUse]] = {
        SourceType.SIMULATED: {
            IntendedUse.DEVELOPMENT,
            IntendedUse.TRAINING,
            IntendedUse.VALIDATION,
            IntendedUse.TEST,
            IntendedUse.SCIENTIFIC_VALIDATION,
            IntendedUse.DEMONSTRATION,
        },
        SourceType.REPLAY: {
            IntendedUse.REPLAY,
            IntendedUse.VALIDATION,
            IntendedUse.TEST,
            IntendedUse.SCIENTIFIC_VALIDATION,
            IntendedUse.DEMONSTRATION,
        },
        SourceType.LIVE: {
            IntendedUse.VALIDATION,
            IntendedUse.DEMONSTRATION,
        },
        SourceType.VALIDATION_GROUND_TRUTH: {
            IntendedUse.VALIDATION,
            IntendedUse.SCIENTIFIC_VALIDATION,
            IntendedUse.TEST,
        },
    }

    @classmethod
    def validate_allocation(cls, manifest: DatasetManifest) -> bool:
        """Validate if manifest source_type is permitted for intended_use."""
        allowed = cls.PERMITTED_ALLOCATIONS.get(manifest.source_type, set())
        if manifest.intended_use not in allowed:
            raise ValidationException(
                f"Source type '{manifest.source_type.value}' is not permitted for intended use "
                f"'{manifest.intended_use.value}'. Allowed uses: {[u.value for u in allowed]}"
            )
        return True


class DatasetSplitter:
    """Deterministic time-series and scenario group dataset splitter."""

    @staticmethod
    def split_by_time(
        container: DatasetRecordContainer,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[DatasetRecordContainer, DatasetRecordContainer, DatasetRecordContainer]:
        """Chronologically split a time-series dataset without random shuffling.

        Returns (train_container, val_container, test_container).
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError("Partition ratios must sum to 1.0")

        # Sort records chronologically
        sorted_records = sorted(container.records, key=lambda r: r.timestamp)
        total = len(sorted_records)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_recs = sorted_records[:train_end]
        val_recs = sorted_records[train_end:val_end]
        test_recs = sorted_records[val_end:]

        # Handle corresponding ground truths if present
        train_gts = container.ground_truths[:train_end] if container.ground_truths else None
        val_gts = container.ground_truths[train_end:val_end] if container.ground_truths else None
        test_gts = container.ground_truths[val_end:] if container.ground_truths else None

        train_manifest = container.manifest.model_copy(
            update={"dataset_id": f"{container.manifest.dataset_id}_train", "intended_use": IntendedUse.TRAINING}
        )
        val_manifest = container.manifest.model_copy(
            update={"dataset_id": f"{container.manifest.dataset_id}_val", "intended_use": IntendedUse.VALIDATION}
        )
        test_manifest = container.manifest.model_copy(
            update={"dataset_id": f"{container.manifest.dataset_id}_test", "intended_use": IntendedUse.TEST}
        )

        return (
            DatasetRecordContainer(manifest=train_manifest, records=train_recs, ground_truths=train_gts),
            DatasetRecordContainer(manifest=val_manifest, records=val_recs, ground_truths=val_gts),
            DatasetRecordContainer(manifest=test_manifest, records=test_recs, ground_truths=test_gts),
        )

    @staticmethod
    def split_by_scenario(
        containers: list[DatasetRecordContainer],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[list[DatasetRecordContainer], list[DatasetRecordContainer], list[DatasetRecordContainer]]:
        """Group-split multiple scenario datasets keeping entire scenarios together."""
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError("Partition ratios must sum to 1.0")

        total = len(containers)
        train_end = max(1, int(total * train_ratio))
        val_end = train_end + int(total * val_ratio)

        train_list = containers[:train_end]
        val_list = containers[train_end:val_end]
        test_list = containers[val_end:]

        return train_list, val_list, test_list


class LeakageDetector:
    """Detects temporal leakage, duplicate record leakage, and ground-truth leakage."""

    @staticmethod
    def check_temporal_leakage(
        part1: DatasetRecordContainer, part2: DatasetRecordContainer
    ) -> bool:
        """Check if time ranges of two dataset partitions overlap."""
        if not part1.records or not part2.records:
            return False

        p1_start = min(r.timestamp for r in part1.records)
        p1_end = max(r.timestamp for r in part1.records)
        p2_start = min(r.timestamp for r in part2.records)
        p2_end = max(r.timestamp for r in part2.records)

        # Overlap if max(p1_start, p2_start) <= min(p1_end, p2_end)
        return max(p1_start, p2_start) < min(p1_end, p2_end)

    @staticmethod
    def check_duplicate_leakage(
        part1: DatasetRecordContainer, part2: DatasetRecordContainer
    ) -> int:
        """Count duplicate timestamps between two partitions."""
        p1_ts = {r.timestamp for r in part1.records}
        p2_ts = {r.timestamp for r in part2.records}
        return len(p1_ts.intersection(p2_ts))

    @staticmethod
    def check_ground_truth_leakage(record: Any) -> bool:
        """Verify that record contains no ground-truth object references."""
        rec_str = str(record).lower()
        has_gt = "simulationgroundtruth" in rec_str or "wiebe" in rec_str
        return has_gt
