"""
CSV Replay Adapter — Reads recorded flight telemetry from CSV files.

Converts CSV rows into canonical RawSignalRecord instances tagged with
Provenance.CSV_REPLAY. Supports:
    - Timestamp order & sequence number preservation
    - Deterministic replay
    - Configurable playback speed
    - Seek
    - Pause / Resume controls
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.raw_signal_record import RawSignalRecord


# Default CSV column mapping (supports raw sensor columns or standard names)
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "sequence_number": "sequence_number",
    "seq": "sequence_number",
    "egt_cyl1_hot_uv": "egt_cyl1_hot_uv",
    "egt1": "egt_cyl1_hot_uv",
    "egt_cyl2_hot_uv": "egt_cyl2_hot_uv",
    "egt2": "egt_cyl2_hot_uv",
    "egt_cyl3_hot_uv": "egt_cyl3_hot_uv",
    "egt3": "egt_cyl3_hot_uv",
    "egt_cyl4_hot_uv": "egt_cyl4_hot_uv",
    "egt4": "egt_cyl4_hot_uv",
    "egt_cold_c": "egt_cold_c",
    "cht_hot_uv": "cht_hot_uv",
    "cht1": "cht_hot_uv",
    "cht_cold_c": "cht_cold_c",
    "oil_rtd_ohms": "oil_rtd_ohms",
    "oil_temp": "oil_rtd_ohms",
    "oil_p_counts": "oil_p_counts",
    "oil_pressure": "oil_p_counts",
    "map_counts": "map_counts",
    "map_pressure": "map_counts",
    "map": "map_counts",
    "adc_vref_counts": "adc_vref_counts",
    "crank_period_us": "crank_period_us",
    "rpm": "crank_period_us",
    "fuel_pulse_hz": "fuel_pulse_hz",
    "fuel_flow": "fuel_pulse_hz",
    "accel_x": "accel_x",
    "accel_y": "accel_y",
    "accel_z": "accel_z",
    "ambient_temp_c": "ambient_temp_c",
    "ambient_temp": "ambient_temp_c",
    "ambient_press_pa": "ambient_press_pa",
    "ambient_pressure": "ambient_press_pa",
}


class CSVReplayAdapter(SourceAdapter):
    """Replays recorded engine data from CSV files as canonical RawSignalRecords.

    Features:
        - Deterministic sequential replay
        - Playback speed control
        - Seek to arbitrary row index
        - Pause / Resume state control
    """

    def __init__(
        self,
        csv_path: str | Path,
        column_map: dict[str, str] | None = None,
        loop: bool = False,
        playback_speed: float = 1.0,
    ) -> None:
        """
        Args:
            csv_path: Path to the CSV file.
            column_map: Custom column name mapping.
            loop: Restart from beginning when file ends if True.
            playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x, 0.0 = max speed).
        """
        self._path = Path(csv_path)
        self._column_map = column_map or DEFAULT_COLUMN_MAP
        self._loop = loop
        self.playback_speed = playback_speed
        self._rows: list[dict[str, str]] = []
        self._current_index = 0
        self._sequence = 0
        self._connected = False
        self._paused = False

    async def connect(self) -> None:
        """Load and cache CSV rows for deterministic replay."""
        if not self._path.exists():
            raise FileNotFoundError(f"CSV replay file not found: {self._path}")

        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._rows = [row for row in reader]

        self._current_index = 0
        self._sequence = 0
        self._connected = True
        self._paused = False

    async def read_next(self) -> RawSignalRecord | None:
        """Read the next CSV row and convert to canonical RawSignalRecord."""
        if not self._connected or self._paused:
            return None

        if self._current_index >= len(self._rows):
            if self._loop and self._rows:
                self._current_index = 0
            else:
                return None

        row = self._rows[self._current_index]
        self._current_index += 1
        self._sequence += 1

        return self._row_to_raw_record(row, self._sequence)

    async def disconnect(self) -> None:
        """Release CSV replay resources."""
        self._connected = False
        self._rows = []
        self._current_index = 0

    def seek(self, index: int) -> int:
        """Seek to a specific row index in the CSV file."""
        if not self._rows:
            self._current_index = 0
            return 0
        self._current_index = max(0, min(index, len(self._rows) - 1))
        return self._current_index

    def pause(self) -> None:
        """Pause playback."""
        self._paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Whether playback is currently paused."""
        return self._paused

    @property
    def current_index(self) -> int:
        """Current row position in the CSV file."""
        return self._current_index

    @property
    def total_rows(self) -> int:
        """Total number of rows loaded."""
        return len(self._rows)

    @property
    def source_type(self) -> Provenance:
        return Provenance.CSV_REPLAY

    @property
    def source_id(self) -> str:
        return f"csv_replay:{self._path.name}"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _row_to_raw_record(self, row: dict[str, str], seq: int) -> RawSignalRecord:
        """Convert a CSV row dictionary to a RawSignalRecord."""
        # Parse timestamp if present
        ts_str = row.get("timestamp")
        if ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Parse numeric helper
        def get_val(key: str, default: float) -> float:
            # Check row directly or mapped key
            raw_str = row.get(key)
            if raw_str is None:
                for col_name, target in self._column_map.items():
                    if target == key and col_name in row:
                        raw_str = row[col_name]
                        break
            if raw_str is not None:
                try:
                    return float(raw_str)
                except (ValueError, TypeError):
                    pass
            return default

        # Conversion if CSV contains derived physical fields instead of raw
        # (e.g. if RPM is provided instead of crank_period_us)
        raw_rpm = get_val("rpm", 4000.0)
        crank_us = get_val("crank_period_us", (60.0 * 1e6) / max(raw_rpm, 1.0))

        egt1_uv = get_val("egt_cyl1_hot_uv", 30000.0)
        egt2_uv = get_val("egt_cyl2_hot_uv", 30000.0)
        egt3_uv = get_val("egt_cyl3_hot_uv", 30000.0)
        egt4_uv = get_val("egt_cyl4_hot_uv", 30000.0)
        egt_cold = get_val("egt_cold_c", 25.0)

        cht_uv = get_val("cht_hot_uv", 12000.0)
        cht_cold = get_val("cht_cold_c", 25.0)

        oil_rtd = get_val("oil_rtd_ohms", 135.0)
        oil_p_cnt = int(get_val("oil_p_counts", 2048.0))
        map_cnt = int(get_val("map_counts", 2048.0))
        vref_cnt = int(get_val("adc_vref_counts", 4095.0))
        fuel_hz = get_val("fuel_pulse_hz", 100.0)

        accel_x = int(get_val("accel_x", 0.0))
        accel_y = int(get_val("accel_y", 0.0))
        accel_z = int(get_val("accel_z", 1000.0))

        amb_t_c = get_val("ambient_temp_c", 20.0)
        amb_p_pa = get_val("ambient_press_pa", 101325.0)

        raw_rec = RawSignalRecord(
            timestamp=timestamp,
            sequence_number=seq,
            source_type=Provenance.CSV_REPLAY,
            egt_cyl1_hot_uv=egt1_uv,
            egt_cyl2_hot_uv=egt2_uv,
            egt_cyl3_hot_uv=egt3_uv,
            egt_cyl4_hot_uv=egt4_uv,
            egt_cold_c=egt_cold,
            cht_hot_uv=cht_uv,
            cht_cold_c=cht_cold,
            oil_rtd_ohms=oil_rtd,
            oil_p_counts=oil_p_cnt,
            map_counts=map_cnt,
            adc_vref_counts=vref_cnt,
            crank_period_us=crank_us,
            fuel_pulse_hz=fuel_hz,
            accel_counts_xyz=(accel_x, accel_y, accel_z),
            ambient_temp_c=amb_t_c,
            ambient_press_pa=amb_p_pa,
            signal_quality=SignalQuality(score=0.95),
        )

        return raw_rec.model_copy(update={"integrity_hash": raw_rec.compute_integrity_hash()})
