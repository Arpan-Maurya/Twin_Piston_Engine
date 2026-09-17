"""
Unit Tests for Original Module 9 — Vibration Processing Subsystem.

Tests time-domain features, FFT spectral analysis, known sine wave frequency detection,
windowing, multi-axis isolation, RPM order tracking, and robustness.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from src.core.config import get_settings
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import VibrationState
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.vibration_processor import (
    VibrationProcessor,
    VibrationSignalProcessingResult,
    evaluate_vibration_processor,
)


def _generate_sine_wave(
    freq_hz: float = 120.0,
    amplitude: float = 5.0,
    duration_s: float = 1.0,
    fs_hz: float = 2048.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Helper to generate a clean synthetic sine wave."""
    t = np.linspace(0, duration_s, int(fs_hz * duration_s), endpoint=False)
    signal = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    if noise_std > 0:
        np.random.seed(42)
        signal += np.random.normal(0, noise_std, len(t))
    return signal


class TestVibrationProcessor:
    """Unit test suite for Original Module 9 Vibration Processor."""

    def test_time_domain_statistics(self) -> None:
        processor = VibrationProcessor()
        # Simple array: [1.0, 3.0, 5.0, 7.0]
        samples = [1.0, 3.0, 5.0, 7.0]
        feat = processor.compute_axis_features(samples, fs_hz=2048.0, axis_name="X")

        assert feat.valid
        assert math.isclose(feat.mean, 4.0, rel_tol=1e-5)
        # RMS = sqrt((1+9+25+49)/4) = sqrt(84/4) = sqrt(21) ≈ 4.58257
        assert math.isclose(feat.rms, math.sqrt(21.0), rel_tol=1e-4)
        assert math.isclose(feat.peak, 7.0, rel_tol=1e-5)
        assert math.isclose(feat.peak_to_peak, 6.0, rel_tol=1e-5)
        # Var(ddof=1) of [1,3,5,7] is 6.66667, Std is 2.58198
        assert math.isclose(feat.variance, 20.0 / 3.0, rel_tol=1e-4)
        assert math.isclose(feat.std_dev, math.sqrt(20.0 / 3.0), rel_tol=1e-4)
        # Crest Factor = Peak / RMS = 7 / sqrt(21) ≈ 1.5275
        assert math.isclose(feat.crest_factor, 7.0 / math.sqrt(21.0), rel_tol=1e-4)

    def test_fft_known_sine_wave_detection(self) -> None:
        processor = VibrationProcessor()
        fs = 2048.0
        target_freq = 150.0  # 150 Hz tone
        amplitude = 10.0
        samples = _generate_sine_wave(freq_hz=target_freq, amplitude=amplitude, duration_s=1.0, fs_hz=fs)

        feat = processor.compute_axis_features(samples, fs_hz=fs, axis_name="X")
        assert feat.valid
        # Dominant frequency should match 150 Hz within resolution (2048/2048 = 1 Hz)
        assert math.isclose(feat.dominant_frequency_hz, target_freq, abs_tol=1.5)
        assert feat.dominant_amplitude > 0.0
        assert feat.spectral_energy > 0.0

    def test_rpm_order_tracking(self) -> None:
        processor = VibrationProcessor()
        fs = 2048.0
        target_freq = 200.0
        samples = _generate_sine_wave(freq_hz=target_freq, duration_s=1.0, fs_hz=fs)

        # 3000 RPM -> f_rot = 3000 / 60 = 50 Hz. Order = 200 / 50 = 4.0 (4th engine order)
        feat = processor.compute_axis_features(samples, fs_hz=fs, axis_name="Z", rpm=3000.0)
        assert feat.valid
        assert feat.dominant_order is not None
        assert math.isclose(feat.dominant_order, 4.0, abs_tol=0.1)

    def test_zero_or_missing_rpm_order_handling(self) -> None:
        processor = VibrationProcessor()
        samples = [1.0, 2.0, 3.0, 4.0]
        feat_zero = processor.compute_axis_features(samples, fs_hz=2048.0, rpm=0.0)
        assert feat_zero.dominant_order is None

        feat_none = processor.compute_axis_features(samples, fs_hz=2048.0, rpm=None)
        assert feat_none.dominant_order is None

    def test_multi_axis_independent_processing(self) -> None:
        processor = VibrationProcessor()
        fs = 2048.0
        x_s = _generate_sine_wave(freq_hz=100.0, fs_hz=fs)
        z_s = _generate_sine_wave(freq_hz=400.0, fs_hz=fs)

        # Y samples invalid/empty
        state, res = processor.process_window(x_s, [], z_s, fs_hz=fs, rpm=3000.0)

        assert isinstance(state, VibrationState)
        assert isinstance(res, VibrationSignalProcessingResult)
        assert res.axis_features["X"].valid
        assert not res.axis_features["Y"].valid
        assert res.axis_features["Z"].valid

        assert math.isclose(res.axis_features["X"].dominant_frequency_hz, 100.0, abs_tol=1.5)
        assert math.isclose(res.axis_features["Z"].dominant_frequency_hz, 400.0, abs_tol=1.5)

    def test_insufficient_or_empty_samples(self) -> None:
        processor = VibrationProcessor()
        feat = processor.compute_axis_features([], fs_hz=2048.0)

        assert not feat.valid
        assert feat.rms == 0.0
        assert feat.dominant_frequency_hz == 0.0

    def test_nan_inf_protection(self) -> None:
        processor = VibrationProcessor()
        bad_samples = [1.0, float("nan"), 3.0, float("inf")]
        feat = processor.compute_axis_features(bad_samples, fs_hz=2048.0)

        assert not feat.valid
        assert feat.rms == 0.0

    def test_evaluate_record_convenience(self) -> None:
        def _ch(val: float, valid: bool = True) -> ChannelValue:
            return ChannelValue(
                value=val,
                provenance=Provenance.DERIVED,
                valid=valid,
                validity_reason=ChannelValidity.VALID if valid else ChannelValidity.INVALID_RANGE,
                quality=1.0 if valid else 0.0,
            )

        rec = NormalizedSignalRecord(
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            source_id="test_source",
            provenance=Provenance.DERIVED,
            integrity_hash="test_hash",
            rpm=_ch(4000.0),
            map_pressure=_ch(120000.0),
            throttle_position=_ch(50.0),
            egt_cyl_1=_ch(900.0),
            egt_cyl_2=_ch(900.0),
            egt_cyl_3=_ch(900.0),
            egt_cyl_4=_ch(900.0),
            cht_cyl_1=_ch(380.0),
            cht_cyl_2=_ch(380.0),
            cht_cyl_3=_ch(380.0),
            cht_cyl_4=_ch(380.0),
            oil_temp=_ch(360.0),
            oil_pressure=_ch(400000.0),
            coolant_temp=_ch(360.0),
            fuel_flow=_ch(0.005),
            fuel_pressure=_ch(350000.0),
            intake_air_temp=_ch(295.0),
            ambient_pressure=_ch(101325.0),
            ambient_temp=_ch(288.15),
            voltage=_ch(13.8),
            current=_ch(15.0),
            vibration_x=_ch(2.5),
            vibration_y=_ch(1.5),
            vibration_z=_ch(9.8),
            vibration_rms=_ch(10.2),
            propeller_speed=_ch(1656.0),
            boost_pressure=_ch(18675.0),
            wastegate_duty=_ch(40.0),
            lambda_sensor=_ch(1.0),
            ignition_timing_cyl_1=_ch(25.0),
            ignition_timing_cyl_2=_ch(25.0),
            ignition_timing_cyl_3=_ch(25.0),
            ignition_timing_cyl_4=_ch(25.0),
            altitude=_ch(1000.0),
            engine_hours=_ch(10.0),
        )

        state, res = evaluate_vibration_processor(rec)
        assert isinstance(state, VibrationState)
        assert state.provenance == Provenance.DERIVED
        assert state.rms_x_m_s2.valid
        assert math.isclose(state.rms_x_m_s2.value, 2.5, rel_tol=1e-5)
