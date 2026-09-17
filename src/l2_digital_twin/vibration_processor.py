"""
Vibration Processing Subsystem — Original Module 9.

Fifth processing stage of L2 Digital Twin.
Performs deterministic signal processing (time-domain feature extraction,
FFT spectral analysis, windowing, and RPM order tracking) on accelerometer telemetry.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (Module 5) or explicit accelerometer signal buffer
    - Output: VibrationState (canonical domain schema) & VibrationSignalProcessingResult
    - Zero simulator internals or ground truth dependencies
    - Zero misfire, combustion stability, ML fault classification, health index, or advisory logic
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, VibrationConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import ProvenanceTaggedValue, VibrationState, make_tagged
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)


class AxisVibrationFeatures(BaseModel):
    """Time-domain and spectral features for a single accelerometer axis."""

    axis: str
    valid: bool
    mean: float
    rms: float
    peak: float
    peak_to_peak: float
    std_dev: float
    variance: float
    crest_factor: float
    dominant_frequency_hz: float
    dominant_amplitude: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_energy: float
    band_energy_low: float
    band_energy_mid: float
    band_energy_high: float
    dominant_order: float | None = None

    model_config = ConfigDict(frozen=True)


class VibrationSignalProcessingResult(BaseModel):
    """Complete multi-axis vibration analysis result container."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    sampling_frequency_hz: float
    window_size: int
    axis_features: dict[str, AxisVibrationFeatures]
    overall_rms_m_s2: float
    overall_peak_m_s2: float
    overall_crest_factor: float
    dominant_frequency_hz: float
    dominant_amplitude_m_s2: float
    dominant_order: float | None = None

    model_config = ConfigDict(frozen=True)


class VibrationProcessor:
    """Deterministic vibration signal processor for time-domain and FFT spectral analysis."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: VibrationConfig = self._settings.vibration

    def _apply_window(self, signal: np.ndarray, window_type: str) -> tuple[np.ndarray, float]:
        """Apply windowing function and compute coherent gain scale factor."""
        n = len(signal)
        if n == 0:
            return signal, 1.0

        w_type = window_type.lower()
        if w_type == "hanning":
            w = np.hanning(n)
        elif w_type == "hamming":
            w = np.hamming(n)
        else:
            w = np.ones(n)

        w_scale = float(np.mean(w)) if np.mean(w) > 0 else 1.0
        return signal * w, w_scale

    def compute_axis_features(
        self,
        samples: Sequence[float],
        fs_hz: float,
        axis_name: str = "X",
        rpm: float | None = None,
    ) -> AxisVibrationFeatures:
        """Compute time-domain and FFT spectral features for a single sample array."""
        if len(samples) == 0 or fs_hz <= 0 or not all(math.isfinite(s) for s in samples):
            return AxisVibrationFeatures(
                axis=axis_name,
                valid=False,
                mean=0.0,
                rms=0.0,
                peak=0.0,
                peak_to_peak=0.0,
                std_dev=0.0,
                variance=0.0,
                crest_factor=0.0,
                dominant_frequency_hz=0.0,
                dominant_amplitude=0.0,
                spectral_centroid_hz=0.0,
                spectral_bandwidth_hz=0.0,
                spectral_energy=0.0,
                band_energy_low=0.0,
                band_energy_mid=0.0,
                band_energy_high=0.0,
                dominant_order=None,
            )

        arr = np.array(samples, dtype=float)
        n = len(arr)

        # 1. Time-domain statistics
        mean_val = float(np.mean(arr))
        detrended = arr - mean_val
        var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.0
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        rms_val = float(np.sqrt(np.mean(arr ** 2)))
        peak_val = float(np.max(np.abs(arr)))
        p2p_val = float(np.max(arr) - np.min(arr))
        cf_val = (peak_val / rms_val) if rms_val > 0 else 0.0

        # 2. Spectral (FFT) analysis
        windowed, w_scale = self._apply_window(detrended, self._cfg.window_function)
        fft_complex = np.fft.rfft(windowed)
        n_half = len(fft_complex)

        mags = (2.0 * np.abs(fft_complex)) / (n * w_scale)
        if n_half > 0:
            mags[0] = np.abs(fft_complex[0]) / n

        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

        ac_mags = mags[1:] if n_half > 1 else mags
        ac_freqs = freqs[1:] if n_half > 1 else freqs

        if len(ac_mags) > 0 and np.max(ac_mags) > 0:
            max_idx = int(np.argmax(ac_mags))
            dom_freq = float(ac_freqs[max_idx])
            dom_amp = float(ac_mags[max_idx])

            sum_mag = float(np.sum(ac_mags))
            if sum_mag > 0:
                spec_centroid = float(np.sum(ac_freqs * ac_mags) / sum_mag)
                spec_bandwidth = float(np.sqrt(np.sum(((ac_freqs - spec_centroid) ** 2) * ac_mags) / sum_mag))
            else:
                spec_centroid = 0.0
                spec_bandwidth = 0.0

            spec_energy = float(np.sum(ac_mags ** 2))

            low_mask = (ac_freqs >= 0.0) & (ac_freqs < self._cfg.freq_band_low_max_hz)
            mid_mask = (ac_freqs >= self._cfg.freq_band_low_max_hz) & (ac_freqs < self._cfg.freq_band_mid_max_hz)
            high_mask = (ac_freqs >= self._cfg.freq_band_mid_max_hz) & (ac_freqs <= self._cfg.freq_band_high_max_hz)

            band_low = float(np.sum(ac_mags[low_mask] ** 2)) if np.any(low_mask) else 0.0
            band_mid = float(np.sum(ac_mags[mid_mask] ** 2)) if np.any(mid_mask) else 0.0
            band_high = float(np.sum(ac_mags[high_mask] ** 2)) if np.any(high_mask) else 0.0
        else:
            dom_freq = 0.0
            dom_amp = 0.0
            spec_centroid = 0.0
            spec_bandwidth = 0.0
            spec_energy = 0.0
            band_low = 0.0
            band_mid = 0.0
            band_high = 0.0

        # 3. Order context
        dom_order: float | None = None
        if rpm is not None and rpm > 0:
            f_rot = rpm / 60.0
            if f_rot > 0:
                dom_order = dom_freq / f_rot

        return AxisVibrationFeatures(
            axis=axis_name,
            valid=True,
            mean=mean_val,
            rms=rms_val,
            peak=peak_val,
            peak_to_peak=p2p_val,
            std_dev=std_val,
            variance=var_val,
            crest_factor=cf_val,
            dominant_frequency_hz=dom_freq,
            dominant_amplitude=dom_amp,
            spectral_centroid_hz=spec_centroid,
            spectral_bandwidth_hz=spec_bandwidth,
            spectral_energy=spec_energy,
            band_energy_low=band_low,
            band_energy_mid=band_mid,
            band_energy_high=band_high,
            dominant_order=dom_order,
        )

    def process_window(
        self,
        x_samples: Sequence[float],
        y_samples: Sequence[float],
        z_samples: Sequence[float],
        fs_hz: float | None = None,
        rpm: float | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[VibrationState, VibrationSignalProcessingResult]:
        """Process a multi-axis sample window."""
        fs = fs_hz if (fs_hz is not None and fs_hz > 0) else self._cfg.sampling_frequency_hz
        ts = timestamp or datetime.now()

        x_feat = self.compute_axis_features(x_samples, fs, "X", rpm)
        y_feat = self.compute_axis_features(y_samples, fs, "Y", rpm)
        z_feat = self.compute_axis_features(z_samples, fs, "Z", rpm)

        if x_feat.valid and y_feat.valid and z_feat.valid and len(x_samples) == len(y_samples) == len(z_samples):
            rms_arr = np.sqrt(np.array(x_samples)**2 + np.array(y_samples)**2 + np.array(z_samples)**2)
            comb_feat = self.compute_axis_features(rms_arr, fs, "RMS", rpm)
        else:
            comb_feat = self.compute_axis_features([], fs, "RMS", rpm)

        axis_map = {"X": x_feat, "Y": y_feat, "Z": z_feat, "RMS": comb_feat}

        valid_feats = [f for f in (x_feat, y_feat, z_feat) if f.valid]
        if valid_feats:
            top_feat = max(valid_feats, key=lambda f: f.dominant_amplitude)
            dom_freq = top_feat.dominant_frequency_hz
            dom_amp = top_feat.dominant_amplitude
            dom_order = top_feat.dominant_order
        else:
            dom_freq = 0.0
            dom_amp = 0.0
            dom_order = None

        result = VibrationSignalProcessingResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            sampling_frequency_hz=fs,
            window_size=len(x_samples),
            axis_features=axis_map,
            overall_rms_m_s2=comb_feat.rms if comb_feat.valid else 0.0,
            overall_peak_m_s2=comb_feat.peak if comb_feat.valid else 0.0,
            overall_crest_factor=comb_feat.crest_factor if comb_feat.valid else 0.0,
            dominant_frequency_hz=dom_freq,
            dominant_amplitude_m_s2=dom_amp,
            dominant_order=dom_order,
        )

        state = VibrationState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            rms_x_m_s2=make_tagged(x_feat.rms, Provenance.DERIVED, valid=x_feat.valid, quality=1.0 if x_feat.valid else 0.0),
            rms_y_m_s2=make_tagged(y_feat.rms, Provenance.DERIVED, valid=y_feat.valid, quality=1.0 if y_feat.valid else 0.0),
            rms_z_m_s2=make_tagged(z_feat.rms, Provenance.DERIVED, valid=z_feat.valid, quality=1.0 if z_feat.valid else 0.0),
            overall_rms_m_s2=make_tagged(comb_feat.rms, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            peak_m_s2=make_tagged(comb_feat.peak, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            crest_factor=make_tagged(comb_feat.crest_factor, Provenance.DERIVED, valid=comb_feat.valid, quality=1.0 if comb_feat.valid else 0.0),
            dominant_freq_hz=make_tagged(dom_freq, Provenance.DERIVED, valid=bool(valid_feats), quality=1.0 if valid_feats else 0.0),
            dominant_amplitude_m_s2=make_tagged(dom_amp, Provenance.DERIVED, valid=bool(valid_feats), quality=1.0 if valid_feats else 0.0),
            dominant_order=make_tagged(dom_order, Provenance.DERIVED, valid=(dom_order is not None), quality=1.0 if dom_order is not None else 0.0),
        )

        return state, result

    def process_record(
        self, record: NormalizedSignalRecord
    ) -> tuple[VibrationState, VibrationSignalProcessingResult]:
        """Process instantaneous telemetry record accelerometer channels."""
        x_val = record.vibration_x.value if record.vibration_x.valid else 0.0
        y_val = record.vibration_y.value if record.vibration_y.valid else 0.0
        z_val = record.vibration_z.value if record.vibration_z.valid else 0.0

        rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value > 0) else None

        return self.process_window(
            [x_val] if record.vibration_x.valid else [],
            [y_val] if record.vibration_y.valid else [],
            [z_val] if record.vibration_z.valid else [],
            fs_hz=self._cfg.sampling_frequency_hz,
            rpm=rpm_val,
            timestamp=record.timestamp,
        )


def evaluate_vibration_processor(
    record: NormalizedSignalRecord,
    settings: AppSettings | None = None,
) -> tuple[VibrationState, VibrationSignalProcessingResult]:
    """Convenience function for Module 9 vibration processing on a telemetry record."""
    processor = VibrationProcessor(settings)
    return processor.process_record(record)
