"""
Sensor Inverse Modelling — Original Module 5.

First processing stage of L2 Digital Twin.
Converts acquisition-level RawSignalRecord (uV, ohms, ADC counts, us, Hz) into
canonical physical engineering quantities (K, Pa, rev/min, kg/s, m/s²) wrapped in
ChannelValue with Provenance.DERIVED.

STRICT BOUNDARY CONSTRAINTS:
    - Input: RawSignalRecord (immutable)
    - Output: NormalizedSignalRecord (engineering units, Provenance.DERIVED)
    - Zero simulator internals or ground truth dependencies
    - Zero thermodynamic/mechanical digital twin derivations (Module 6)
    - Zero diagnostic, ML, advisory, or API logic
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, FlightPhase, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

logger = get_logger(__name__)


class SensorInverseModel:
    """Sensor Inverse Model for Rotax 915 iS acquisition telemetry."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def convert_raw_to_engineering(
        self, raw_record: RawSignalRecord
    ) -> NormalizedSignalRecord:
        """Convert a RawSignalRecord into a NormalizedSignalRecord.

        Emits physical engineering values with Provenance.DERIVED while preserving
        signal quality and failure isolation metadata.
        """
        cal = self._settings.sensor_calibration
        raw_sq = raw_record.signal_quality
        invalid_raw_channels = set(raw_sq.invalid_channels)

        def _make_channel(
            channel_name: str,
            eng_value: float,
            raw_channel_key: str,
            default_valid: bool = True,
            error_reason: str | None = None,
        ) -> ChannelValue:
            """Helper to create a ChannelValue with quality & error propagation."""
            is_valid = default_valid and (raw_channel_key not in invalid_raw_channels)
            validity_reason = ChannelValidity.VALID if is_valid else ChannelValidity.INVALID_RANGE

            fault_flag = None
            if not is_valid:
                fault_flag = error_reason or raw_sq.invalid_reasons.get(
                    raw_channel_key, f"Raw channel {raw_channel_key} invalid or out of bounds"
                )

            quality = raw_sq.score if is_valid else 0.0

            return ChannelValue(
                value=eng_value,
                provenance=Provenance.DERIVED,
                valid=is_valid,
                validity_reason=validity_reason,
                quality=quality,
                fault_flag=fault_flag,
            )

        # ---------------------------------------------------------------------
        # 1. Crank Period -> Engine Speed (RPM)
        # ---------------------------------------------------------------------
        rpm_valid = raw_record.crank_period_us > 0
        rpm_val = (60.0 * 1e6) / raw_record.crank_period_us if rpm_valid else 0.0
        rpm_channel = _make_channel(
            "rpm",
            rpm_val,
            "crank_period_us",
            default_valid=rpm_valid,
            error_reason="Non-positive crank period us <= 0",
        )

        # ---------------------------------------------------------------------
        # 2. Thermocouple EGT (Microvolts -> Kelvin with Cold Junction)
        # ---------------------------------------------------------------------
        def _convert_tc_egt(hot_uv: float, cold_c: float, raw_key: str) -> ChannelValue:
            valid = hot_uv >= 0.0
            egt_c = cold_c + (hot_uv / cal.type_k_uv_per_c) if valid else 0.0
            egt_k = egt_c + 273.15
            return _make_channel(
                "egt",
                egt_k,
                raw_key,
                default_valid=valid,
                error_reason="Negative thermocouple microvolts",
            )

        egt1 = _convert_tc_egt(raw_record.egt_cyl1_hot_uv, raw_record.egt_cold_c, "egt_cyl1_hot_uv")
        egt2 = _convert_tc_egt(raw_record.egt_cyl2_hot_uv, raw_record.egt_cold_c, "egt_cyl2_hot_uv")
        egt3 = _convert_tc_egt(raw_record.egt_cyl3_hot_uv, raw_record.egt_cold_c, "egt_cyl3_hot_uv")
        egt4 = _convert_tc_egt(raw_record.egt_cyl4_hot_uv, raw_record.egt_cold_c, "egt_cyl4_hot_uv")

        # ---------------------------------------------------------------------
        # 3. Thermocouple CHT (Microvolts -> Kelvin with Cold Junction)
        # ---------------------------------------------------------------------
        cht_valid = raw_record.cht_hot_uv >= 0.0
        cht_c = raw_record.cht_cold_c + (raw_record.cht_hot_uv / cal.type_k_uv_per_c) if cht_valid else 0.0
        cht_k = cht_c + 273.15
        cht_ch = _make_channel(
            "cht",
            cht_k,
            "cht_hot_uv",
            default_valid=cht_valid,
            error_reason="Negative CHT thermocouple microvolts",
        )

        # ---------------------------------------------------------------------
        # 4. Oil RTD (Resistance -> Oil Temperature Kelvin)
        # ---------------------------------------------------------------------
        rtd_valid = raw_record.oil_rtd_ohms > 0.0
        oil_temp_c = (
            (raw_record.oil_rtd_ohms - cal.pt100_r0_ohms) / (cal.pt100_r0_ohms * cal.pt100_alpha_per_c)
            if rtd_valid
            else 0.0
        )
        oil_temp_k = oil_temp_c + 273.15
        oil_temp_ch = _make_channel(
            "oil_temp",
            oil_temp_k,
            "oil_rtd_ohms",
            default_valid=rtd_valid,
            error_reason="Non-positive RTD resistance",
        )

        # ---------------------------------------------------------------------
        # 5. ADC Pressures (MAP & Oil Pressure)
        # ---------------------------------------------------------------------
        vref_scale = (
            raw_record.adc_vref_counts
            if raw_record.adc_vref_counts > 0
            else 4095
        )

        map_pa = (raw_record.map_counts / vref_scale) * cal.map_full_scale_pa
        map_ch = _make_channel("map_pressure", map_pa, "map_counts")

        oil_p_pa = (raw_record.oil_p_counts / vref_scale) * cal.oil_p_full_scale_pa
        oil_p_ch = _make_channel("oil_pressure", oil_p_pa, "oil_p_counts")

        # ---------------------------------------------------------------------
        # 6. Fuel Flow (Pulse Hz -> kg/s)
        # ---------------------------------------------------------------------
        fuel_valid = raw_record.fuel_pulse_hz >= 0.0
        fuel_kg_s = raw_record.fuel_pulse_hz * cal.fuel_flow_kg_s_per_hz if fuel_valid else 0.0
        fuel_ch = _make_channel(
            "fuel_flow",
            fuel_kg_s,
            "fuel_pulse_hz",
            default_valid=fuel_valid,
            error_reason="Negative fuel pulse frequency",
        )

        # ---------------------------------------------------------------------
        # 7. Accelerometer (Counts -> m/s²)
        # ---------------------------------------------------------------------
        accel_scale = cal.accel_counts_per_m_s2 if cal.accel_counts_per_m_s2 > 0 else 10.0
        vx = raw_record.accel_counts_xyz[0] / accel_scale
        vy = raw_record.accel_counts_xyz[1] / accel_scale
        vz = raw_record.accel_counts_xyz[2] / accel_scale
        vrms = math.sqrt(vx**2 + vy**2 + vz**2)

        vib_x_ch = _make_channel("vibration_x", vx, "accel_counts_xyz")
        vib_y_ch = _make_channel("vibration_y", vy, "accel_counts_xyz")
        vib_z_ch = _make_channel("vibration_z", vz, "accel_counts_xyz")
        vib_rms_ch = _make_channel("vibration_rms", vrms, "accel_counts_xyz")

        # ---------------------------------------------------------------------
        # 8. Ambient Conditions
        # ---------------------------------------------------------------------
        amb_t_k = raw_record.ambient_temp_c + 273.15
        amb_temp_ch = _make_channel("ambient_temp", amb_t_k, "ambient_temp_c")

        amb_press_pa = raw_record.ambient_press_pa
        amb_press_ch = _make_channel("ambient_pressure", amb_press_pa, "ambient_press_pa")

        # ---------------------------------------------------------------------
        # Auxiliary default channels (derived/default values)
        # ---------------------------------------------------------------------
        def _def_ch(val: float) -> ChannelValue:
            return ChannelValue(value=val, provenance=Provenance.DERIVED, valid=True, quality=1.0)

        coolant_ch = _def_ch(363.15)  # Nominal 90°C coolant
        fuel_press_ch = _def_ch(350000.0)
        iat_ch = _def_ch(300.15)
        voltage_ch = _def_ch(13.8)
        current_ch = _def_ch(15.0)
        prop_rpm_ch = _def_ch(rpm_val * 0.414)
        boost_press_ch = _def_ch(map_pa)
        wastegate_ch = _def_ch(50.0)
        lambda_ch = _def_ch(1.0)
        ign_ch = _def_ch(25.0)
        alt_ch = _def_ch(1000.0)
        hours_ch = _def_ch(0.0)

        return NormalizedSignalRecord(
            timestamp=raw_record.timestamp,
            sequence_number=raw_record.sequence_number,
            source_id=f"sensor_inverse:{raw_record.source_type.value}",
            provenance=Provenance.DERIVED,
            integrity_hash=raw_record.integrity_hash,
            flight_phase=FlightPhase.GROUND,
            rpm=rpm_channel,
            map_pressure=map_ch,
            throttle_position=_def_ch(50.0),
            egt_cyl_1=egt1,
            egt_cyl_2=egt2,
            egt_cyl_3=egt3,
            egt_cyl_4=egt4,
            cht_cyl_1=cht_ch,
            cht_cyl_2=cht_ch,
            cht_cyl_3=cht_ch,
            cht_cyl_4=cht_ch,
            oil_temp=oil_temp_ch,
            oil_pressure=oil_p_ch,
            coolant_temp=coolant_ch,
            fuel_flow=fuel_ch,
            fuel_pressure=fuel_press_ch,
            intake_air_temp=iat_ch,
            ambient_pressure=amb_press_ch,
            ambient_temp=amb_temp_ch,
            voltage=voltage_ch,
            current=current_ch,
            vibration_x=vib_x_ch,
            vibration_y=vib_y_ch,
            vibration_z=vib_z_ch,
            vibration_rms=vib_rms_ch,
            propeller_speed=prop_rpm_ch,
            boost_pressure=boost_press_ch,
            wastegate_duty=wastegate_ch,
            lambda_sensor=lambda_ch,
            ignition_timing_cyl_1=ign_ch,
            ignition_timing_cyl_2=ign_ch,
            ignition_timing_cyl_3=ign_ch,
            ignition_timing_cyl_4=ign_ch,
            altitude=alt_ch,
            engine_hours=hours_ch,
        )


def convert_raw_to_engineering_state(
    raw_record: RawSignalRecord,
    settings: AppSettings | None = None,
) -> NormalizedSignalRecord:
    """Convenience function for Module 5 sensor inverse modelling."""
    model = SensorInverseModel(settings)
    return model.convert_raw_to_engineering(raw_record)
