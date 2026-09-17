"""
Sensor Forward & Conversion Models — Bridge between acquisition-level raw values
and engineering physical units.

Forward Path: Normalized telemetry / simulator state → RawSignalRecord
Inverse Path: RawSignalRecord → Normalized engineering values
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.core.provenance import ChannelValidity, Provenance
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord

# Sensor conversion constants
TYPE_K_UV_PER_C = 41.27           # Type K thermocouple sensitivity (~41.27 uV/°C)
PT100_R0 = 100.0                  # Pt100 RTD nominal resistance at 0°C [ohms]
PT100_ALPHA = 0.00385             # Pt100 RTD temperature coefficient [1/°C]
ADC_12BIT_MAX = 4095              # 12-bit ADC max counts
MAP_FULL_SCALE_PA = 200000.0      # 200 kPa full scale for MAP sensor
OIL_P_FULL_SCALE_PA = 800000.0    # 800 kPa full scale for oil pressure sensor
FUEL_FLOW_KG_S_PER_HZ = 0.0001    # Fuel flow turbine sensor gain


def normalized_to_raw(
    record: NormalizedSignalRecord,
    sequence_number: int | None = None,
    source_type: Provenance | None = None,
) -> RawSignalRecord:
    """Convert a NormalizedSignalRecord into a RawSignalRecord (Sensor-Forward Path).

    Maps physical quantities (RPM, Pa, K) into acquisition raw signals
    (uV, ohms, ADC counts, us, Hz).
    """
    seq = sequence_number if sequence_number is not None else record.sequence_number
    src = source_type if source_type is not None else record.provenance

    # Cold junction temperature reference (°C)
    egt_cold_c = 25.0
    cht_cold_c = 25.0

    # Exhaust gas temperatures (K -> °C -> uV)
    egt1_c = record.egt_cyl_1.value - 273.15
    egt2_c = record.egt_cyl_2.value - 273.15
    egt3_c = record.egt_cyl_3.value - 273.15
    egt4_c = record.egt_cyl_4.value - 273.15

    egt1_uv = max(0.0, (egt1_c - egt_cold_c) * TYPE_K_UV_PER_C)
    egt2_uv = max(0.0, (egt2_c - egt_cold_c) * TYPE_K_UV_PER_C)
    egt3_uv = max(0.0, (egt3_c - egt_cold_c) * TYPE_K_UV_PER_C)
    egt4_uv = max(0.0, (egt4_c - egt_cold_c) * TYPE_K_UV_PER_C)

    # Cylinder head temperature (K -> °C -> uV)
    cht1_c = record.cht_cyl_1.value - 273.15
    cht_uv = max(0.0, (cht1_c - cht_cold_c) * TYPE_K_UV_PER_C)

    # Oil temperature (K -> °C -> RTD Ohms)
    oil_temp_c = record.oil_temp.value - 273.15
    oil_rtd_ohms = max(10.0, PT100_R0 * (1.0 + PT100_ALPHA * oil_temp_c))

    # Pressures to ADC counts
    oil_p_counts = int(min(max(0.0, (record.oil_pressure.value / OIL_P_FULL_SCALE_PA) * ADC_12BIT_MAX), ADC_12BIT_MAX))
    map_counts = int(min(max(0.0, (record.map_pressure.value / MAP_FULL_SCALE_PA) * ADC_12BIT_MAX), ADC_12BIT_MAX))
    adc_vref_counts = ADC_12BIT_MAX

    # RPM to Crank Period in microseconds
    rpm = max(1.0, record.rpm.value)
    crank_period_us = (60.0 * 1e6) / rpm

    # Fuel flow to pulse frequency Hz
    fuel_pulse_hz = max(0.0, record.fuel_flow.value / FUEL_FLOW_KG_S_PER_HZ)

    # Vibration m/s² to raw accel counts
    accel_x = int(record.vibration_x.value * 10.0)
    accel_y = int(record.vibration_y.value * 10.0)
    accel_z = int(record.vibration_z.value * 10.0)

    # Ambient
    ambient_temp_c = record.ambient_temp.value - 273.15
    ambient_press_pa = record.ambient_pressure.value

    # Signal Quality
    invalid_channels = record.invalid_channel_names
    quality_score = 1.0 if not invalid_channels else max(0.0, 1.0 - len(invalid_channels) / 38.0)
    signal_quality = SignalQuality(
        score=quality_score,
        valid=len(invalid_channels) == 0,
        invalid_channels=invalid_channels,
    )

    raw_rec = RawSignalRecord(
        timestamp=record.timestamp,
        sequence_number=seq,
        source_type=src,
        egt_cyl1_hot_uv=egt1_uv,
        egt_cyl2_hot_uv=egt2_uv,
        egt_cyl3_hot_uv=egt3_uv,
        egt_cyl4_hot_uv=egt4_uv,
        egt_cold_c=egt_cold_c,
        cht_hot_uv=cht_uv,
        cht_cold_c=cht_cold_c,
        oil_rtd_ohms=oil_rtd_ohms,
        oil_p_counts=oil_p_counts,
        map_counts=map_counts,
        adc_vref_counts=adc_vref_counts,
        crank_period_us=crank_period_us,
        fuel_pulse_hz=fuel_pulse_hz,
        accel_counts_xyz=(accel_x, accel_y, accel_z),
        ambient_temp_c=ambient_temp_c,
        ambient_press_pa=ambient_press_pa,
        signal_quality=signal_quality,
    )

    return raw_rec.model_copy(update={"integrity_hash": raw_rec.compute_integrity_hash()})


def raw_to_normalized(raw_record: RawSignalRecord) -> NormalizedSignalRecord:
    """Convert a RawSignalRecord into a NormalizedSignalRecord (Sensor-Inverse Path).

    Converts raw acquisition fields (uV, ohms, ADC counts, us, Hz) back into SI
    engineering values (RPM, Pa, K).
    """
    src = raw_record.source_type

    # RPM from crank period
    rpm = (60.0 * 1e6) / max(raw_record.crank_period_us, 1.0)

    # EGT temperatures
    egt1_k = (raw_record.egt_cyl1_hot_uv / TYPE_K_UV_PER_C) + raw_record.egt_cold_c + 273.15
    egt2_k = (raw_record.egt_cyl2_hot_uv / TYPE_K_UV_PER_C) + raw_record.egt_cold_c + 273.15
    egt3_k = (raw_record.egt_cyl3_hot_uv / TYPE_K_UV_PER_C) + raw_record.egt_cold_c + 273.15
    egt4_k = (raw_record.egt_cyl4_hot_uv / TYPE_K_UV_PER_C) + raw_record.egt_cold_c + 273.15

    # CHT temperature
    cht_k = (raw_record.cht_hot_uv / TYPE_K_UV_PER_C) + raw_record.cht_cold_c + 273.15

    # Oil temperature
    oil_temp_c = (raw_record.oil_rtd_ohms - PT100_R0) / (PT100_R0 * PT100_ALPHA)
    oil_temp_k = oil_temp_c + 273.15

    # Pressures
    vref_scale = raw_record.adc_vref_counts if raw_record.adc_vref_counts > 0 else ADC_12BIT_MAX
    map_pa = (raw_record.map_counts / vref_scale) * MAP_FULL_SCALE_PA
    oil_p_pa = (raw_record.oil_p_counts / vref_scale) * OIL_P_FULL_SCALE_PA

    # Fuel flow
    fuel_flow_kg_s = raw_record.fuel_pulse_hz * FUEL_FLOW_KG_S_PER_HZ

    # Vibration
    vib_x = raw_record.accel_counts_xyz[0] / 10.0
    vib_y = raw_record.accel_counts_xyz[1] / 10.0
    vib_z = raw_record.accel_counts_xyz[2] / 10.0
    vib_rms = math.sqrt(vib_x**2 + vib_y**2 + vib_z**2)

    # Ambient
    ambient_temp_k = raw_record.ambient_temp_c + 273.15
    ambient_press_pa = raw_record.ambient_press_pa

    def _ch(val: float, valid: bool = True) -> ChannelValue:
        return ChannelValue(
            value=val,
            provenance=src,
            valid=valid,
            quality=raw_record.signal_quality.score,
        )

    return NormalizedSignalRecord(
        timestamp=raw_record.timestamp,
        sequence_number=raw_record.sequence_number,
        source_id=f"raw_conversion:{src.value}",
        provenance=src,
        integrity_hash=raw_record.integrity_hash,
        rpm=_ch(rpm),
        map_pressure=_ch(map_pa),
        throttle_position=_ch(50.0),
        egt_cyl_1=_ch(egt1_k),
        egt_cyl_2=_ch(egt2_k),
        egt_cyl_3=_ch(egt3_k),
        egt_cyl_4=_ch(egt4_k),
        cht_cyl_1=_ch(cht_k),
        cht_cyl_2=_ch(cht_k),
        cht_cyl_3=_ch(cht_k),
        cht_cyl_4=_ch(cht_k),
        oil_temp=_ch(oil_temp_k),
        oil_pressure=_ch(oil_p_pa),
        coolant_temp=_ch(363.15),
        fuel_flow=_ch(fuel_flow_kg_s),
        fuel_pressure=_ch(350000.0),
        intake_air_temp=_ch(300.15),
        ambient_pressure=_ch(ambient_press_pa),
        ambient_temp=_ch(ambient_temp_k),
        voltage=_ch(13.8),
        current=_ch(15.0),
        vibration_x=_ch(vib_x),
        vibration_y=_ch(vib_y),
        vibration_z=_ch(vib_z),
        vibration_rms=_ch(vib_rms),
        propeller_speed=_ch(rpm * 0.414),
        boost_pressure=_ch(map_pa),
        wastegate_duty=_ch(50.0),
        lambda_sensor=_ch(1.0),
        ignition_timing_cyl_1=_ch(25.0),
        ignition_timing_cyl_2=_ch(25.0),
        ignition_timing_cyl_3=_ch(25.0),
        ignition_timing_cyl_4=_ch(25.0),
        altitude=_ch(1000.0),
        engine_hours=_ch(0.0),
    )
