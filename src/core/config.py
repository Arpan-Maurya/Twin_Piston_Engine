"""
Config — Externalized configuration system using Pydantic Settings.

Loads configuration from config/default.yaml with environment variable
overrides. No physical constant or threshold is hardcoded in business
logic — everything comes from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.provenance import CommunicationState, DeploymentRole, OverflowPolicy


# =============================================================================
# Configuration Sub-Models
# =============================================================================


class EngineConfig(BaseModel):
    """Rotax 915 iS engine geometric and operational parameters."""

    name: str = "Rotax 915 iS"
    bore_mm: float = 84.0
    stroke_mm: float = 61.0
    displacement_cc: float = 1352.0
    num_cylinders: int = 4
    compression_ratio: float = 10.5
    rated_power_kw: float = 105.0
    rated_rpm: int = 5800
    max_rpm: int = 5800
    firing_order: list[int] = Field(default_factory=lambda: [1, 3, 2, 4])
    connecting_rod_mm: float = 105.0
    intake_valve_close_btdc_deg: float = 50.0
    exhaust_valve_open_bbdc_deg: float = 55.0
    piston_mass_kg: float = 0.35
    conrod_mass_kg: float = 0.30
    fuel_type: str = "avgas_100ll"
    lhv_mj_per_kg: float = 43.5
    stoichiometric_afr: float = 14.7


class TurbochargerConfig(BaseModel):
    """Turbocharger operational parameters."""

    max_boost_bar: float = 1.45
    wastegate_target_bar: float = 1.35
    compressor_efficiency: float = 0.72
    turbine_efficiency: float = 0.68
    compressor_pr_max: float = 2.5
    turbine_pr_max: float = 2.0
    shaft_inertia_kg_m2: float = 0.00005
    wastegate_kp: float = 2.0
    wastegate_ki: float = 0.5


class LubricationConfig(BaseModel):
    """Lubrication system parameters."""

    oil_type: str = "AeroShell Sport Plus 4"
    nominal_pressure_bar: float = 4.0
    min_pressure_bar: float = 1.5
    max_pressure_bar: float = 7.0
    warn_low_pressure_bar: float = 2.0
    crit_low_pressure_bar: float = 1.5
    warn_high_pressure_bar: float = 6.0
    crit_high_pressure_bar: float = 7.0
    nominal_temp_c: float = 90.0
    max_temp_c: float = 130.0
    min_temp_c: float = 40.0
    warn_high_temp_c: float = 120.0
    crit_high_temp_c: float = 130.0
    warn_low_temp_c: float = 50.0
    crit_low_temp_c: float = 40.0
    vogel_a: float = 0.0002
    vogel_b: float = 1200.0
    vogel_c: float = 140.0


class CoolingConfig(BaseModel):
    """Cooling system parameters."""

    coolant_type: str = "50/50 ethylene glycol"
    thermostat_open_c: float = 85.0
    thermostat_full_open_c: float = 95.0
    nominal_cht_c: float = 100.0
    max_cht_c: float = 135.0
    coolant_flow_rate_kg_s: float = 1.0
    coolant_specific_heat_j_kg_k: float = 3500.0
    radiator_effectiveness: float = 0.65


class TelemetryConfig(BaseModel):
    """Telemetry acquisition settings."""

    sample_rate_hz: float = 1.0
    num_channels: int = 38


class SensorNoiseConfig(BaseModel):
    """Simulator sensor noise standard deviations."""

    rpm_std: float = 5.0
    temperature_std_k: float = 1.5
    pressure_std_pa: float = 500.0
    vibration_std_m_s2: float = 0.2
    fuel_flow_std_kg_s: float = 0.0001
    voltage_std_v: float = 0.05
    lambda_std: float = 0.005


class SimulatorConfig(BaseModel):
    """Physics simulator configuration."""

    time_step_s: float = 0.001
    cycles_per_output: int = 1
    default_rpm: float = 4000.0
    default_throttle_pct: float = 50.0
    default_altitude_m: float = 1000.0
    default_ambient_temp_k: float = 288.15
    default_ambient_pressure_pa: float = 101325.0
    sensor_noise: SensorNoiseConfig = Field(default_factory=SensorNoiseConfig)
    sensor_dropout_probability: float = 0.001


class PipelineConfig(BaseModel):
    """Main processing pipeline settings."""

    cycle_timeout_ms: int = 500
    max_invalid_channels_before_halt: int = 10
    residual_window_size: int = 60
    enable_persistence: bool = True
    persistence_interval_s: float = 5.0


class HealthIndexWeights(BaseModel):
    """Weights for health index aggregation."""

    thermal: float = 0.25
    mechanical: float = 0.25
    combustion: float = 0.25
    lubrication: float = 0.15
    vibration: float = 0.10


class DegradationThresholds(BaseModel):
    """Health index thresholds for degradation state classification."""

    healthy: float = 0.8
    watch: float = 0.6
    caution: float = 0.4
    warning: float = 0.2
    critical: float = 0.0


class MLConfig(BaseModel):
    """Machine learning pipeline configuration."""

    anomaly_threshold: float = 0.85
    fault_confidence_threshold: float = 0.70
    rul_horizon_hours: float = 500.0
    health_index_weights: HealthIndexWeights = Field(default_factory=HealthIndexWeights)
    degradation_states: DegradationThresholds = Field(default_factory=DegradationThresholds)


class AdvisorySeverityConfig(BaseModel):
    """Health-index thresholds for advisory severity levels."""

    monitor: float = 0.7
    schedule_maintenance: float = 0.5
    immediate_inspection: float = 0.3
    ground_aircraft: float = 0.1


class AdvisoryConfig(BaseModel):
    """Advisory layer configuration."""

    severity_levels: AdvisorySeverityConfig = Field(default_factory=AdvisorySeverityConfig)


class ChannelRangeConfig(BaseModel):
    """Min/max range for a single channel."""

    min: float
    max: float


class SecurityConfig(BaseModel):
    """Telemetry integrity and security configuration."""

    hmac_algorithm: str = "sha256"
    secret_key_env_var: str = "TELEMETRY_SECRET_KEY"
    default_secret_key: str = "dev_prototype_secret_key_change_in_prod"
    require_signature: bool = True
    max_sequence_gap: int = 1000
    stale_timeout_s: float = 5.0
    per_channel_stale_timeouts_s: dict[str, float] = Field(
        default_factory=lambda: {
            "egt_cyl1_hot_uv": 3.0,
            "egt_cyl2_hot_uv": 3.0,
            "egt_cyl3_hot_uv": 3.0,
            "egt_cyl4_hot_uv": 3.0,
            "crank_period_us": 2.0,
            "oil_p_counts": 3.0,
        }
    )


class PersistenceConfig(BaseModel):
    """Raw telemetry persistence configuration."""

    enabled: bool = True
    storage_backend: str = "in_memory"
    storage_path: str = "data/raw_telemetry.db"
    max_records: int = 100000


class SensorCalibrationConfig(BaseModel):
    """Sensor calibration coefficients and constants for inverse modelling."""

    type_k_uv_per_c: float = 41.27
    pt100_r0_ohms: float = 100.0
    pt100_alpha_per_c: float = 0.00385
    map_full_scale_pa: float = 200000.0
    oil_p_full_scale_pa: float = 800000.0
    fuel_flow_kg_s_per_hz: float = 0.0001
    accel_counts_per_m_s2: float = 10.0


class EGTDiagnosticConfig(BaseModel):
    """Configuration for per-cylinder EGT diagnostics."""

    egt_max_k: float = 1200.0
    egt_warning_temp_k: float = 1123.15      # 850°C
    egt_critical_temp_k: float = 1173.15     # 900°C
    egt_spread_warning_k: float = 50.0
    egt_spread_critical_k: float = 80.0
    egt_dev_warning_k: float = 35.0
    egt_dev_critical_k: float = 60.0
    egt_rate_warning_k_s: float = 15.0
    egt_rate_critical_k_s: float = 30.0


class VibrationConfig(BaseModel):
    """Vibration signal processing configuration."""

    sampling_frequency_hz: float = 2048.0
    window_size: int = 2048
    overlap_pct: float = 0.0
    window_function: str = "hanning"
    freq_band_low_max_hz: float = 100.0
    freq_band_mid_max_hz: float = 500.0
    freq_band_high_max_hz: float = 1000.0


class MisfireConfig(BaseModel):
    """Configuration for combustion stability and misfire diagnostics."""

    egt_drop_threshold_k: float = 50.0
    egt_dev_threshold_k: float = 40.0
    egt_rate_drop_threshold_k_s: float = 20.0
    rpm_std_threshold: float = 50.0
    alpha_std_threshold: float = 15.0
    vibration_rms_threshold_m_s2: float = 15.0
    misfire_confidence_threshold: float = 0.70
    unstable_confidence_threshold: float = 0.40
    egt_evidence_weight: float = 0.40
    crank_evidence_weight: float = 0.35
    vibration_evidence_weight: float = 0.25


class HealthyBaselineConfig(BaseModel):
    """Healthy expectation baseline parameters and normalization scales."""

    base_egt_k: float = 950.0
    k_egt_map_pa: float = 0.002
    k_egt_rpm: float = 0.05
    base_oil_temp_rise_k: float = 65.0
    base_vibration_rms_m_s2: float = 5.0
    k_vib_rpm: float = 10.0
    scale_egt_k: float = 50.0
    scale_oil_pressure_pa: float = 50000.0
    scale_oil_temp_k: float = 10.0
    scale_vibration_m_s2: float = 5.0
    scale_power_kw: float = 15.0


class HealthConfig(BaseModel):
    """Configuration for engine health index and degradation supervision."""

    weight_thermal: float = 0.20
    weight_lubrication: float = 0.20
    weight_vibration: float = 0.20
    weight_combustion: float = 0.15
    weight_performance: float = 0.10
    weight_anomaly_fault: float = 0.15

    # Thresholds for DegradationState
    healthy_threshold: float = 0.85
    watch_threshold: float = 0.70
    caution_threshold: float = 0.50
    warning_threshold: float = 0.30

    # Hysteresis delta (prevents flickering near state boundaries)
    hysteresis_delta: float = 0.03

    # Minimum quality threshold required for valid health assessment
    min_evidence_quality: float = 0.20


class RULConfig(BaseModel):
    """Configuration for RUL prediction and history tracking."""

    min_history_samples: int = 3
    max_prediction_horizon_hours: float = 2000.0
    default_operating_assumption: str = "constant_cruise_operating_profile"
    confidence_level: float = 0.95
    history_window_max_samples: int = 1000
    baseline_degradation_target_hi: float = 0.20


class MissionConfig(BaseModel):
    """Configuration for mission phase classification and mission risk assessment."""

    # Phase detection thresholds
    ground_max_rpm: float = 1200.0
    ground_max_throttle_pct: float = 15.0
    takeoff_min_rpm: float = 5000.0
    takeoff_min_map_pa: float = 110000.0
    climb_min_rpm: float = 4500.0
    climb_min_vrate_m_s: float = 1.0
    descent_max_rpm: float = 3800.0
    descent_max_vrate_m_s: float = -1.0
    landing_max_altitude_m: float = 200.0

    # Phase risk multipliers
    phase_risk_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "TAKEOFF": 1.5,
            "CLIMB": 1.3,
            "LANDING": 1.2,
            "CRUISE": 1.0,
            "DESCENT": 1.0,
            "GROUND": 0.5,
            "UNKNOWN": 1.2,
        }
    )

    # Risk level thresholds
    low_risk_threshold: float = 0.25
    moderate_risk_threshold: float = 0.50
    high_risk_threshold: float = 0.75

    # Phase transition debounce / minimum persistence samples
    phase_debounce_samples: int = 2


class DeploymentConfig(BaseModel):
    """Deployment role and transport settings for Edge/Ground partition (Module 21)."""

    role: DeploymentRole = DeploymentRole.SIMULATION
    transport_backend: str = "in_memory"
    endpoint_url: str = "http://localhost:8000/api/v1/telemetry"
    timeout_s: float = 5.0
    retry_attempts: int = 3
    retry_backoff_s: float = 1.0
    buffer_capacity: int = 1000
    buffer_overflow_policy: OverflowPolicy = OverflowPolicy.DISCARD_OLDEST
    reconnect_interval_s: float = 2.0
    freshness_threshold_s: float = 5.0


# =============================================================================
# Top-Level Settings
# =============================================================================


class AppSettings(BaseSettings):
    """Top-level application settings.

    Load order (highest priority wins):
        1. Environment variables (prefixed APP_)
        2. .env file
        3. config/default.yaml

    Usage:
        settings = load_settings()
        bore = settings.engine.bore_mm
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Sub-configs ---
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    turbocharger: TurbochargerConfig = Field(default_factory=TurbochargerConfig)
    lubrication: LubricationConfig = Field(default_factory=LubricationConfig)
    cooling: CoolingConfig = Field(default_factory=CoolingConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    advisory: AdvisoryConfig = Field(default_factory=AdvisoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    sensor_calibration: SensorCalibrationConfig = Field(default_factory=SensorCalibrationConfig)
    egt_diagnostics: EGTDiagnosticConfig = Field(default_factory=EGTDiagnosticConfig)
    vibration: VibrationConfig = Field(default_factory=VibrationConfig)
    misfire: MisfireConfig = Field(default_factory=MisfireConfig)
    healthy_baseline: HealthyBaselineConfig = Field(default_factory=HealthyBaselineConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    rul: RULConfig = Field(default_factory=RULConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    channel_ranges: dict[str, ChannelRangeConfig] = Field(default_factory=dict)
    raw_channel_ranges: dict[str, ChannelRangeConfig] = Field(default_factory=dict)

    # --- Application-level ---
    version: str = "1.0.0"
    app_env: str = "development"
    log_level: str = "INFO"
    config_path: str = "config/default.yaml"
    database_url: str = "sqlite+aiosqlite:///./digital_twin.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    telemetry_source: str = "simulator"
    csv_replay_path: str = "data/sample_flights/"
    model_dir: str = "models/"


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Returns an empty dict if the file does not exist, so the system
    can start with pure defaults or environment variables.
    """
    config_file = Path(path)
    if config_file.exists():
        with config_file.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def load_settings(config_path: str | Path = "config/default.yaml") -> AppSettings:
    """Load and validate application settings.

    Merges YAML file values with environment variable overrides.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppSettings instance.

    Raises:
        ConfigurationError: If configuration values fail validation.
    """
    from pydantic import ValidationError
    from src.core.exceptions import ConfigurationError

    yaml_config = _load_yaml_config(config_path)
    try:
        settings = AppSettings(**yaml_config)
        # Custom sanity validation
        if settings.engine.bore_mm <= 0 or settings.engine.num_cylinders <= 0:
            raise ConfigurationError("Engine bore and cylinder count must be positive")
        if settings.telemetry.sample_rate_hz <= 0:
            raise ConfigurationError("Sample rate must be positive")
        if settings.app_env.lower() in ("production", "prod"):
            import os
            env_secret = os.environ.get(settings.security.secret_key_env_var)
            if not env_secret or env_secret == "dev_prototype_secret_key_change_in_prod":
                raise ConfigurationError(
                    f"Production mode requires a secret key injected via environment variable '{settings.security.secret_key_env_var}'. The default development secret key is prohibited in production."
                )
        return settings
    except (ValidationError, ValueError) as e:
        raise ConfigurationError(f"Failed to validate configuration from {config_path}: {e}") from e



# Module-level singleton (lazily initialized)
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get the global settings singleton.

    Initializes from default config path on first call.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings singleton (useful for testing)."""
    global _settings  # noqa: PLW0603
    _settings = None
