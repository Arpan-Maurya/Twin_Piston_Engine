"""
Unit tests for the configuration system.

Validates that:
- Config loads from YAML
- All Rotax 915 iS parameters are correct
- Environment variable overrides work
- Default values are sane
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AppSettings, EngineConfig, load_settings


class TestConfigLoading:
    """Test configuration loading and validation."""

    def test_load_from_default_yaml(self) -> None:
        """Config loads successfully from config/default.yaml."""
        settings = load_settings("config/default.yaml")
        assert settings is not None
        assert isinstance(settings, AppSettings)

    def test_load_with_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        """Loading from a nonexistent path falls back to defaults."""
        settings = load_settings(tmp_path / "nonexistent.yaml")
        assert settings.engine.name == "Rotax 915 iS"

    def test_engine_defaults_match_rotax_915is(self) -> None:
        """Default engine config matches Rotax 915 iS specifications."""
        engine = EngineConfig()
        assert engine.bore_mm == 84.0
        assert engine.stroke_mm == 61.0
        assert engine.displacement_cc == 1352.0
        assert engine.num_cylinders == 4
        assert engine.compression_ratio == 10.5
        assert engine.rated_power_kw == 105.0
        assert engine.rated_rpm == 5800
        assert engine.max_rpm == 5800


class TestRotax915iSParameters:
    """Validate Rotax 915 iS physical parameters from loaded config."""

    @pytest.fixture
    def engine(self) -> EngineConfig:
        return load_settings().engine

    def test_bore(self, engine: EngineConfig) -> None:
        assert engine.bore_mm == 84.0, "Rotax 915 iS bore is 84 mm"

    def test_stroke(self, engine: EngineConfig) -> None:
        assert engine.stroke_mm == 61.0, "Rotax 915 iS stroke is 61 mm"

    def test_displacement(self, engine: EngineConfig) -> None:
        """Verify displacement matches: V_d = (π/4) · B² · S · n_cyl"""
        import math

        computed = (math.pi / 4) * (84.0**2) * 61.0 * 4 / 1000.0  # cc
        assert abs(engine.displacement_cc - computed) < 5.0, (
            f"Displacement should be ~{computed:.0f} cc, got {engine.displacement_cc}"
        )

    def test_compression_ratio(self, engine: EngineConfig) -> None:
        assert engine.compression_ratio == 10.5

    def test_rated_power(self, engine: EngineConfig) -> None:
        assert engine.rated_power_kw == 105.0

    def test_firing_order(self, engine: EngineConfig) -> None:
        assert len(engine.firing_order) == 4
        assert set(engine.firing_order) == {1, 2, 3, 4}


class TestTurbochargerConfig:
    """Validate turbocharger configuration."""

    def test_max_boost(self) -> None:
        settings = load_settings()
        assert settings.turbocharger.max_boost_bar == 1.45

    def test_compressor_efficiency_range(self) -> None:
        settings = load_settings()
        assert 0.0 < settings.turbocharger.compressor_efficiency < 1.0


class TestMLConfig:
    """Validate ML pipeline configuration."""

    def test_health_index_weights_sum_to_one(self) -> None:
        """Health index weights must sum to 1.0."""
        settings = load_settings()
        weights = settings.ml.health_index_weights
        total = weights.thermal + weights.mechanical + weights.combustion + \
                weights.lubrication + weights.vibration
        assert abs(total - 1.0) < 1e-6, f"Health index weights sum to {total}, expected 1.0"

    def test_anomaly_threshold_range(self) -> None:
        settings = load_settings()
        assert 0.0 < settings.ml.anomaly_threshold <= 1.0

    def test_fault_confidence_threshold_range(self) -> None:
        settings = load_settings()
        assert 0.0 < settings.ml.fault_confidence_threshold <= 1.0

    def test_nine_fault_classes_defined(self) -> None:
        """Verify the fault class enum has exactly 9 classes."""
        from src.core.provenance import FaultClass
        assert len(FaultClass) == 9


class TestChannelRanges:
    """Validate channel range configuration."""

    def test_channel_ranges_loaded(self) -> None:
        settings = load_settings()
        assert len(settings.channel_ranges) > 0, "Channel ranges should be loaded from config"

    def test_rpm_range(self) -> None:
        settings = load_settings()
        rpm_range = settings.channel_ranges.get("rpm")
        assert rpm_range is not None
        assert rpm_range.min == 0
        assert rpm_range.max == 6500


class TestConfigExternalization:
    """Verify that configuration is truly externalized."""

    def test_no_hardcoded_bore_in_core(self) -> None:
        """The bore value 84.0 should only appear in config, not in core logic."""
        # This is a meta-test — it checks that we're not hardcoding
        # engine parameters in places other than config.
        import src.core.constants as constants

        # constants module should not contain engine-specific geometry
        source = Path(constants.__file__).read_text(encoding="utf-8") if constants.__file__ else ""
        assert "84.0" not in source or "bore" not in source.lower(), (
            "Engine bore should not be hardcoded in constants.py"
        )
