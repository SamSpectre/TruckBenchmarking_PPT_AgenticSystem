"""
Unit Tests for Configuration Settings (src/config/settings.py)

Tests cover:
- Settings loading from environment
- Default values
- Cost calculation methods
- Retry logic
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSettingsLoading:
    """Test settings loading and environment variables."""

    @pytest.mark.unit
    def test_settings_loads_from_env(self, mock_env_with_api_keys):
        """Test that settings loads API keys from environment."""
        # Create new Settings instance with mocked env vars
        from src.config.settings import Settings
        settings = Settings()
        # Either it loads our test key or the real key from .env
        assert settings.openai_api_key is not None
        assert len(settings.openai_api_key) > 5

    @pytest.mark.unit
    def test_settings_has_default_output_directory(self, mock_env_with_api_keys):
        """Test default output directory."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.output_directory == Path("outputs")

    @pytest.mark.unit
    def test_settings_has_default_template_path(self, mock_env_with_api_keys):
        """Test default template path."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.ppt_template_path == Path("templates/IAA_Template.pptx")


class TestSettingsDefaults:
    """Test default configuration values."""

    @pytest.mark.unit
    def test_default_scraping_model(self, mock_env_with_api_keys):
        """Test default scraping model."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.scraping_model == "sonar-pro"

    @pytest.mark.unit
    def test_default_quality_validator_model(self, mock_env_with_api_keys):
        """Test default quality validator model."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.quality_validator_model == "gpt-4o"

    @pytest.mark.unit
    def test_default_minimum_quality_score(self, mock_env_with_api_keys):
        """Test default minimum quality score."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.minimum_quality_score == 0.75

    @pytest.mark.unit
    def test_default_max_retry_attempts(self, mock_env_with_api_keys):
        """Test default max retry attempts."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.max_retry_attempts == 3

    @pytest.mark.unit
    def test_default_max_tokens_per_scrape(self, mock_env_with_api_keys):
        """Test default max tokens per scrape."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.max_tokens_per_scrape == 8000

    @pytest.mark.unit
    def test_default_scraping_temperature(self, mock_env_with_api_keys):
        """Test default scraping temperature."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.scraping_temperature == 0.1

    @pytest.mark.unit
    def test_default_track_costs_enabled(self, mock_env_with_api_keys):
        """Test that cost tracking is enabled by default."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.track_costs is True

    @pytest.mark.unit
    def test_default_required_fields(self, mock_env_with_api_keys):
        """Test default required fields list."""
        from src.config.settings import Settings
        settings = Settings()

        expected = ["vehicle_name", "battery_capacity_kwh", "motor_power_kw", "range_km"]
        assert settings.required_fields == expected


class TestCostCalculation:
    """Test cost calculation methods."""

    @pytest.mark.unit
    def test_get_model_cost_sonar_pro(self, mock_env_with_api_keys):
        """Test cost calculation for sonar-pro model."""
        from src.config.settings import Settings
        settings = Settings()

        # 1000 input tokens, 500 output tokens
        cost = settings.get_model_cost(
            "sonar-pro",
            input_tokens=1000,
            output_tokens=500,
            include_request_fee=False
        )

        # Input: 1000/1000 * 0.003 = 0.003
        # Output: 500/1000 * 0.015 = 0.0075
        expected = 0.003 + 0.0075
        assert abs(cost - expected) < 0.0001

    @pytest.mark.unit
    def test_get_model_cost_with_request_fee(self, mock_env_with_api_keys):
        """Test cost calculation with Perplexity request fee."""
        from src.config.settings import Settings
        settings = Settings()

        cost = settings.get_model_cost(
            "sonar-pro",
            input_tokens=1000,
            output_tokens=500,
            include_request_fee=True
        )

        # Input: 1000/1000 * 0.003 = 0.003
        # Output: 500/1000 * 0.015 = 0.0075
        # Fee: 0.005
        expected = 0.003 + 0.0075 + 0.005
        assert abs(cost - expected) < 0.0001

    @pytest.mark.unit
    def test_get_model_cost_gpt4o(self, mock_env_with_api_keys):
        """Test cost calculation for gpt-4o model."""
        from src.config.settings import Settings
        settings = Settings()

        cost = settings.get_model_cost(
            "gpt-4o",
            input_tokens=2000,
            output_tokens=1000
        )

        # Input: 2000/1000 * 0.0025 = 0.005
        # Output: 1000/1000 * 0.01 = 0.01
        expected = 0.005 + 0.01
        assert abs(cost - expected) < 0.0001

    @pytest.mark.unit
    def test_get_model_cost_unknown_model(self, mock_env_with_api_keys):
        """Test cost calculation for unknown model returns 0."""
        from src.config.settings import Settings
        settings = Settings()

        cost = settings.get_model_cost(
            "unknown-model",
            input_tokens=1000,
            output_tokens=500
        )

        assert cost == 0.0

    @pytest.mark.unit
    def test_get_model_cost_zero_tokens(self, mock_env_with_api_keys):
        """Test cost calculation with zero tokens."""
        from src.config.settings import Settings
        settings = Settings()

        cost = settings.get_model_cost(
            "sonar-pro",
            input_tokens=0,
            output_tokens=0
        )

        assert cost == 0.0

    @pytest.mark.unit
    def test_get_model_cost_gpt5_mini(self, mock_env_with_api_keys):
        """Test cost calculation for gpt-5-mini model."""
        from src.config.settings import Settings
        settings = Settings()

        cost = settings.get_model_cost(
            "gpt-5-mini",
            input_tokens=5000,
            output_tokens=2000
        )

        # Input: 5000/1000 * 0.00025 = 0.00125
        # Output: 2000/1000 * 0.002 = 0.004
        expected = 0.00125 + 0.004
        assert abs(cost - expected) < 0.0001


class TestRetryLogic:
    """Test retry decision logic."""

    @pytest.mark.unit
    def test_should_retry_low_quality_first_attempt(self, mock_env_with_api_keys):
        """Test retry with low quality on first attempt."""
        from src.config.settings import Settings
        settings = Settings()

        should_retry = settings.should_retry(quality_score=0.5, attempt_number=1)

        assert should_retry is True

    @pytest.mark.unit
    def test_should_retry_high_quality(self, mock_env_with_api_keys):
        """Test no retry needed with high quality."""
        from src.config.settings import Settings
        settings = Settings()

        should_retry = settings.should_retry(quality_score=0.85, attempt_number=1)

        assert should_retry is False

    @pytest.mark.unit
    def test_should_retry_max_attempts_reached(self, mock_env_with_api_keys):
        """Test no retry when max attempts reached."""
        from src.config.settings import Settings
        settings = Settings()

        # On 3rd attempt (default max is 3)
        should_retry = settings.should_retry(quality_score=0.5, attempt_number=3)

        assert should_retry is False

    @pytest.mark.unit
    def test_should_retry_edge_case_threshold(self, mock_env_with_api_keys):
        """Test retry at exact threshold boundary."""
        from src.config.settings import Settings
        settings = Settings()

        # At threshold (0.75), should not retry
        should_retry = settings.should_retry(quality_score=0.75, attempt_number=1)
        assert should_retry is False

        # Just below threshold, should retry
        should_retry = settings.should_retry(quality_score=0.74, attempt_number=1)
        assert should_retry is True


class TestCostPerTokenMappings:
    """Test cost per token mappings."""

    @pytest.mark.unit
    def test_input_token_costs_exist(self, mock_env_with_api_keys):
        """Test that input token costs are defined."""
        from src.config.settings import Settings
        settings = Settings()

        assert "sonar-pro" in settings.cost_per_1k_input_tokens
        assert "gpt-4o" in settings.cost_per_1k_input_tokens
        assert "gpt-5-mini" in settings.cost_per_1k_input_tokens

    @pytest.mark.unit
    def test_output_token_costs_exist(self, mock_env_with_api_keys):
        """Test that output token costs are defined."""
        from src.config.settings import Settings
        settings = Settings()

        assert "sonar-pro" in settings.cost_per_1k_output_tokens
        assert "gpt-4o" in settings.cost_per_1k_output_tokens
        assert "gpt-5-mini" in settings.cost_per_1k_output_tokens

    @pytest.mark.unit
    def test_output_tokens_more_expensive(self, mock_env_with_api_keys):
        """Test that output tokens cost more than input tokens."""
        from src.config.settings import Settings
        settings = Settings()

        for model in settings.cost_per_1k_input_tokens:
            input_cost = settings.cost_per_1k_input_tokens[model]
            output_cost = settings.cost_per_1k_output_tokens.get(model, 0)
            # Output should be >= input for most models
            assert output_cost >= input_cost, f"{model} output cost should be >= input cost"


class TestIntelligentNavigationSettings:
    """Test intelligent navigation settings."""

    @pytest.mark.unit
    def test_intelligent_navigation_enabled_by_default(self, mock_env_with_api_keys):
        """Test intelligent navigation is enabled by default."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.enable_intelligent_navigation is True

    @pytest.mark.unit
    def test_max_pages_per_oem_default(self, mock_env_with_api_keys):
        """Test default max pages per OEM."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.max_pages_per_oem == 5
        assert settings.max_pages_per_oem >= 1
        assert settings.max_pages_per_oem <= 10

    @pytest.mark.unit
    def test_llm_extraction_model_default(self, mock_env_with_api_keys):
        """Test default LLM extraction model."""
        from src.config.settings import Settings
        settings = Settings()

        assert "gpt-4o" in settings.llm_extraction_model


class TestSettingsValidation:
    """Test settings validation constraints."""

    @pytest.mark.unit
    def test_minimum_quality_score_bounds(self, mock_env_with_api_keys):
        """Test minimum quality score is between 0 and 1."""
        from src.config.settings import Settings
        settings = Settings()

        assert 0.0 <= settings.minimum_quality_score <= 1.0

    @pytest.mark.unit
    def test_scraping_temperature_bounds(self, mock_env_with_api_keys):
        """Test scraping temperature is between 0 and 2."""
        from src.config.settings import Settings
        settings = Settings()

        assert 0.0 <= settings.scraping_temperature <= 2.0

    @pytest.mark.unit
    def test_max_retry_attempts_bounds(self, mock_env_with_api_keys):
        """Test max retry attempts is between 1 and 5."""
        from src.config.settings import Settings
        settings = Settings()

        assert 1 <= settings.max_retry_attempts <= 5

    @pytest.mark.unit
    def test_perplexity_request_fee_positive(self, mock_env_with_api_keys):
        """Test Perplexity request fee is positive."""
        from src.config.settings import Settings
        settings = Settings()

        assert settings.perplexity_request_fee > 0


class TestSettingsGlobalInstance:
    """Test global settings instance."""

    @pytest.mark.unit
    def test_global_settings_instance_exists(self, mock_env_with_api_keys):
        """Test that global settings instance is accessible."""
        from src.config.settings import settings

        assert settings is not None
        assert hasattr(settings, "openai_api_key")

    @pytest.mark.unit
    def test_global_settings_is_singleton(self, mock_env_with_api_keys):
        """Test that settings module provides consistent instance."""
        from src.config.settings import settings, Settings

        # Settings instance should have consistent values
        assert settings.minimum_quality_score == 0.75
        # Creating a new instance should have same defaults
        new_settings = Settings()
        assert new_settings.minimum_quality_score == settings.minimum_quality_score
