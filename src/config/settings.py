"""
Configuration Management for E-Powertrain Benchmarking System

Centralized settings for:
- API keys 
- File paths
- Quality thresholds
- Model configurations
- Retry logic
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, List, Dict, Any
from pathlib import Path

class Settings(BaseSettings):
    """
    Main configuration class using Pydantic Settings.
    Automatically loads from .env file.
    
    Example .env:
    OPENAI_API_KEY=sk-...
    PERPLEXITY_API_KEY=pplx-...
    """
    
    # API KEYS
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key (required)"
    )
    
    perplexity_api_key: Optional[str] = Field(
        None,
        description="Perplexity API key (optional)"
    )
    
    # FILE PATHS
    output_directory: Path = Field(
        default=Path("outputs"),
        description="Where to save outputs"
    )

    ppt_template_path: Path = Field(
        default=Path("templates/IAA_Template.pptx"),
        description="PowerPoint template path"
    )
    
    # MODEL CONFIGURATIONS
    scraping_model: str = Field(
        default="sonar-pro",
        description="Perplexity model for web scraping"
    )

    quality_validator_model: str = Field(
        default="gpt-4o",
        description="OpenAI model for quality validation"
    )

    presentation_model: str = Field(
        default="gpt-5-mini",
        description="OpenAI model for presentation generation"
    )
    
    # QUALITY THRESHOLDS
    minimum_quality_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum quality score (0-1) to proceed"
    )
    
    required_fields: list[str] = Field(
        default=[
            "vehicle_name",
            "battery_capacity_kwh",
            "motor_power_kw",
            "range_km"
        ],
        description="Required fields for each vehicle"
    )
    
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum retry attempts"
    )
    
    # SCRAPING PARAMETERS
    max_tokens_per_scrape: int = Field(
        default=8000,
        description="Max tokens for scraping"
    )

    scraping_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for scraping (lower = more deterministic)"
    )

    # INTELLIGENT NAVIGATION SETTINGS
    enable_intelligent_navigation: bool = Field(
        default=True,
        description="Use LLM-guided navigation to find spec pages (vs direct URL scraping)"
    )

    max_pages_per_oem: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum pages to crawl per OEM website"
    )

    llm_extraction_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model for extraction (via LiteLLM format)"
    )
    
    # COST TRACKING
    track_costs: bool = Field(
        default=True,
        description="Track API costs"
    )

    cost_per_1k_input_tokens: Dict[str, float] = Field(
        default={
            "sonar-pro": 0.003,       # Perplexity Sonar Pro
            "gpt-4o": 0.0025,         # OpenAI GPT-4o
            "gpt-5": 0.00125,         # OpenAI GPT-5
            "gpt-5-mini": 0.00025,    # OpenAI GPT-5 Mini
            "gpt-5-nano": 0.00005,    # OpenAI GPT-5 Nano
        },
        description="Cost per 1K input tokens (USD)"
    )

    cost_per_1k_output_tokens: Dict[str, float] = Field(
        default={
            "sonar-pro": 0.015,       # Perplexity Sonar Pro
            "gpt-4o": 0.01,           # OpenAI GPT-4o
            "gpt-5": 0.01,            # OpenAI GPT-5
            "gpt-5-mini": 0.002,      # OpenAI GPT-5 Mini
            "gpt-5-nano": 0.0004,     # OpenAI GPT-5 Nano
        },
        description="Cost per 1K output tokens (USD)"
    )

    perplexity_request_fee: float = Field(
        default=0.005,
        description="Perplexity API request fee ($5 per 1000 requests)"
    )
    
    # PYDANTIC CONFIG
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # HELPER METHODS
    def get_model_cost(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        include_request_fee: bool = False
    ) -> float:
        """
        Calculate cost for model and token count.

        Args:
            model_name: Model name (e.g., "sonar-pro", "gpt-5-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            include_request_fee: Add Perplexity request fee if applicable

        Returns:
            Total cost in USD
        """
        input_cost_per_1k = self.cost_per_1k_input_tokens.get(model_name, 0.0)
        output_cost_per_1k = self.cost_per_1k_output_tokens.get(model_name, 0.0)

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost

        # Add Perplexity request fee if applicable
        if include_request_fee and model_name.startswith("sonar"):
            total_cost += self.perplexity_request_fee

        return total_cost
    
    def should_retry(self, quality_score: float, attempt_number: int) -> bool:
        """Check if should retry based on quality and attempts"""
        return (
            quality_score < self.minimum_quality_score 
            and attempt_number < self.max_retry_attempts
        )


# Global instance
settings = Settings()


if __name__ == "__main__":
    """Test configuration loading"""
    print("=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    
    print("\nAPI KEYS:")
    print(f"  OpenAI: {'Loaded' if settings.openai_api_key else 'Missing'}")
    print(f"  Perplexity: {'Loaded' if settings.perplexity_api_key else 'Missing'}")
    
    print("\nMODELS:")
    print(f"  Scraping: {settings.scraping_model}")
    print(f"  Quality: {settings.quality_validator_model}")
    print(f"  Presentation: {settings.presentation_model}")
    
    print("\nQUALITY:")
    print(f"  Minimum Score: {settings.minimum_quality_score}")
    print(f"  Max Retries: {settings.max_retry_attempts}")
    
    print("\nCOST TRACKING:")
    print(f"  Enabled: {settings.track_costs}")

    # Example cost calculations
    sonar_cost = settings.get_model_cost("sonar-pro", input_tokens=5000, output_tokens=2000, include_request_fee=True)
    gpt5_cost = settings.get_model_cost("gpt-5-mini", input_tokens=3000, output_tokens=1000)

    print(f"\n  Example Costs:")
    print(f"    Sonar-Pro (5K in, 2K out + request fee): ${sonar_cost:.4f}")
    print(f"    GPT-5-Mini (3K in, 1K out): ${gpt5_cost:.4f}")

    print("\n" + "=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)

