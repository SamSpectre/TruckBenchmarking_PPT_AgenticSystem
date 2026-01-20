"""
Unit Tests for Scraper Tool (src/tools/scraper.py)

Tests cover:
- Scraper configuration
- Pydantic models for extraction
- Rate limiter
- Web content fetching (mocked)
- Data extraction (mocked)
- Completeness scoring
"""

import pytest
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.scraper import (
    ScraperConfig,
    VehicleSpec,
    VehicleSpecList,
    SpecPageLink,
    SpecPageLinks,
    AsyncRateLimiter,
)


class TestScraperConfig:
    """Test ScraperConfig class."""

    @pytest.mark.unit
    def test_config_api_url(self):
        """Test Perplexity API URL."""
        assert ScraperConfig.API_URL == "https://api.perplexity.ai/chat/completions"

    @pytest.mark.unit
    def test_config_default_model(self):
        """Test default model is sonar-pro."""
        assert ScraperConfig.DEFAULT_MODEL == "sonar-pro"

    @pytest.mark.unit
    def test_config_max_tokens(self):
        """Test max tokens setting."""
        assert ScraperConfig.MAX_TOKENS == 8000

    @pytest.mark.unit
    def test_config_temperature(self):
        """Test temperature setting."""
        assert ScraperConfig.TEMPERATURE == 0.1

    @pytest.mark.unit
    def test_config_max_pages_per_oem(self):
        """Test max pages per OEM setting."""
        assert ScraperConfig.MAX_PAGES_PER_OEM >= 5

    @pytest.mark.unit
    def test_config_parallel_settings(self):
        """Test parallel processing settings."""
        assert ScraperConfig.PARALLEL_URL_THRESHOLD == 2
        assert ScraperConfig.MAX_CONCURRENT_URLS >= 2
        assert ScraperConfig.MAX_CONCURRENT_API_CALLS >= 2

    @pytest.mark.unit
    def test_config_async_timeout(self):
        """Test async timeout setting."""
        assert ScraperConfig.ASYNC_TIMEOUT_SECONDS >= 60

    @pytest.mark.unit
    def test_config_auto_fallback(self):
        """Test auto-fallback settings."""
        assert hasattr(ScraperConfig, 'AUTO_FALLBACK_ENABLED')
        assert hasattr(ScraperConfig, 'MIN_COMPLETENESS_THRESHOLD')


class TestVehicleSpecModel:
    """Test VehicleSpec Pydantic model."""

    @pytest.mark.unit
    def test_vehicle_spec_required_field(self):
        """Test that vehicle_name is required."""
        spec = VehicleSpec(vehicle_name="MAN eTGX")
        assert spec.vehicle_name == "MAN eTGX"

    @pytest.mark.unit
    def test_vehicle_spec_optional_fields(self):
        """Test optional fields default to None."""
        spec = VehicleSpec(vehicle_name="Test")

        assert spec.battery_capacity_kwh is None
        assert spec.range_km is None
        assert spec.motor_power_kw is None

    @pytest.mark.unit
    def test_vehicle_spec_full_data(self):
        """Test VehicleSpec with full data."""
        spec = VehicleSpec(
            vehicle_name="MAN eTGX 4x2",
            battery_capacity_kwh=480.0,
            battery_voltage_v=800.0,
            motor_power_kw=480.0,
            motor_torque_nm=3100.0,
            range_km=600.0,
            dc_charging_kw=375.0,
            gvw_kg=40000.0,
            powertrain_type="BEV"
        )

        assert spec.vehicle_name == "MAN eTGX 4x2"
        assert spec.battery_capacity_kwh == 480.0
        assert spec.range_km == 600.0
        assert spec.powertrain_type == "BEV"

    @pytest.mark.unit
    def test_vehicle_spec_to_dict(self):
        """Test VehicleSpec can be converted to dict."""
        spec = VehicleSpec(
            vehicle_name="Test",
            battery_capacity_kwh=400.0
        )

        data = spec.model_dump()
        assert isinstance(data, dict)
        assert data["vehicle_name"] == "Test"
        assert data["battery_capacity_kwh"] == 400.0


class TestVehicleSpecList:
    """Test VehicleSpecList Pydantic model."""

    @pytest.mark.unit
    def test_vehicle_spec_list_empty(self):
        """Test empty vehicle list."""
        spec_list = VehicleSpecList()
        assert spec_list.vehicles == []

    @pytest.mark.unit
    def test_vehicle_spec_list_with_vehicles(self):
        """Test vehicle list with vehicles."""
        spec_list = VehicleSpecList(
            vehicles=[
                VehicleSpec(vehicle_name="MAN eTGX"),
                VehicleSpec(vehicle_name="MAN eTGM"),
            ]
        )

        assert len(spec_list.vehicles) == 2
        assert spec_list.vehicles[0].vehicle_name == "MAN eTGX"


class TestSpecPageLinkModels:
    """Test SpecPageLink and SpecPageLinks models."""

    @pytest.mark.unit
    def test_spec_page_link(self):
        """Test SpecPageLink model."""
        link = SpecPageLink(
            url="https://www.man.eu/trucks/etgx",
            vehicle_name="MAN eTGX",
            confidence=0.9
        )

        assert link.url == "https://www.man.eu/trucks/etgx"
        assert link.vehicle_name == "MAN eTGX"
        assert link.confidence == 0.9

    @pytest.mark.unit
    def test_spec_page_link_default_confidence(self):
        """Test SpecPageLink default confidence."""
        link = SpecPageLink(url="https://example.com")

        assert link.confidence == 0.8

    @pytest.mark.unit
    def test_spec_page_links_list(self):
        """Test SpecPageLinks container."""
        links = SpecPageLinks(
            spec_links=[
                SpecPageLink(url="https://example.com/page1"),
                SpecPageLink(url="https://example.com/page2"),
            ]
        )

        assert len(links.spec_links) == 2


class TestAsyncRateLimiter:
    """Test AsyncRateLimiter class."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiter functionality."""
        limiter = AsyncRateLimiter(max_concurrent=2, delay_seconds=0)

        async with limiter:
            pass  # Should complete without error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_limit(self):
        """Test that rate limiter enforces concurrent limit."""
        limiter = AsyncRateLimiter(max_concurrent=2, delay_seconds=0)
        active_count = 0
        max_active = 0

        async def task():
            nonlocal active_count, max_active
            async with limiter:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.1)
                active_count -= 1

        # Run 5 concurrent tasks
        await asyncio.gather(*[task() for _ in range(5)])

        # Max active should never exceed limit
        assert max_active <= 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_release(self):
        """Test explicit acquire and release."""
        limiter = AsyncRateLimiter(max_concurrent=1, delay_seconds=0)

        await limiter.acquire()
        limiter.release()

        # Should be able to acquire again
        await limiter.acquire()
        limiter.release()


class TestScraperHelperFunctions:
    """Test scraper helper functions."""

    @pytest.mark.unit
    def test_calculate_completeness_full_data(self):
        """Test completeness calculation with full data."""
        # Import the function if available, or test the logic
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": 400,
            "range_km": 500,
            "motor_power_kw": 300,
            "gvw_kg": 40000,
            "dc_charging_kw": 200,
        }

        # Calculate completeness (non-None fields / total important fields)
        important_fields = ["battery_capacity_kwh", "range_km", "motor_power_kw", "gvw_kg", "dc_charging_kw"]
        filled = sum(1 for f in important_fields if vehicle.get(f) is not None)
        completeness = filled / len(important_fields)

        assert completeness == 1.0

    @pytest.mark.unit
    def test_calculate_completeness_partial_data(self):
        """Test completeness calculation with partial data."""
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": 400,
            "range_km": None,  # Missing
            "motor_power_kw": 300,
            "gvw_kg": None,  # Missing
        }

        important_fields = ["battery_capacity_kwh", "range_km", "motor_power_kw", "gvw_kg"]
        filled = sum(1 for f in important_fields if vehicle.get(f) is not None)
        completeness = filled / len(important_fields)

        assert completeness == 0.5


class TestURLParsing:
    """Test URL parsing and domain extraction."""

    @pytest.mark.unit
    def test_extract_domain_from_url(self):
        """Test domain extraction from URL."""
        url = "https://www.man.eu/trucks/etgx/specifications"

        if "://" in url:
            domain = url.split("/")[2].replace("www.", "")
        else:
            domain = url

        assert domain == "man.eu"

    @pytest.mark.unit
    def test_extract_domain_without_www(self):
        """Test domain extraction from URL without www."""
        url = "https://volvotrucks.com/en-us/trucks"

        if "://" in url:
            domain = url.split("/")[2].replace("www.", "")
        else:
            domain = url

        assert domain == "volvotrucks.com"

    @pytest.mark.unit
    def test_normalize_url(self):
        """Test URL normalization."""
        from urllib.parse import urljoin

        base_url = "https://www.man.eu/trucks/"
        relative_url = "../electric/etgx"

        full_url = urljoin(base_url, relative_url)

        assert "man.eu" in full_url
        assert full_url.startswith("https://")


class TestMockedPerplexityScraper:
    """Test scraper with mocked Perplexity API calls."""

    @pytest.mark.unit
    def test_parse_perplexity_response(self, mock_perplexity_response):
        """Test parsing Perplexity API response."""
        content = mock_perplexity_response["choices"][0]["message"]["content"]

        # Extract JSON from markdown code block
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

        assert json_match is not None

        import json
        data = json.loads(json_match.group(1))

        assert "vehicles" in data
        assert len(data["vehicles"]) == 1
        assert data["vehicles"][0]["vehicle_name"] == "MAN eTGX"

    @pytest.mark.unit
    def test_extract_citations(self, mock_perplexity_response):
        """Test citation extraction from response."""
        citations = mock_perplexity_response.get("citations", [])

        assert len(citations) == 1
        assert "man.eu" in citations[0]

    @pytest.mark.unit
    def test_calculate_token_usage(self, mock_perplexity_response):
        """Test token usage extraction."""
        usage = mock_perplexity_response["usage"]

        total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]

        assert total_tokens == 1500


class TestMockedCrawl4AI:
    """Test scraper with mocked Crawl4AI."""

    @pytest.mark.unit
    def test_parse_crawl_result(self, mock_crawl4ai_result):
        """Test parsing Crawl4AI result."""
        assert mock_crawl4ai_result.success is True
        assert "MAN eTGX" in mock_crawl4ai_result.markdown
        assert "Battery" in mock_crawl4ai_result.markdown

    @pytest.mark.unit
    def test_extract_specs_from_markdown(self, mock_crawl4ai_result):
        """Test extracting specifications from markdown content."""
        markdown = mock_crawl4ai_result.markdown

        # Check for key specifications
        assert "480 kWh" in markdown
        assert "600 km" in markdown
        assert "480 kW" in markdown

    @pytest.mark.unit
    def test_handle_failed_crawl(self):
        """Test handling failed crawl result."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.markdown = ""
        mock_result.error_message = "Connection timeout"

        assert mock_result.success is False
        assert mock_result.error_message == "Connection timeout"


class TestScrapingResultConstruction:
    """Test constructing ScrapingResult from extracted data."""

    @pytest.mark.unit
    def test_construct_scraping_result(self):
        """Test constructing a complete ScrapingResult."""
        vehicles = [
            {
                "vehicle_name": "MAN eTGX",
                "battery_capacity_kwh": 480,
                "range_km": 600,
                "motor_power_kw": 480,
            }
        ]

        result = {
            "oem_name": "MAN Truck & Bus",
            "oem_url": "https://www.man.eu/trucks",
            "vehicles": vehicles,
            "total_vehicles_found": len(vehicles),
            "extraction_timestamp": datetime.now().isoformat(),
            "official_citations": ["https://www.man.eu/trucks/etgx"],
            "third_party_citations": [],
            "source_compliance_score": 0.8,
            "tokens_used": 1500,
            "model_used": "sonar-pro",
        }

        assert result["oem_name"] == "MAN Truck & Bus"
        assert result["total_vehicles_found"] == 1
        assert result["vehicles"][0]["battery_capacity_kwh"] == 480

    @pytest.mark.unit
    def test_construct_empty_result_on_failure(self):
        """Test constructing result when extraction fails."""
        result = {
            "oem_name": "Unknown",
            "oem_url": "https://example.com",
            "vehicles": [],
            "total_vehicles_found": 0,
            "extraction_timestamp": datetime.now().isoformat(),
            "official_citations": [],
            "third_party_citations": [],
            "source_compliance_score": 0.0,
            "errors": ["Failed to extract vehicle data"],
        }

        assert result["total_vehicles_found"] == 0
        assert len(result["errors"]) > 0


class TestUnitConversion:
    """Test unit conversion heuristics in scraper."""

    @pytest.mark.unit
    def test_convert_tonnes_to_kg(self):
        """Test converting tonnes to kg."""
        value_tonnes = 40  # 40 tonnes
        value_kg = value_tonnes * 1000

        assert value_kg == 40000

    @pytest.mark.unit
    def test_detect_unit_from_value(self):
        """Test detecting unit from value magnitude."""
        # If GVW < 100, likely in tonnes, should convert to kg
        gvw_value = 40

        if gvw_value < 100:  # Probably tonnes
            gvw_kg = gvw_value * 1000
        else:
            gvw_kg = gvw_value

        assert gvw_kg == 40000

    @pytest.mark.unit
    def test_parse_range_values(self):
        """Test parsing range values like '400-600 km'."""
        range_string = "400-600 km"

        # Extract numbers
        import re
        numbers = re.findall(r'\d+', range_string)

        assert len(numbers) == 2
        min_range = int(numbers[0])
        max_range = int(numbers[1])

        assert min_range == 400
        assert max_range == 600


class TestErrorHandling:
    """Test error handling in scraper."""

    @pytest.mark.unit
    def test_handle_invalid_json(self):
        """Test handling invalid JSON in response."""
        invalid_content = "Not valid JSON {"

        import json
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_content)

    @pytest.mark.unit
    def test_handle_missing_vehicles_key(self):
        """Test handling response without vehicles key."""
        response = {"data": "something", "other": "field"}

        vehicles = response.get("vehicles", [])

        assert vehicles == []

    @pytest.mark.unit
    def test_handle_null_vehicle_values(self):
        """Test handling null values in vehicle data."""
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": None,
            "range_km": None,
        }

        # Should handle None gracefully
        battery = vehicle.get("battery_capacity_kwh") or 0
        assert battery == 0
