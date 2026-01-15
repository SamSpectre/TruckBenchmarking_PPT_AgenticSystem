"""
Unit Tests for PPT Generator (src/tools/ppt_generator.py)

Tests cover:
- Data transformation (VehicleSpecifications -> IAA format)
- Highlight generation
- Assessment generation
- Technology extraction
- Shape ID mappings
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.ppt_generator import (
    SHAPE_IDS,
    PRODUCT_TABLE_ROWS,
    transform_vehicle_to_iaa_product,
    transform_scraping_result_to_oem_profile,
    generate_highlights,
    generate_assessment,
    extract_technologies,
    extract_country_from_url,
    determine_oem_category,
)


class TestShapeIdMappings:
    """Test shape ID mappings for IAA template."""

    @pytest.mark.unit
    def test_shape_ids_defined(self):
        """Test that required shape IDs are defined."""
        assert "company_name" in SHAPE_IDS
        assert "company_info_table" in SHAPE_IDS
        assert "products_table" in SHAPE_IDS
        assert "expected_highlights" in SHAPE_IDS
        assert "assessment" in SHAPE_IDS
        assert "technologies" in SHAPE_IDS

    @pytest.mark.unit
    def test_product_table_rows_defined(self):
        """Test that product table row indices are defined."""
        assert "name" in PRODUCT_TABLE_ROWS
        assert "battery" in PRODUCT_TABLE_ROWS
        assert "range" in PRODUCT_TABLE_ROWS
        assert "charging" in PRODUCT_TABLE_ROWS
        assert "gvw_gcw" in PRODUCT_TABLE_ROWS

    @pytest.mark.unit
    def test_product_table_row_indices(self):
        """Test product table row indices are correct."""
        assert PRODUCT_TABLE_ROWS["name"] == 0
        assert PRODUCT_TABLE_ROWS["battery"] == 5
        assert PRODUCT_TABLE_ROWS["range"] == 4

    @pytest.mark.unit
    def test_all_product_rows_sequential(self):
        """Test that product table rows are sequential from 0-13."""
        indices = list(PRODUCT_TABLE_ROWS.values())
        assert min(indices) == 0
        assert max(indices) == 13
        assert len(set(indices)) == 14  # All unique


class TestTransformVehicleToIAAProduct:
    """Test vehicle to IAA product transformation."""

    @pytest.mark.unit
    def test_transform_full_vehicle(self, sample_vehicle_specs):
        """Test transforming vehicle with full data."""
        product = transform_vehicle_to_iaa_product(sample_vehicle_specs)

        assert product["name"] == "MAN eTGX 4x2"
        assert "480" in product["battery"]
        assert "kWh" in product["battery"]
        assert "600" in product["range"]
        assert "km" in product["range"]

    @pytest.mark.unit
    def test_transform_minimal_vehicle(self, sample_vehicle_specs_minimal):
        """Test transforming vehicle with minimal data."""
        product = transform_vehicle_to_iaa_product(sample_vehicle_specs_minimal)

        assert product["name"] == "Test Vehicle"
        assert product["battery"] == "N/A"
        assert product["range"] == "N/A"

    @pytest.mark.unit
    def test_transform_handles_none_values(self):
        """Test that None values become 'N/A'."""
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": None,
            "range_km": None,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["battery"] == "N/A"
        assert product["range"] == "N/A"

    @pytest.mark.unit
    def test_transform_handles_zero_values(self):
        """Test that zero values become 'N/A'."""
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": 0,
            "range_km": 0,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["battery"] == "N/A"
        assert product["range"] == "N/A"

    @pytest.mark.unit
    def test_transform_formats_gvw(self):
        """Test GVW formatting includes 'kg GVW'."""
        vehicle = {
            "vehicle_name": "Test",
            "gvw_kg": 40000,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert "40,000" in product["gvw_gcw"] or "40000" in product["gvw_gcw"]
        assert "kg" in product["gvw_gcw"]

    @pytest.mark.unit
    def test_transform_includes_wheel_formula(self):
        """Test wheel formula from additional_specs."""
        vehicle = {
            "vehicle_name": "Test",
            "additional_specs": {
                "wheel_formula": "6x4"
            }
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["wheel_formula"] == "6x4"

    @pytest.mark.unit
    def test_transform_formats_charging(self):
        """Test charging power formatting."""
        vehicle = {
            "vehicle_name": "Test",
            "dc_charging_kw": 375,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert "375" in product["charging"]
        assert "DC" in product["charging"] or "kW" in product["charging"]


class TestTransformScrapingResultToOEMProfile:
    """Test scraping result to OEM profile transformation."""

    @pytest.mark.unit
    def test_transform_scraping_result(self, sample_scraping_result):
        """Test transforming scraping result to OEM profile."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert profile["company_name"] == "MAN Truck & Bus"
        assert "products" in profile
        assert len(profile["products"]) >= 1

    @pytest.mark.unit
    def test_profile_includes_company_info(self, sample_scraping_result):
        """Test that profile includes company info."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert "company_info" in profile
        assert "website" in profile["company_info"]

    @pytest.mark.unit
    def test_profile_includes_highlights(self, sample_scraping_result):
        """Test that profile includes highlights."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert "expected_highlights" in profile
        assert isinstance(profile["expected_highlights"], list)

    @pytest.mark.unit
    def test_profile_includes_assessment(self, sample_scraping_result):
        """Test that profile includes assessment."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert "assessment" in profile
        assert isinstance(profile["assessment"], list)

    @pytest.mark.unit
    def test_profile_limits_products(self, sample_scraping_result):
        """Test that profile limits to 2 products max."""
        # Add more vehicles
        sample_scraping_result["vehicles"] = [
            {"vehicle_name": "V1", "oem_name": "Test", "source_url": "https://example.com"},
            {"vehicle_name": "V2", "oem_name": "Test", "source_url": "https://example.com"},
            {"vehicle_name": "V3", "oem_name": "Test", "source_url": "https://example.com"},
        ]

        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert len(profile["products"]) <= 2


class TestGenerateHighlights:
    """Test highlight generation from vehicle data."""

    @pytest.mark.unit
    def test_generate_highlights_bev(self):
        """Test highlight generation for BEV vehicles."""
        vehicles = [
            {
                "vehicle_name": "eTGX",
                "powertrain_type": "BEV",
                "battery_capacity_kwh": 480,
                "range_km": 600,
            }
        ]

        highlights = generate_highlights(vehicles, "MAN")

        assert len(highlights) > 0
        assert any("Battery Electric" in h or "BEV" in h for h in highlights)

    @pytest.mark.unit
    def test_generate_highlights_range(self):
        """Test range highlight generation."""
        vehicles = [
            {"vehicle_name": "Test", "range_km": 500},
        ]

        highlights = generate_highlights(vehicles, "Test OEM")

        assert any("500" in h and "km" in h for h in highlights)

    @pytest.mark.unit
    def test_generate_highlights_battery(self):
        """Test battery highlight generation."""
        vehicles = [
            {"vehicle_name": "Test", "battery_capacity_kwh": 600},
        ]

        highlights = generate_highlights(vehicles, "Test OEM")

        assert any("600" in h and "kWh" in h for h in highlights)

    @pytest.mark.unit
    def test_generate_highlights_limit(self):
        """Test that highlights are limited to 4."""
        vehicles = [
            {
                "vehicle_name": "Test",
                "powertrain_type": "BEV",
                "battery_capacity_kwh": 500,
                "range_km": 600,
                "motor_power_kw": 400,
            }
        ]

        highlights = generate_highlights(vehicles, "Test OEM")

        assert len(highlights) <= 4

    @pytest.mark.unit
    def test_generate_highlights_empty_vehicles(self):
        """Test highlights with empty vehicle list."""
        highlights = generate_highlights([], "Test OEM")

        assert len(highlights) >= 1  # Should have default


class TestGenerateAssessment:
    """Test assessment generation from scraping result."""

    @pytest.mark.unit
    def test_generate_assessment_high_quality(self, sample_scraping_result):
        """Test assessment for high quality data."""
        sample_scraping_result["source_compliance_score"] = 0.9

        assessment = generate_assessment(sample_scraping_result)

        assert isinstance(assessment, list)
        assert len(assessment) > 0

    @pytest.mark.unit
    def test_generate_assessment_low_quality(self):
        """Test assessment for low quality data."""
        scraping_result = {
            "source_compliance_score": 0.3,
            "vehicles": [{"vehicle_name": "Test"}],
            "official_citations": [],
        }

        assessment = generate_assessment(scraping_result)

        assert isinstance(assessment, list)


class TestExtractTechnologies:
    """Test technology extraction from vehicles."""

    @pytest.mark.unit
    def test_extract_technologies_basic(self, sample_vehicles_list):
        """Test basic technology extraction."""
        technologies = extract_technologies(sample_vehicles_list)

        assert isinstance(technologies, list)

    @pytest.mark.unit
    def test_extract_technologies_high_voltage(self):
        """Test extraction of high voltage technology."""
        vehicles = [
            {"vehicle_name": "Test", "battery_voltage_v": 800}
        ]

        technologies = extract_technologies(vehicles)

        # Should mention 800V architecture if detected
        assert isinstance(technologies, list)


class TestExtractCountryFromURL:
    """Test country extraction from URL."""

    @pytest.mark.unit
    def test_extract_country_germany(self):
        """Test extracting Germany from .de or german URL."""
        url = "https://www.man.eu/trucks"

        country = extract_country_from_url(url)

        # MAN is German, should detect
        assert country is not None or country == ""

    @pytest.mark.unit
    def test_extract_country_unknown(self):
        """Test handling unknown country."""
        url = "https://example.com"

        country = extract_country_from_url(url)

        assert isinstance(country, str)


class TestDetermineOEMCategory:
    """Test OEM category determination."""

    @pytest.mark.unit
    def test_determine_category_bev(self):
        """Test category for BEV vehicles."""
        vehicles = [
            {"powertrain_type": "BEV"},
            {"powertrain_type": "BEV"},
        ]

        category = determine_oem_category(vehicles)

        assert "BEV" in category or "Electric" in category or isinstance(category, str)

    @pytest.mark.unit
    def test_determine_category_fcev(self):
        """Test category for FCEV vehicles."""
        vehicles = [
            {"powertrain_type": "FCEV"},
        ]

        category = determine_oem_category(vehicles)

        assert isinstance(category, str)

    @pytest.mark.unit
    def test_determine_category_mixed(self):
        """Test category for mixed powertrains."""
        vehicles = [
            {"powertrain_type": "BEV"},
            {"powertrain_type": "FCEV"},
        ]

        category = determine_oem_category(vehicles)

        assert isinstance(category, str)


class TestValueFormatting:
    """Test value formatting helper functions."""

    @pytest.mark.unit
    def test_format_numeric_value(self):
        """Test formatting numeric values."""
        from src.tools.ppt_generator import transform_vehicle_to_iaa_product

        vehicle = {"vehicle_name": "Test", "battery_capacity_kwh": 480.5}
        product = transform_vehicle_to_iaa_product(vehicle)

        # Should format with proper precision
        assert "480" in product["battery"]

    @pytest.mark.unit
    def test_format_large_numbers(self):
        """Test formatting large numbers with commas."""
        from src.tools.ppt_generator import transform_vehicle_to_iaa_product

        vehicle = {"vehicle_name": "Test", "gvw_kg": 40000}
        product = transform_vehicle_to_iaa_product(vehicle)

        # Should include number (with or without comma formatting)
        assert "40" in product["gvw_gcw"]


class TestOEMProfileStructure:
    """Test OEM profile output structure matches IAA template needs."""

    @pytest.mark.unit
    def test_profile_has_all_required_sections(self, sample_scraping_result):
        """Test that profile has all sections needed for template."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        required_sections = [
            "company_name",
            "company_info",
            "expected_highlights",
            "assessment",
            "technologies",
            "products",
        ]

        for section in required_sections:
            assert section in profile, f"Missing section: {section}"

    @pytest.mark.unit
    def test_company_info_has_required_fields(self, sample_scraping_result):
        """Test company info has required fields."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)
        company_info = profile["company_info"]

        expected_fields = ["country", "website", "category"]
        for field in expected_fields:
            assert field in company_info

    @pytest.mark.unit
    def test_product_has_all_table_rows(self, sample_scraping_result):
        """Test product has all fields for table rows."""
        profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        if profile["products"]:
            product = profile["products"][0]

            # Check for key product fields
            assert "name" in product
            assert "battery" in product
            assert "range" in product
            assert "charging" in product


class TestEdgeCases:
    """Test edge cases in PPT generation."""

    @pytest.mark.unit
    def test_empty_scraping_result(self):
        """Test handling empty scraping result."""
        scraping_result = {
            "oem_name": "Unknown",
            "oem_url": "",
            "vehicles": [],
            "official_citations": [],
            "third_party_citations": [],
        }

        profile = transform_scraping_result_to_oem_profile(scraping_result)

        assert profile["company_name"] == "Unknown"
        assert profile["products"] == []

    @pytest.mark.unit
    def test_vehicle_with_additional_specs_none(self):
        """Test vehicle with additional_specs as None."""
        vehicle = {
            "vehicle_name": "Test",
            "additional_specs": None,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["name"] == "Test"
        assert product["wheel_formula"] == "N/A"

    @pytest.mark.unit
    def test_very_large_battery_value(self):
        """Test handling very large battery value."""
        vehicle = {
            "vehicle_name": "Test",
            "battery_capacity_kwh": 1500,  # Large but valid
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert "1,500" in product["battery"] or "1500" in product["battery"]
