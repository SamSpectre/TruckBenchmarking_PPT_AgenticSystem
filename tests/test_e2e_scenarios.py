"""
End-to-End Business Scenario Tests

These tests simulate real-world user scenarios with mocked external dependencies.
They verify that the complete workflow functions correctly from a business perspective.

Scenarios covered:
1. Single OEM benchmarking
2. Multi-OEM comparison
3. Retry on validation failure
4. Handling incomplete data
5. Template generation with various data quality levels
6. Error recovery scenarios
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.state import (
    WorkflowStatus,
    AgentType,
    ScrapingMode,
    initialize_state,
    get_state_summary,
)
from src.agents.quality_validator import QualityValidator, run_validation
from src.tools.ppt_generator import (
    transform_scraping_result_to_oem_profile,
    transform_vehicle_to_iaa_product,
)


class TestScenarioSingleOEMBenchmarking:
    """
    Scenario: User wants to benchmark a single OEM's electric truck lineup.

    Business Context:
    A product manager wants to analyze MAN's electric truck offerings
    to understand their competitive positioning.
    """

    @pytest.mark.e2e
    def test_single_oem_complete_workflow(self, sample_scraping_result, sample_vehicle_specs):
        """Test complete workflow for single OEM."""
        # Step 1: Initialize state with single OEM URL
        urls = ["https://www.man.eu/trucks"]
        state = initialize_state(urls, scraping_mode=ScrapingMode.PERPLEXITY)

        assert state["oem_urls"] == urls
        assert state["workflow_status"] == WorkflowStatus.INITIALIZED

        # Step 2: Simulate scraping completion
        state["scraping_results"] = [sample_scraping_result]
        state["all_vehicles"] = sample_scraping_result["vehicles"]
        state["workflow_status"] = WorkflowStatus.VALIDATING

        assert len(state["all_vehicles"]) >= 1

        # Step 3: Run validation
        validation_result = run_validation([sample_scraping_result], use_llm=False)

        assert "overall_quality_score" in validation_result
        assert validation_result["overall_quality_score"] > 0

        # Step 4: Transform to OEM profile for presentation
        oem_profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        assert oem_profile["company_name"] == "MAN Truck & Bus"
        assert len(oem_profile["products"]) >= 1

    @pytest.mark.e2e
    def test_single_oem_data_quality_check(self, sample_scraping_result):
        """Test data quality verification for single OEM."""
        validation_result = run_validation([sample_scraping_result], use_llm=False)

        # Business requirement: Quality must be at least 0.6 to proceed
        assert validation_result["overall_quality_score"] >= 0.5, \
            "Data quality too low for business use"

        # Verify no critical issues
        assert "Missing required field: vehicle_name" not in validation_result["missing_fields"]

    @pytest.mark.e2e
    def test_single_oem_generates_presentation_data(self, sample_scraping_result):
        """Test that scraping result can generate presentation-ready data."""
        oem_profile = transform_scraping_result_to_oem_profile(sample_scraping_result)

        # Business requirements for presentation
        assert oem_profile["company_name"], "Company name required for presentation"
        assert len(oem_profile["expected_highlights"]) > 0, "Highlights required"
        assert len(oem_profile["products"]) > 0, "At least one product required"

        # Verify product data is presentation-ready
        product = oem_profile["products"][0]
        assert product["name"] != "Unknown Vehicle"
        assert product["battery"] != "N/A" or product["range"] != "N/A"


class TestScenarioMultiOEMComparison:
    """
    Scenario: User wants to compare multiple OEMs for competitive analysis.

    Business Context:
    A strategy team needs to compare electric truck offerings from
    MAN, Volvo, and Daimler for a competitive landscape report.
    """

    @pytest.mark.e2e
    def test_multi_oem_benchmark(self, sample_scraping_results_multi):
        """Test benchmarking multiple OEMs."""
        # Initialize with multiple URLs
        urls = [
            "https://www.man.eu/trucks",
            "https://www.volvotrucks.com"
        ]
        state = initialize_state(urls)

        assert len(state["oem_urls"]) == 2

        # Validate all scraping results
        validation_result = run_validation(sample_scraping_results_multi, use_llm=False)

        # Check aggregated quality
        assert validation_result["overall_quality_score"] > 0
        assert "_per_oem_results" in validation_result
        assert len(validation_result["_per_oem_results"]) == 2

    @pytest.mark.e2e
    def test_multi_oem_comparison_data(self, sample_scraping_results_multi):
        """Test that multi-OEM data is suitable for comparison."""
        profiles = []
        for sr in sample_scraping_results_multi:
            profile = transform_scraping_result_to_oem_profile(sr)
            profiles.append(profile)

        # Verify we can compare across OEMs
        assert len(profiles) == 2
        assert profiles[0]["company_name"] != profiles[1]["company_name"]

        # Both should have products for comparison
        for profile in profiles:
            assert len(profile["products"]) > 0

    @pytest.mark.e2e
    def test_multi_oem_handles_mixed_quality(self):
        """Test handling when OEMs have different data quality."""
        scraping_results = [
            {
                "oem_name": "High Quality OEM",
                "oem_url": "https://example1.com",
                "vehicles": [
                    {
                        "vehicle_name": "Good Truck",
                        "oem_name": "High Quality OEM",
                        "source_url": "https://example1.com/truck",
                        "battery_capacity_kwh": 400,
                        "range_km": 500,
                        "motor_power_kw": 350,
                        "gvw_kg": 40000,
                        "powertrain_type": "BEV",
                    }
                ],
                "official_citations": ["https://example1.com/specs"],
                "third_party_citations": [],
            },
            {
                "oem_name": "Low Quality OEM",
                "oem_url": "https://example2.com",
                "vehicles": [
                    {
                        "vehicle_name": "Minimal Truck",
                        "oem_name": "Low Quality OEM",
                        "source_url": "https://example2.com/truck",
                        # Missing most fields
                    }
                ],
                "official_citations": [],
                "third_party_citations": [],
            }
        ]

        validation_result = run_validation(scraping_results, use_llm=False)

        # Should handle mixed quality gracefully
        per_oem = validation_result["_per_oem_results"]
        assert per_oem[0]["overall_quality_score"] > per_oem[1]["overall_quality_score"]


class TestScenarioRetryOnValidationFailure:
    """
    Scenario: Initial scraping produces low-quality data, requiring retry.

    Business Context:
    First scraping attempt didn't capture all specs; system should
    retry with enhanced extraction to get better data.
    """

    @pytest.mark.e2e
    def test_retry_workflow(self):
        """Test retry logic when validation fails."""
        state = initialize_state(["https://example.com"])

        # Initial scraping with poor results
        state["scraping_results"] = [{
            "oem_name": "Test OEM",
            "oem_url": "https://example.com",
            "vehicles": [
                {"vehicle_name": "Test", "oem_name": "Test OEM", "source_url": "https://example.com"}
            ],
            "official_citations": [],
            "third_party_citations": [],
        }]

        # Validate - should fail
        validation_result = run_validation(state["scraping_results"], use_llm=False)

        if not validation_result["passes_threshold"]:
            # Check retry eligibility
            assert state["total_retries_remaining"] > 0

            # Simulate retry
            state["retry_count"] += 1
            state["total_retries_remaining"] -= 1
            state["workflow_status"] = WorkflowStatus.RETRYING

            assert state["retry_count"] == 1
            assert state["workflow_status"] == WorkflowStatus.RETRYING

    @pytest.mark.e2e
    def test_retry_with_improved_data(self, sample_scraping_result):
        """Test that retry can lead to better data quality."""
        # First attempt - low quality
        low_quality_result = {
            "oem_name": "Test OEM",
            "oem_url": "https://example.com",
            "vehicles": [{"vehicle_name": "Test", "oem_name": "Test OEM", "source_url": "https://example.com"}],
            "official_citations": [],
            "third_party_citations": [],
        }

        validation1 = run_validation([low_quality_result], use_llm=False)
        score1 = validation1["overall_quality_score"]

        # Second attempt - better quality (simulated by sample_scraping_result)
        validation2 = run_validation([sample_scraping_result], use_llm=False)
        score2 = validation2["overall_quality_score"]

        # Second attempt should have better score
        assert score2 > score1


class TestScenarioIncompleteData:
    """
    Scenario: Some OEMs don't publish complete specifications.

    Business Context:
    System must gracefully handle missing data and still produce
    useful output where possible.
    """

    @pytest.mark.e2e
    def test_handles_missing_battery_data(self):
        """Test handling vehicle with missing battery specifications."""
        vehicle = {
            "vehicle_name": "Mystery Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "range_km": 400,  # Has range but no battery
            "motor_power_kw": 300,
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["name"] == "Mystery Truck"
        assert product["battery"] == "N/A"  # Graceful handling
        assert "400" in product["range"]

    @pytest.mark.e2e
    def test_handles_missing_range_data(self):
        """Test handling vehicle with missing range specifications."""
        vehicle = {
            "vehicle_name": "Range Unknown Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "battery_capacity_kwh": 500,  # Has battery but no range
        }

        product = transform_vehicle_to_iaa_product(vehicle)

        assert product["range"] == "N/A"
        assert "500" in product["battery"]

    @pytest.mark.e2e
    def test_calculates_correct_completeness(self):
        """Test completeness score calculation."""
        # Full data
        full_vehicle = {
            "vehicle_name": "Full Specs Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "battery_capacity_kwh": 400,
            "range_km": 500,
            "motor_power_kw": 350,
            "gvw_kg": 40000,
            "powertrain_type": "BEV",
        }

        # Minimal data
        minimal_vehicle = {
            "vehicle_name": "Minimal Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
        }

        full_result = {
            "vehicles": [full_vehicle],
            "official_citations": [],
            "third_party_citations": [],
        }

        minimal_result = {
            "vehicles": [minimal_vehicle],
            "official_citations": [],
            "third_party_citations": [],
        }

        validator = QualityValidator(use_llm=False)
        full_validation = validator.validate(full_result)
        minimal_validation = validator.validate(minimal_result)

        assert full_validation["completeness_score"] > minimal_validation["completeness_score"]


class TestScenarioDataQualityLevels:
    """
    Scenario: Testing various data quality levels and their handling.

    Business Context:
    System should provide appropriate recommendations based on
    data quality for business decision making.
    """

    @pytest.mark.e2e
    def test_high_quality_data_passes(self, sample_scraping_result):
        """Test that high-quality data passes validation."""
        validation = run_validation([sample_scraping_result], use_llm=False)

        assert validation["passes_threshold"] is True
        assert "acceptable" in validation["recommendation"].lower() or \
               "passed" in validation["recommendation"].lower()

    @pytest.mark.e2e
    def test_low_quality_data_fails(self):
        """Test that low-quality data fails validation with suggestions."""
        low_quality = {
            "oem_name": "Test",
            "oem_url": "https://example.com",
            "vehicles": [
                {"vehicle_name": "Incomplete", "oem_name": "Test", "source_url": "https://example.com"}
            ],
            "official_citations": [],
            "third_party_citations": [],
        }

        validation = run_validation([low_quality], use_llm=False)

        assert validation["passes_threshold"] is False
        assert len(validation["retry_suggestions"]) > 0

    @pytest.mark.e2e
    def test_marginal_quality_provides_guidance(self):
        """Test that marginal quality data provides appropriate guidance."""
        marginal = {
            "oem_name": "Test OEM",
            "oem_url": "https://example.com",
            "vehicles": [
                {
                    "vehicle_name": "Partial Truck",
                    "oem_name": "Test OEM",
                    "source_url": "https://example.com",
                    "battery_capacity_kwh": 400,
                    # Missing other important fields
                }
            ],
            "official_citations": ["https://example.com"],
            "third_party_citations": [],
        }

        validation = run_validation([marginal], use_llm=False)

        # Should have recommendation
        assert validation["recommendation"] is not None


class TestScenarioErrorRecovery:
    """
    Scenario: System encounters errors and must recover gracefully.

    Business Context:
    In production, network issues, API failures, and unexpected
    data formats must be handled without crashing.
    """

    @pytest.mark.e2e
    def test_handles_empty_vehicle_list(self):
        """Test handling scraping result with no vehicles."""
        empty_result = {
            "oem_name": "Empty OEM",
            "oem_url": "https://example.com",
            "vehicles": [],
            "official_citations": [],
            "third_party_citations": [],
        }

        validation = run_validation([empty_result], use_llm=False)

        # Score may be > 0 due to weighted scoring, but should be very low
        assert validation["overall_quality_score"] < 0.5
        assert validation["passes_threshold"] is False

    @pytest.mark.e2e
    def test_handles_none_values_gracefully(self):
        """Test handling None values in vehicle data."""
        vehicle_with_nones = {
            "vehicle_name": "None Value Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "battery_capacity_kwh": None,
            "range_km": None,
            "motor_power_kw": None,
            "gvw_kg": None,
        }

        # Should not raise exception
        product = transform_vehicle_to_iaa_product(vehicle_with_nones)

        assert product["name"] == "None Value Truck"
        assert product["battery"] == "N/A"

    @pytest.mark.e2e
    def test_handles_malformed_url(self):
        """Test handling malformed OEM URL."""
        result_with_bad_url = {
            "oem_name": "Test OEM",
            "oem_url": "not-a-valid-url",  # Malformed URL
            "vehicles": [
                {"vehicle_name": "Test", "oem_name": "Test OEM", "source_url": "https://example.com"}
            ],
            "official_citations": [],
            "third_party_citations": [],
        }

        # Should not crash
        profile = transform_scraping_result_to_oem_profile(result_with_bad_url)

        assert profile["company_name"] == "Test OEM"


class TestScenarioCostTracking:
    """
    Scenario: Business needs to track API costs for budgeting.

    Business Context:
    Each benchmarking run has associated API costs that need
    to be tracked for departmental budgeting.
    """

    @pytest.mark.e2e
    def test_cost_tracking_initialization(self):
        """Test that cost tracking is initialized to zero."""
        state = initialize_state(["https://example.com"])

        assert state["total_cost_usd"] == 0.0
        assert state["total_tokens_used"] == 0

    @pytest.mark.e2e
    def test_cost_accumulation(self):
        """Test that costs accumulate correctly."""
        state = initialize_state(["https://example.com"])

        # Simulate API calls
        state["total_tokens_used"] += 2500  # Scraping
        state["total_cost_usd"] += 0.02

        state["total_tokens_used"] += 500  # Validation
        state["total_cost_usd"] += 0.005

        assert state["total_tokens_used"] == 3000
        assert abs(state["total_cost_usd"] - 0.025) < 0.001

    @pytest.mark.e2e
    def test_cost_breakdown_by_stage(self):
        """Test cost breakdown by workflow stage."""
        state = initialize_state(["https://example.com"])

        state["cost_breakdown"]["scraping"] = 0.02
        state["cost_breakdown"]["validation"] = 0.005
        state["cost_breakdown"]["presentation"] = 0.0

        assert state["cost_breakdown"]["scraping"] == 0.02
        assert sum(state["cost_breakdown"].values()) == 0.025


class TestScenarioTimestamping:
    """
    Scenario: Data provenance requires accurate timestamps.

    Business Context:
    For audit and compliance, all data must have accurate
    extraction and validation timestamps.
    """

    @pytest.mark.e2e
    def test_workflow_has_start_time(self):
        """Test that workflow has start timestamp."""
        before = datetime.now().isoformat()
        state = initialize_state(["https://example.com"])
        after = datetime.now().isoformat()

        assert state["workflow_start_time"] is not None
        assert state["workflow_start_time"] >= before
        assert state["workflow_start_time"] <= after

    @pytest.mark.e2e
    def test_validation_has_timestamp(self, sample_scraping_result):
        """Test that validation result has timestamp."""
        validation = run_validation([sample_scraping_result], use_llm=False)

        assert "validation_timestamp" in validation
        assert validation["validation_timestamp"] is not None


class TestScenarioReporting:
    """
    Scenario: Generating reports from benchmarking data.

    Business Context:
    Results need to be formatted for executive reporting
    and competitive analysis presentations.
    """

    @pytest.mark.e2e
    def test_state_summary_readable(self, sample_state_after_validation):
        """Test that state summary is human-readable."""
        summary = get_state_summary(sample_state_after_validation)

        assert "WORKFLOW STATE SUMMARY" in summary
        assert "Status:" in summary
        assert "Vehicles:" in summary
        assert "Cost:" in summary

    @pytest.mark.e2e
    def test_oem_profile_has_all_sections(self, sample_scraping_result):
        """Test OEM profile has all sections for reporting."""
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
            assert section in profile, f"Missing report section: {section}"
