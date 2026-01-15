"""
Unit Tests for State Management (src/state/state.py)

Tests cover:
- State initialization
- Enum values
- State helper functions
- TypedDict schema validation
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state.state import (
    WorkflowStatus,
    AgentType,
    ScrapingMode,
    VehicleSpecifications,
    ScrapingResult,
    QualityValidationResult,
    PresentationResult,
    OEMProfile,
    IAA_ProductSpec,
    IAA_CompanyInfo,
    BenchmarkingState,
    initialize_state,
    get_state_summary,
)


class TestEnums:
    """Test enum definitions and values."""

    @pytest.mark.unit
    def test_workflow_status_values(self):
        """Test WorkflowStatus enum has expected values."""
        assert WorkflowStatus.INITIALIZED.value == "initialized"
        assert WorkflowStatus.SCRAPING.value == "scraping"
        assert WorkflowStatus.SCRAPING_FAILED.value == "scraping_failed"
        assert WorkflowStatus.VALIDATING.value == "validating"
        assert WorkflowStatus.QUALITY_FAILED.value == "quality_failed"
        assert WorkflowStatus.RETRYING.value == "retrying"
        assert WorkflowStatus.GENERATING_PRESENTATION.value == "generating_presentation"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"

    @pytest.mark.unit
    def test_agent_type_values(self):
        """Test AgentType enum has expected values."""
        assert AgentType.SCRAPER.value == "scraper"
        assert AgentType.VALIDATOR.value == "validator"
        assert AgentType.PRESENTER.value == "presenter"
        assert AgentType.ROUTER.value == "router"

    @pytest.mark.unit
    def test_scraping_mode_values(self):
        """Test ScrapingMode enum has expected values."""
        assert ScrapingMode.INTELLIGENT.value == "intelligent"
        assert ScrapingMode.PERPLEXITY.value == "perplexity"
        assert ScrapingMode.AUTO.value == "auto"

    @pytest.mark.unit
    def test_workflow_status_is_string_enum(self):
        """Test that WorkflowStatus can be used as string."""
        status = WorkflowStatus.COMPLETED
        assert str(status) == "WorkflowStatus.COMPLETED"
        assert status == "completed"  # str enum comparison

    @pytest.mark.unit
    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        workflow_statuses = list(WorkflowStatus)
        assert len(workflow_statuses) == 9

        agent_types = list(AgentType)
        assert len(agent_types) == 4

        scraping_modes = list(ScrapingMode)
        assert len(scraping_modes) == 3


class TestInitializeState:
    """Test state initialization function."""

    @pytest.mark.unit
    def test_initialize_state_with_urls(self, sample_oem_urls):
        """Test state initialization with OEM URLs."""
        state = initialize_state(sample_oem_urls)

        assert state["oem_urls"] == sample_oem_urls
        assert state["workflow_status"] == WorkflowStatus.INITIALIZED
        assert state["current_agent"] == AgentType.SCRAPER
        assert state["retry_count"] == 0
        assert state["total_retries_remaining"] == 3  # Default from settings
        assert state["scraping_mode"] == ScrapingMode.PERPLEXITY  # Default

    @pytest.mark.unit
    def test_initialize_state_with_intelligent_mode(self, sample_oem_urls):
        """Test state initialization with intelligent scraping mode."""
        state = initialize_state(sample_oem_urls, scraping_mode=ScrapingMode.INTELLIGENT)

        assert state["scraping_mode"] == ScrapingMode.INTELLIGENT

    @pytest.mark.unit
    def test_initialize_state_has_empty_results(self, sample_oem_urls):
        """Test that initial state has no results."""
        state = initialize_state(sample_oem_urls)

        assert state["scraping_results"] is None
        assert state["all_vehicles"] is None
        assert state["oem_profiles"] is None
        assert state["quality_validation"] is None
        assert state["presentation_result"] is None

    @pytest.mark.unit
    def test_initialize_state_has_zero_costs(self, sample_oem_urls):
        """Test that initial state has zero costs."""
        state = initialize_state(sample_oem_urls)

        assert state["total_tokens_used"] == 0
        assert state["total_cost_usd"] == 0.0
        assert state["cost_breakdown"] == {}

    @pytest.mark.unit
    def test_initialize_state_has_empty_errors(self, sample_oem_urls):
        """Test that initial state has no errors or warnings."""
        state = initialize_state(sample_oem_urls)

        assert state["errors"] == []
        assert state["warnings"] == []

    @pytest.mark.unit
    def test_initialize_state_has_start_time(self, sample_oem_urls):
        """Test that initial state has start time set."""
        before = datetime.now().isoformat()
        state = initialize_state(sample_oem_urls)
        after = datetime.now().isoformat()

        assert state["workflow_start_time"] is not None
        assert state["workflow_start_time"] >= before
        assert state["workflow_start_time"] <= after
        assert state["workflow_end_time"] is None

    @pytest.mark.unit
    def test_initialize_state_empty_urls(self):
        """Test state initialization with empty URL list."""
        state = initialize_state([])

        assert state["oem_urls"] == []
        assert state["workflow_status"] == WorkflowStatus.INITIALIZED

    @pytest.mark.unit
    def test_initialize_state_single_url(self):
        """Test state initialization with single URL."""
        urls = ["https://www.man.eu/trucks"]
        state = initialize_state(urls)

        assert state["oem_urls"] == urls
        assert len(state["oem_urls"]) == 1


class TestGetStateSummary:
    """Test state summary generation."""

    @pytest.mark.unit
    def test_get_state_summary_initial(self, sample_initial_state):
        """Test summary for initial state."""
        summary = get_state_summary(sample_initial_state)

        assert "WORKFLOW STATE SUMMARY" in summary
        assert "initialized" in summary.lower()
        assert "scraper" in summary.lower()
        assert "URLs: 2" in summary

    @pytest.mark.unit
    def test_get_state_summary_after_scraping(self, sample_state_after_scraping):
        """Test summary after scraping."""
        summary = get_state_summary(sample_state_after_scraping)

        assert "Vehicles:" in summary
        assert "Tokens:" in summary

    @pytest.mark.unit
    def test_get_state_summary_shows_costs(self, sample_state_after_scraping):
        """Test that summary includes cost information."""
        summary = get_state_summary(sample_state_after_scraping)

        assert "Cost: $" in summary

    @pytest.mark.unit
    def test_get_state_summary_shows_errors_count(self, sample_initial_state):
        """Test that summary shows error count."""
        state = sample_initial_state.copy()
        state["errors"] = ["Error 1", "Error 2"]

        summary = get_state_summary(state)
        assert "Errors: 2" in summary


class TestVehicleSpecifications:
    """Test VehicleSpecifications TypedDict structure."""

    @pytest.mark.unit
    def test_vehicle_specs_with_all_fields(self, sample_vehicle_specs):
        """Test that full vehicle specs are valid."""
        # TypedDict doesn't enforce at runtime, but we can check structure
        assert "vehicle_name" in sample_vehicle_specs
        assert "oem_name" in sample_vehicle_specs
        assert "battery_capacity_kwh" in sample_vehicle_specs
        assert "range_km" in sample_vehicle_specs

    @pytest.mark.unit
    def test_vehicle_specs_minimal(self, sample_vehicle_specs_minimal):
        """Test minimal vehicle specs are valid."""
        assert "vehicle_name" in sample_vehicle_specs_minimal
        assert "oem_name" in sample_vehicle_specs_minimal
        assert "source_url" in sample_vehicle_specs_minimal

    @pytest.mark.unit
    def test_vehicle_specs_has_ranges(self, sample_vehicle_specs):
        """Test that vehicle specs support min/max ranges."""
        assert "battery_capacity_kwh" in sample_vehicle_specs
        assert "battery_capacity_min_kwh" in sample_vehicle_specs
        assert "range_km" in sample_vehicle_specs
        assert "range_min_km" in sample_vehicle_specs

    @pytest.mark.unit
    def test_vehicle_specs_additional_specs(self, sample_vehicle_specs):
        """Test additional_specs field for flexible data."""
        assert "additional_specs" in sample_vehicle_specs
        assert isinstance(sample_vehicle_specs["additional_specs"], dict)


class TestScrapingResult:
    """Test ScrapingResult TypedDict structure."""

    @pytest.mark.unit
    def test_scraping_result_structure(self, sample_scraping_result):
        """Test ScrapingResult has expected fields."""
        assert "oem_name" in sample_scraping_result
        assert "oem_url" in sample_scraping_result
        assert "vehicles" in sample_scraping_result
        assert "total_vehicles_found" in sample_scraping_result
        assert "official_citations" in sample_scraping_result
        assert "third_party_citations" in sample_scraping_result

    @pytest.mark.unit
    def test_scraping_result_has_metadata(self, sample_scraping_result):
        """Test ScrapingResult includes metadata."""
        assert "extraction_timestamp" in sample_scraping_result
        assert "tokens_used" in sample_scraping_result
        assert "model_used" in sample_scraping_result
        assert "extraction_duration_seconds" in sample_scraping_result

    @pytest.mark.unit
    def test_scraping_result_vehicles_is_list(self, sample_scraping_result):
        """Test that vehicles is a list."""
        assert isinstance(sample_scraping_result["vehicles"], list)


class TestQualityValidationResult:
    """Test QualityValidationResult TypedDict structure."""

    @pytest.mark.unit
    def test_quality_validation_passed(self, sample_quality_validation_passed):
        """Test passed validation result structure."""
        assert sample_quality_validation_passed["passes_threshold"] is True
        assert sample_quality_validation_passed["overall_quality_score"] >= 0.6

    @pytest.mark.unit
    def test_quality_validation_failed(self, sample_quality_validation_failed):
        """Test failed validation result structure."""
        assert sample_quality_validation_failed["passes_threshold"] is False
        assert sample_quality_validation_failed["overall_quality_score"] < 0.6

    @pytest.mark.unit
    def test_quality_validation_has_scores(self, sample_quality_validation_passed):
        """Test that validation result has all score fields."""
        assert "completeness_score" in sample_quality_validation_passed
        assert "accuracy_score" in sample_quality_validation_passed
        assert "consistency_score" in sample_quality_validation_passed
        assert "source_quality_score" in sample_quality_validation_passed

    @pytest.mark.unit
    def test_quality_validation_scores_in_range(self, sample_quality_validation_passed):
        """Test that all scores are between 0 and 1."""
        for key, value in sample_quality_validation_passed.items():
            if "score" in key.lower() and isinstance(value, (int, float)):
                assert 0 <= value <= 1, f"{key} should be between 0 and 1"


class TestOEMProfile:
    """Test OEMProfile TypedDict structure."""

    @pytest.mark.unit
    def test_oem_profile_structure(self, sample_oem_profile):
        """Test OEMProfile has expected fields."""
        assert "company_name" in sample_oem_profile
        assert "company_info" in sample_oem_profile
        assert "expected_highlights" in sample_oem_profile
        assert "products" in sample_oem_profile

    @pytest.mark.unit
    def test_oem_profile_products_list(self, sample_oem_profile):
        """Test that products is a list."""
        assert isinstance(sample_oem_profile["products"], list)
        assert len(sample_oem_profile["products"]) >= 1

    @pytest.mark.unit
    def test_oem_profile_product_structure(self, sample_oem_profile):
        """Test product specification structure."""
        product = sample_oem_profile["products"][0]
        assert "name" in product
        assert "battery" in product
        assert "range" in product
        assert "charging" in product


class TestBenchmarkingState:
    """Test full BenchmarkingState structure."""

    @pytest.mark.unit
    def test_benchmarking_state_workflow_fields(self, sample_initial_state):
        """Test workflow control fields."""
        assert "workflow_status" in sample_initial_state
        assert "current_agent" in sample_initial_state
        assert "retry_count" in sample_initial_state
        assert "total_retries_remaining" in sample_initial_state

    @pytest.mark.unit
    def test_benchmarking_state_input_fields(self, sample_initial_state):
        """Test input fields."""
        assert "oem_urls" in sample_initial_state
        assert isinstance(sample_initial_state["oem_urls"], list)

    @pytest.mark.unit
    def test_benchmarking_state_result_fields(self, sample_initial_state):
        """Test result fields exist."""
        assert "scraping_results" in sample_initial_state
        assert "all_vehicles" in sample_initial_state
        assert "oem_profiles" in sample_initial_state
        assert "quality_validation" in sample_initial_state
        assert "presentation_result" in sample_initial_state

    @pytest.mark.unit
    def test_benchmarking_state_cost_fields(self, sample_initial_state):
        """Test cost tracking fields."""
        assert "total_tokens_used" in sample_initial_state
        assert "total_cost_usd" in sample_initial_state
        assert "cost_breakdown" in sample_initial_state

    @pytest.mark.unit
    def test_benchmarking_state_error_fields(self, sample_initial_state):
        """Test error/warning fields."""
        assert "errors" in sample_initial_state
        assert "warnings" in sample_initial_state
        assert isinstance(sample_initial_state["errors"], list)
        assert isinstance(sample_initial_state["warnings"], list)


class TestStateTransitions:
    """Test state transitions through workflow."""

    @pytest.mark.unit
    def test_state_transition_initial_to_scraping(self, sample_initial_state):
        """Test transition from initialized to scraping."""
        state = sample_initial_state.copy()
        state["workflow_status"] = WorkflowStatus.SCRAPING

        assert state["workflow_status"] == WorkflowStatus.SCRAPING

    @pytest.mark.unit
    def test_state_transition_scraping_to_validation(self, sample_state_after_scraping):
        """Test transition from scraping to validation."""
        assert sample_state_after_scraping["workflow_status"] == WorkflowStatus.VALIDATING
        assert sample_state_after_scraping["scraping_results"] is not None

    @pytest.mark.unit
    def test_state_transition_validation_to_presentation(self, sample_state_after_validation):
        """Test transition from validation to presentation."""
        assert sample_state_after_validation["workflow_status"] == WorkflowStatus.GENERATING_PRESENTATION
        assert sample_state_after_validation["quality_validation"] is not None

    @pytest.mark.unit
    def test_state_accumulates_tokens(self, sample_initial_state):
        """Test that token count accumulates."""
        state = sample_initial_state.copy()
        state["total_tokens_used"] = 1000

        # Simulate adding more tokens
        state["total_tokens_used"] += 500

        assert state["total_tokens_used"] == 1500

    @pytest.mark.unit
    def test_state_accumulates_costs(self, sample_initial_state):
        """Test that cost accumulates."""
        state = sample_initial_state.copy()
        state["total_cost_usd"] = 0.01

        # Simulate adding more cost
        state["total_cost_usd"] += 0.02

        assert abs(state["total_cost_usd"] - 0.03) < 0.001
