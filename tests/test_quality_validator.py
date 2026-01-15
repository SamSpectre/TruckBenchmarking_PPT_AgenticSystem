"""
Unit Tests for Quality Validator (src/agents/quality_validator.py)

Tests cover:
- Rule-based validation
- Vehicle validation
- Source quality scoring
- Consistency checks
- Overall quality scoring
- Validation node function
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.quality_validator import (
    ValidationConfig,
    RuleBasedValidator,
    QualityValidator,
    run_validation,
    validation_node,
)


class TestValidationConfig:
    """Test ValidationConfig dataclass."""

    @pytest.mark.unit
    def test_default_config_values(self):
        """Test default configuration values."""
        config = ValidationConfig()

        assert config.min_overall_score == 0.6
        assert config.min_completeness_score == 0.5
        assert config.min_source_quality_score == 0.3

    @pytest.mark.unit
    def test_default_required_fields(self):
        """Test default required vehicle fields."""
        config = ValidationConfig()

        expected = ["vehicle_name", "oem_name", "source_url"]
        assert config.required_vehicle_fields == expected

    @pytest.mark.unit
    def test_default_important_fields(self):
        """Test default important vehicle fields."""
        config = ValidationConfig()

        expected = ["battery_capacity_kwh", "range_km", "motor_power_kw", "gvw_kg", "powertrain_type"]
        assert config.important_vehicle_fields == expected

    @pytest.mark.unit
    def test_value_bounds_defined(self):
        """Test that value bounds are defined for key fields."""
        config = ValidationConfig()

        assert "battery_capacity_kwh" in config.value_bounds
        assert "range_km" in config.value_bounds
        assert "motor_power_kw" in config.value_bounds
        assert "gvw_kg" in config.value_bounds

    @pytest.mark.unit
    def test_value_bounds_reasonable(self):
        """Test that value bounds are reasonable."""
        config = ValidationConfig()

        # Battery: 10 to 2000 kWh
        min_bat, max_bat = config.value_bounds["battery_capacity_kwh"]
        assert min_bat == 10
        assert max_bat == 2000

        # Range: 50 to 1500 km
        min_range, max_range = config.value_bounds["range_km"]
        assert min_range == 50
        assert max_range == 1500


class TestRuleBasedValidator:
    """Test RuleBasedValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleBasedValidator()

    @pytest.mark.unit
    def test_validate_vehicle_valid(self, validator, sample_vehicle_specs):
        """Test validation of valid vehicle."""
        result = validator.validate_vehicle(sample_vehicle_specs)

        assert result["is_valid"] is True
        assert result["completeness_score"] > 0.5
        assert len(result["issues"]) == 0

    @pytest.mark.unit
    def test_validate_vehicle_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        vehicle = {
            "battery_capacity_kwh": 400,
            "range_km": 500,
        }

        result = validator.validate_vehicle(vehicle)

        assert result["is_valid"] is False
        assert any("vehicle_name" in issue for issue in result["issues"])

    @pytest.mark.unit
    def test_validate_vehicle_missing_important_fields(self, validator):
        """Test validation with missing important fields."""
        vehicle = {
            "vehicle_name": "Test Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            # Missing: battery_capacity_kwh, range_km, motor_power_kw, etc.
        }

        result = validator.validate_vehicle(vehicle)

        assert result["is_valid"] is True  # Required fields present
        assert result["completeness_score"] < 0.5  # But completeness is low
        assert len(result["warnings"]) > 0

    @pytest.mark.unit
    def test_validate_vehicle_suspicious_battery(self, validator):
        """Test detection of suspicious battery value."""
        vehicle = {
            "vehicle_name": "Test Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "battery_capacity_kwh": 5000,  # Too high (max is 2000)
        }

        result = validator.validate_vehicle(vehicle)

        assert any("Suspicious" in w and "battery" in w.lower() for w in result["warnings"])

    @pytest.mark.unit
    def test_validate_vehicle_suspicious_range(self, validator):
        """Test detection of suspicious range value."""
        vehicle = {
            "vehicle_name": "Test Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "range_km": 3000,  # Too high (max is 1500)
        }

        result = validator.validate_vehicle(vehicle)

        assert any("Suspicious" in w and "range" in w.lower() for w in result["warnings"])

    @pytest.mark.unit
    def test_validate_vehicle_suspicious_gvw(self, validator):
        """Test detection of suspicious GVW value."""
        vehicle = {
            "vehicle_name": "Test Truck",
            "oem_name": "Test OEM",
            "source_url": "https://example.com",
            "gvw_kg": 500,  # Too low (min is 2000)
        }

        result = validator.validate_vehicle(vehicle)

        assert any("Suspicious" in w and "gvw" in w.lower() for w in result["warnings"])


class TestSourceQualityValidation:
    """Test source quality validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleBasedValidator()

    @pytest.mark.unit
    def test_source_quality_with_official_citations(self, validator):
        """Test source quality with matching official citations."""
        scraping_result = {
            "oem_url": "https://www.man.eu/trucks",
            "official_citations": [
                "https://www.man.eu/trucks/etgx",
                "https://www.man.eu/trucks/specs"
            ],
            "third_party_citations": [],
        }

        result = validator.validate_source_quality(scraping_result)

        assert result["score"] >= 0.7
        assert result["official_count"] == 2
        assert "official" in result["assessment"].lower()

    @pytest.mark.unit
    def test_source_quality_with_third_party_only(self, validator):
        """Test source quality with only third-party citations."""
        scraping_result = {
            "oem_url": "https://www.man.eu/trucks",
            "official_citations": [],
            "third_party_citations": [
                "https://electrek.co/man",
                "https://insideevs.com/man"
            ],
        }

        result = validator.validate_source_quality(scraping_result)

        assert result["score"] <= 0.3
        assert result["third_party_count"] == 2
        assert "third-party" in result["assessment"].lower()

    @pytest.mark.unit
    def test_source_quality_no_citations(self, validator):
        """Test source quality with no citations."""
        scraping_result = {
            "oem_url": "https://www.man.eu/trucks",
            "official_citations": [],
            "third_party_citations": [],
        }

        result = validator.validate_source_quality(scraping_result)

        assert result["score"] == 0.0
        assert "No citations" in result["assessment"]


class TestConsistencyValidation:
    """Test consistency validation across vehicles."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleBasedValidator()

    @pytest.mark.unit
    def test_consistency_single_vehicle(self, validator, sample_vehicle_specs):
        """Test consistency check with single vehicle."""
        result = validator.check_consistency([sample_vehicle_specs])

        assert result["score"] == 1.0
        assert len(result["issues"]) == 0

    @pytest.mark.unit
    def test_consistency_matching_oem_names(self, validator):
        """Test consistency with matching OEM names."""
        vehicles = [
            {"oem_name": "MAN Truck & Bus", "vehicle_name": "eTGX"},
            {"oem_name": "MAN Truck & Bus", "vehicle_name": "eTGM"},
        ]

        result = validator.check_consistency(vehicles)

        assert result["score"] == 1.0

    @pytest.mark.unit
    def test_consistency_mismatching_oem_names(self, validator):
        """Test consistency with mismatching OEM names."""
        vehicles = [
            {"oem_name": "MAN Truck & Bus", "vehicle_name": "eTGX"},
            {"oem_name": "Volvo Trucks", "vehicle_name": "FH Electric"},  # Different OEM!
        ]

        result = validator.check_consistency(vehicles)

        assert result["score"] < 1.0
        assert any("Inconsistent OEM" in issue for issue in result["issues"])

    @pytest.mark.unit
    def test_consistency_duplicate_vehicle_names(self, validator):
        """Test consistency with duplicate vehicle names."""
        vehicles = [
            {"oem_name": "MAN Truck & Bus", "vehicle_name": "eTGX"},
            {"oem_name": "MAN Truck & Bus", "vehicle_name": "eTGX"},  # Duplicate!
        ]

        result = validator.check_consistency(vehicles)

        assert result["score"] < 1.0
        assert any("Duplicate" in issue for issue in result["issues"])


class TestOverallValidation:
    """Test overall validation workflow."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleBasedValidator()

    @pytest.mark.unit
    def test_validate_scraping_result(self, validator, sample_scraping_result):
        """Test full validation of scraping result."""
        result = validator.validate(sample_scraping_result)

        assert "overall_quality_score" in result
        assert "passes_threshold" in result
        assert "completeness_score" in result
        assert "accuracy_score" in result
        assert "recommendation" in result

    @pytest.mark.unit
    def test_validate_scraping_result_passes(self, validator, sample_scraping_result):
        """Test that good data passes validation."""
        result = validator.validate(sample_scraping_result)

        assert result["overall_quality_score"] >= 0.6
        assert result["passes_threshold"] is True

    @pytest.mark.unit
    def test_validate_empty_vehicles(self, validator):
        """Test validation with empty vehicles list."""
        scraping_result = {
            "oem_name": "Test OEM",
            "oem_url": "https://example.com",
            "vehicles": [],
            "official_citations": [],
            "third_party_citations": [],
        }

        result = validator.validate(scraping_result)

        # Score may be > 0 due to weighted scoring, but should be very low
        assert result["overall_quality_score"] < 0.5
        assert result["passes_threshold"] is False

    @pytest.mark.unit
    def test_validate_generates_recommendations(self, validator):
        """Test that validation generates recommendations."""
        scraping_result = {
            "oem_name": "Test OEM",
            "oem_url": "https://example.com",
            "vehicles": [
                {"vehicle_name": "Test", "oem_name": "Test OEM", "source_url": "https://example.com"}
            ],
            "official_citations": [],
            "third_party_citations": [],
        }

        result = validator.validate(scraping_result)

        assert "recommendation" in result
        assert len(result["retry_suggestions"]) > 0


class TestQualityValidator:
    """Test main QualityValidator class."""

    @pytest.mark.unit
    def test_quality_validator_rule_based_only(self, sample_scraping_result):
        """Test QualityValidator with rule-based validation only."""
        validator = QualityValidator(use_llm=False)
        result = validator.validate(sample_scraping_result)

        assert "overall_quality_score" in result
        assert "_llm_validation" not in result  # LLM not used

    @pytest.mark.unit
    def test_quality_validator_with_custom_config(self, sample_scraping_result):
        """Test QualityValidator with custom config."""
        config = ValidationConfig(min_overall_score=0.8)  # Higher threshold
        validator = QualityValidator(use_llm=False, config=config)
        result = validator.validate(sample_scraping_result)

        # With higher threshold, might not pass
        assert "overall_quality_score" in result


class TestRunValidation:
    """Test run_validation standalone function."""

    @pytest.mark.unit
    def test_run_validation_single_result(self, sample_scraping_result):
        """Test run_validation with single scraping result."""
        result = run_validation([sample_scraping_result], use_llm=False)

        assert "overall_quality_score" in result
        assert "passes_threshold" in result

    @pytest.mark.unit
    def test_run_validation_multiple_results(self, sample_scraping_results_multi):
        """Test run_validation with multiple scraping results."""
        result = run_validation(sample_scraping_results_multi, use_llm=False)

        assert "overall_quality_score" in result
        assert "_per_oem_results" in result
        assert len(result["_per_oem_results"]) == 2

    @pytest.mark.unit
    def test_run_validation_empty_results(self):
        """Test run_validation with empty results."""
        result = run_validation([], use_llm=False)

        assert result["overall_quality_score"] == 0.0
        assert result["passes_threshold"] is False

    @pytest.mark.unit
    def test_run_validation_aggregates_scores(self, sample_scraping_results_multi):
        """Test that scores are properly aggregated."""
        result = run_validation(sample_scraping_results_multi, use_llm=False)

        # Check that aggregated scores are between 0 and 1
        assert 0 <= result["overall_quality_score"] <= 1
        assert 0 <= result["completeness_score"] <= 1
        assert 0 <= result["accuracy_score"] <= 1


class TestValidationNode:
    """Test validation_node LangGraph function."""

    @pytest.mark.unit
    def test_validation_node_with_results(self, sample_state_after_scraping):
        """Test validation node with scraping results."""
        from src.state.state import WorkflowStatus, AgentType

        result = validation_node(sample_state_after_scraping)

        assert result["workflow_status"] in [
            WorkflowStatus.GENERATING_PRESENTATION,
            WorkflowStatus.RETRYING,
            WorkflowStatus.QUALITY_FAILED,
        ]
        assert "quality_validation" in result

    @pytest.mark.unit
    def test_validation_node_no_results(self, sample_initial_state):
        """Test validation node with no scraping results."""
        from src.state.state import WorkflowStatus

        result = validation_node(sample_initial_state)

        assert result["workflow_status"] == WorkflowStatus.QUALITY_FAILED
        assert "No scraping results" in str(result["errors"])

    @pytest.mark.unit
    def test_validation_node_updates_agent(self, sample_state_after_scraping):
        """Test that validation node updates current agent."""
        from src.state.state import AgentType

        result = validation_node(sample_state_after_scraping)

        assert result["current_agent"] == AgentType.VALIDATOR

    @pytest.mark.unit
    def test_validation_node_retry_logic(self, sample_state_after_scraping):
        """Test validation node retry count updates."""
        from src.state.state import WorkflowStatus

        # Modify state to have low quality results
        state = sample_state_after_scraping.copy()
        state["scraping_results"] = [{
            "oem_name": "Test",
            "oem_url": "https://example.com",
            "vehicles": [{"vehicle_name": "Test", "oem_name": "Test", "source_url": "https://example.com"}],
            "official_citations": [],
            "third_party_citations": [],
        }]

        result = validation_node(state)

        # Should trigger retry if quality is low and retries available
        if result["workflow_status"] == WorkflowStatus.RETRYING:
            assert result["total_retries_remaining"] < state["total_retries_remaining"]


class TestValidationScoring:
    """Test specific scoring calculations."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleBasedValidator()

    @pytest.mark.unit
    def test_completeness_score_full_data(self, validator, sample_vehicle_specs):
        """Test completeness score with full data."""
        result = validator.validate_vehicle(sample_vehicle_specs)

        # Should have high completeness with all fields present
        assert result["completeness_score"] >= 0.8

    @pytest.mark.unit
    def test_completeness_score_minimal_data(self, validator, sample_vehicle_specs_minimal):
        """Test completeness score with minimal data."""
        result = validator.validate_vehicle(sample_vehicle_specs_minimal)

        # Should have low completeness with only required fields
        assert result["completeness_score"] < 0.5

    @pytest.mark.unit
    def test_accuracy_score_all_valid(self, validator, sample_vehicles_list):
        """Test accuracy score when all vehicles are valid."""
        scraping_result = {
            "oem_name": "Test",
            "oem_url": "https://example.com",
            "vehicles": sample_vehicles_list,
            "official_citations": ["https://example.com"],
            "third_party_citations": [],
        }

        result = validator.validate(scraping_result)

        assert result["accuracy_score"] == 1.0  # All vehicles valid

    @pytest.mark.unit
    def test_weighted_overall_score(self, validator):
        """Test that overall score is weighted correctly."""
        scraping_result = {
            "oem_name": "Test OEM",
            "oem_url": "https://www.test.com",
            "vehicles": [{
                "vehicle_name": "Test",
                "oem_name": "Test OEM",
                "source_url": "https://www.test.com/vehicle",
                "battery_capacity_kwh": 400,
                "range_km": 500,
                "motor_power_kw": 300,
                "gvw_kg": 40000,
                "powertrain_type": "BEV",
            }],
            "official_citations": ["https://www.test.com/specs"],
            "third_party_citations": [],
        }

        result = validator.validate(scraping_result)

        # Verify overall score is a weighted average
        # Weights: completeness 0.35, accuracy 0.25, consistency 0.15, source 0.25
        expected_overall = (
            result["completeness_score"] * 0.35 +
            result["accuracy_score"] * 0.25 +
            result["consistency_score"] * 0.15 +
            result["source_quality_score"] * 0.25
        )

        assert abs(result["overall_quality_score"] - expected_overall) < 0.01
