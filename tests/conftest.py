"""
Pytest Configuration and Fixtures for E-Powertrain Benchmarking System

This module provides shared fixtures and test data for all test modules.
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =====================================================================
# MOCK DATA FIXTURES
# =====================================================================

@pytest.fixture
def sample_vehicle_specs() -> Dict[str, Any]:
    """Sample vehicle specifications matching VehicleSpecifications schema."""
    return {
        "vehicle_name": "MAN eTGX 4x2",
        "oem_name": "MAN Truck & Bus",
        "category": "Heavy-duty Truck",
        "powertrain_type": "BEV",
        "source_url": "https://www.man.eu/trucks/etgx",
        "battery_capacity_kwh": 480.0,
        "battery_capacity_min_kwh": 320.0,
        "battery_voltage_v": 800.0,
        "motor_power_kw": 480.0,
        "motor_torque_nm": 3100.0,
        "range_km": 600.0,
        "range_min_km": 400.0,
        "dc_charging_kw": 375.0,
        "gvw_kg": 40000.0,
        "payload_capacity_kg": 22000.0,
        "extraction_timestamp": datetime.now().isoformat(),
        "data_completeness_score": 0.85,
        "additional_specs": {
            "wheel_formula": "4x2",
            "wheelbase_mm": 3600,
        }
    }


@pytest.fixture
def sample_vehicle_specs_minimal() -> Dict[str, Any]:
    """Minimal vehicle specs with only required fields."""
    return {
        "vehicle_name": "Test Vehicle",
        "oem_name": "Test OEM",
        "source_url": "https://example.com/vehicle",
        "extraction_timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_vehicle_specs_invalid() -> Dict[str, Any]:
    """Invalid vehicle specs for testing validation failures."""
    return {
        "vehicle_name": "Invalid Truck",
        "oem_name": "Test OEM",
        "source_url": "https://example.com/invalid",
        "battery_capacity_kwh": 50000.0,  # Unrealistic - too high
        "range_km": 5000.0,  # Unrealistic - too high
        "gvw_kg": 500.0,  # Unrealistic - too low
        "extraction_timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_vehicles_list(sample_vehicle_specs) -> List[Dict[str, Any]]:
    """List of sample vehicles for multi-vehicle tests."""
    vehicle1 = sample_vehicle_specs.copy()
    vehicle2 = {
        "vehicle_name": "MAN eTGM 4x2",
        "oem_name": "MAN Truck & Bus",
        "category": "Medium-duty Truck",
        "powertrain_type": "BEV",
        "source_url": "https://www.man.eu/trucks/etgm",
        "battery_capacity_kwh": 264.0,
        "motor_power_kw": 264.0,
        "range_km": 300.0,
        "dc_charging_kw": 150.0,
        "gvw_kg": 26000.0,
        "extraction_timestamp": datetime.now().isoformat(),
        "data_completeness_score": 0.75,
    }
    vehicle3 = {
        "vehicle_name": "Volvo FH Electric",
        "oem_name": "Volvo Trucks",
        "category": "Heavy-duty Truck",
        "powertrain_type": "BEV",
        "source_url": "https://www.volvotrucks.com/fh-electric",
        "battery_capacity_kwh": 540.0,
        "motor_power_kw": 490.0,
        "range_km": 500.0,
        "dc_charging_kw": 250.0,
        "gvw_kg": 44000.0,
        "extraction_timestamp": datetime.now().isoformat(),
        "data_completeness_score": 0.80,
    }
    return [vehicle1, vehicle2, vehicle3]


@pytest.fixture
def sample_scraping_result(sample_vehicle_specs) -> Dict[str, Any]:
    """Sample ScrapingResult for testing."""
    return {
        "oem_name": "MAN Truck & Bus",
        "oem_url": "https://www.man.eu/trucks",
        "vehicles": [sample_vehicle_specs],
        "total_vehicles_found": 1,
        "extraction_timestamp": datetime.now().isoformat(),
        "official_citations": [
            "https://www.man.eu/trucks/etgx/specs",
            "https://www.man.eu/trucks/etgx/battery"
        ],
        "third_party_citations": [],
        "source_compliance_score": 0.85,
        "raw_content": "# MAN eTGX Technical Specifications...",
        "pages_crawled": 3,
        "spec_urls_found": ["https://www.man.eu/trucks/etgx"],
        "extraction_details": [],
        "fetched_content_length": 15000,
        "tokens_used": 2500,
        "model_used": "sonar-pro",
        "extraction_duration_seconds": 12.5,
        "errors": [],
        "warnings": [],
    }


@pytest.fixture
def sample_scraping_results_multi(sample_scraping_result, sample_vehicles_list) -> List[Dict[str, Any]]:
    """Multiple scraping results for testing multi-OEM scenarios."""
    result1 = sample_scraping_result.copy()
    result1["vehicles"] = [sample_vehicles_list[0], sample_vehicles_list[1]]
    result1["total_vehicles_found"] = 2

    result2 = {
        "oem_name": "Volvo Trucks",
        "oem_url": "https://www.volvotrucks.com",
        "vehicles": [sample_vehicles_list[2]],
        "total_vehicles_found": 1,
        "extraction_timestamp": datetime.now().isoformat(),
        "official_citations": ["https://www.volvotrucks.com/fh-electric"],
        "third_party_citations": [],
        "source_compliance_score": 0.80,
        "raw_content": "# Volvo FH Electric...",
        "pages_crawled": 2,
        "spec_urls_found": [],
        "extraction_details": [],
        "fetched_content_length": 12000,
        "tokens_used": 2000,
        "model_used": "sonar-pro",
        "extraction_duration_seconds": 10.0,
        "errors": [],
        "warnings": [],
    }

    return [result1, result2]


@pytest.fixture
def sample_quality_validation_passed() -> Dict[str, Any]:
    """Sample QualityValidationResult that passes threshold."""
    return {
        "overall_quality_score": 0.82,
        "passes_threshold": True,
        "completeness_score": 0.85,
        "accuracy_score": 1.0,
        "consistency_score": 1.0,
        "source_quality_score": 0.70,
        "missing_fields": [],
        "suspicious_values": [],
        "low_quality_vehicles": [],
        "recommendation": "Data quality acceptable. Proceed to presentation generation.",
        "retry_suggestions": [],
        "validation_timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_quality_validation_failed() -> Dict[str, Any]:
    """Sample QualityValidationResult that fails threshold."""
    return {
        "overall_quality_score": 0.45,
        "passes_threshold": False,
        "completeness_score": 0.40,
        "accuracy_score": 0.50,
        "consistency_score": 0.80,
        "source_quality_score": 0.20,
        "missing_fields": ["Missing required field: battery_capacity_kwh"],
        "suspicious_values": [
            {"vehicle": "Test Vehicle", "issue": "Suspicious value for range_km: 5000"}
        ],
        "low_quality_vehicles": ["Test Vehicle"],
        "recommendation": "Data quality too low. Retry required.",
        "retry_suggestions": ["Request more specific technical specifications"],
        "validation_timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_oem_profile() -> Dict[str, Any]:
    """Sample OEMProfile for presentation generation."""
    return {
        "company_name": "MAN Truck & Bus",
        "company_info": {
            "country": "Germany",
            "address": "Munich, Germany",
            "website": "www.man.eu",
            "booth": "Hall 5, A12",
            "category": "OEM - BEV",
        },
        "expected_highlights": [
            "2 Battery Electric Vehicles in portfolio",
            "Up to 600 km range capability",
            "Battery capacity up to 480 kWh",
        ],
        "assessment": [
            "Strong BEV portfolio for long-haul applications",
            "High data quality from official sources",
        ],
        "technologies": [
            "800V architecture",
            "CCS fast charging up to 375 kW",
        ],
        "products": [
            {
                "name": "MAN eTGX 4x2",
                "wheel_formula": "4x2",
                "wheelbase": "3,600 mm",
                "gvw_gcw": "40,000 kg GVW",
                "range": "600 km",
                "battery": "480 kWh",
                "fuel_cell": "N/A",
                "h2_tank": "N/A",
                "charging": "375 kW DC",
                "performance": "480 kW",
                "powertrain": "Electric",
                "sop": "2024",
                "markets": "EU",
                "application": "Long-haul trucking",
            }
        ],
        "cooperations": [],
        "source_url": "https://www.man.eu/trucks",
        "extraction_timestamp": datetime.now().isoformat(),
        "data_quality_score": 0.85,
    }


# =====================================================================
# STATE FIXTURES
# =====================================================================

@pytest.fixture
def sample_initial_state() -> Dict[str, Any]:
    """Sample initial BenchmarkingState."""
    from src.state.state import WorkflowStatus, AgentType, ScrapingMode

    return {
        "messages": [],
        "workflow_status": WorkflowStatus.INITIALIZED,
        "current_agent": AgentType.SCRAPER,
        "retry_count": 0,
        "total_retries_remaining": 3,
        "scraping_mode": ScrapingMode.PERPLEXITY,
        "oem_urls": [
            "https://www.man.eu/trucks",
            "https://www.volvotrucks.com"
        ],
        "scraping_results": None,
        "all_vehicles": None,
        "oem_profiles": None,
        "quality_validation": None,
        "presentation_result": None,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        "cost_breakdown": {},
        "errors": [],
        "warnings": [],
        "workflow_start_time": datetime.now().isoformat(),
        "workflow_end_time": None,
        "execution_duration_seconds": None,
    }


@pytest.fixture
def sample_state_after_scraping(sample_initial_state, sample_scraping_results_multi, sample_vehicles_list):
    """State after successful scraping."""
    from src.state.state import WorkflowStatus, AgentType

    state = sample_initial_state.copy()
    state.update({
        "workflow_status": WorkflowStatus.VALIDATING,
        "current_agent": AgentType.VALIDATOR,
        "scraping_results": sample_scraping_results_multi,
        "all_vehicles": sample_vehicles_list,
        "total_tokens_used": 4500,
        "total_cost_usd": 0.025,
    })
    return state


@pytest.fixture
def sample_state_after_validation(sample_state_after_scraping, sample_quality_validation_passed):
    """State after successful validation."""
    from src.state.state import WorkflowStatus, AgentType

    state = sample_state_after_scraping.copy()
    state.update({
        "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
        "current_agent": AgentType.PRESENTER,
        "quality_validation": sample_quality_validation_passed,
    })
    return state


# =====================================================================
# MOCK FIXTURES
# =====================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"vehicles": []}'
    mock_response.usage.prompt_tokens = 500
    mock_response.usage.completion_tokens = 200
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_perplexity_response():
    """Mock Perplexity API response."""
    return {
        "id": "test-id",
        "model": "sonar-pro",
        "choices": [{
            "message": {
                "content": '''```json
{
    "vehicles": [
        {
            "vehicle_name": "MAN eTGX",
            "battery_capacity_kwh": 480,
            "range_km": 600,
            "motor_power_kw": 480
        }
    ]
}
```''',
                "role": "assistant"
            }
        }],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500
        },
        "citations": [
            "https://www.man.eu/trucks/etgx"
        ]
    }


@pytest.fixture
def mock_crawl4ai_result():
    """Mock Crawl4AI result."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.markdown = """
# MAN eTGX Electric Truck

## Technical Specifications

- Battery: 480 kWh
- Range: up to 600 km
- Motor Power: 480 kW
- GVW: 40,000 kg
"""
    mock_result.html = "<html><body>...</body></html>"
    mock_result.title = "MAN eTGX"
    return mock_result


@pytest.fixture
def mock_pptx_presentation():
    """Mock python-pptx Presentation object."""
    mock_prs = MagicMock()
    mock_slide = MagicMock()
    mock_prs.slides = [mock_slide]
    mock_slide.shapes = []
    return mock_prs


# =====================================================================
# PATH FIXTURES
# =====================================================================

@pytest.fixture
def project_root_path() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_output_dir(project_root_path, tmp_path) -> Path:
    """Temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_template_path(project_root_path) -> Path:
    """Path to the IAA template."""
    return project_root_path / "templates" / "IAA_Template.pptx"


# =====================================================================
# ENVIRONMENT FIXTURES
# =====================================================================

@pytest.fixture
def mock_env_with_api_keys(monkeypatch):
    """Set mock API keys for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test-key-12345")


@pytest.fixture
def mock_env_without_api_keys(monkeypatch):
    """Remove API keys for testing error handling."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)


# =====================================================================
# URL FIXTURES
# =====================================================================

@pytest.fixture
def sample_oem_urls() -> List[str]:
    """Sample OEM URLs for testing."""
    return [
        "https://www.man.eu/trucks",
        "https://www.volvotrucks.com/en-en/trucks/electric.html",
        "https://www.daimler-truck.com/products/trucks/mercedes-benz/eactros.html",
    ]


@pytest.fixture
def sample_invalid_urls() -> List[str]:
    """Invalid URLs for testing error handling."""
    return [
        "not-a-url",
        "ftp://invalid-protocol.com",
        "",
        "http://",
    ]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def assert_valid_vehicle_specs(vehicle: Dict[str, Any]):
    """Assert that a vehicle dict has valid structure."""
    assert "vehicle_name" in vehicle
    assert "oem_name" in vehicle
    assert "source_url" in vehicle


def assert_valid_scraping_result(result: Dict[str, Any]):
    """Assert that a scraping result has valid structure."""
    assert "oem_name" in result
    assert "vehicles" in result
    assert isinstance(result["vehicles"], list)


def assert_valid_quality_validation(validation: Dict[str, Any]):
    """Assert that a quality validation result has valid structure."""
    assert "overall_quality_score" in validation
    assert "passes_threshold" in validation
    assert 0 <= validation["overall_quality_score"] <= 1


# Make helper functions available to tests
@pytest.fixture
def validation_helpers():
    """Provide validation helper functions to tests."""
    return {
        "assert_valid_vehicle_specs": assert_valid_vehicle_specs,
        "assert_valid_scraping_result": assert_valid_scraping_result,
        "assert_valid_quality_validation": assert_valid_quality_validation,
    }
