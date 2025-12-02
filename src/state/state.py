"""
State Management for E-Powertrain Benchmarking System

This defines the state structure that flows through all agents.
State = the "memory" that gets passed from agent to agent.

UPDATED: Added OEMProfile schema for IAA template compatibility
"""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, List, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
from enum import Enum


# =====================================================================
# ENUMS - Predefined choices
# =====================================================================

class WorkflowStatus(str, Enum):
    """Current workflow status"""
    INITIALIZED = "initialized"
    SCRAPING = "scraping"
    SCRAPING_FAILED = "scraping_failed"
    VALIDATING = "validating"
    QUALITY_FAILED = "quality_failed"
    RETRYING = "retrying"
    GENERATING_PRESENTATION = "generating_presentation"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    """Which agent is currently active"""
    SCRAPER = "scraper"
    VALIDATOR = "validator"
    PRESENTER = "presenter"
    ROUTER = "router"


# =====================================================================
# VEHICLE DATA - Flexible structure for open-ended scraping
# =====================================================================

class VehicleSpecifications(TypedDict, total=False):
    """
    Technical specifications for one vehicle.
    
    IMPORTANT: total=False makes ALL fields optional.
    This is intentional - different OEMs have different specs.
    Your scraper extracts whatever exists on the website.
    
    Common fields are listed below, but vehicles can have
    additional fields in 'additional_specs' dictionary.
    """
    # Identification
    vehicle_name: str
    oem_name: str
    category: Optional[str]
    powertrain_type: Optional[str]
    source_url: str

    # Battery
    battery_capacity_kwh: Optional[float]
    battery_capacity_min_kwh: Optional[float]  # NEW: Min for ranges like "240-560 kWh"
    battery_voltage_v: Optional[float]
    battery_chemistry: Optional[str]  # e.g., "NMC", "LFP"

    # Motor
    motor_power_kw: Optional[float]
    motor_torque_nm: Optional[float]

    # Range
    range_km: Optional[float]
    range_min_km: Optional[float]  # NEW: Min for ranges like "500-750 km"
    energy_consumption_kwh_per_100km: Optional[float]

    # Charging
    dc_charging_kw: Optional[float]
    charging_time_minutes: Optional[Dict[str, float]]

    # Vehicle
    gvw_kg: Optional[float]
    payload_capacity_kg: Optional[float]
    available_configurations: Optional[List[str]]  # NEW: e.g., ["4x2", "6x2"]

    # FLEXIBLE fields
    additional_specs: Optional[Dict[str, Any]]
    raw_table_data: Optional[str]

    # Metadata
    extraction_timestamp: str
    data_completeness_score: Optional[float]


# =====================================================================
# IAA TEMPLATE COMPATIBLE STRUCTURES
# =====================================================================

class IAA_CompanyInfo(TypedDict, total=False):
    """Company information for IAA template"""
    country: str
    address: str
    website: str
    booth: str           # e.g., "Hall 5, Stand A12"
    category: str        # e.g., "OEM - BEV", "Tier 1 Supplier"


class IAA_ProductSpec(TypedDict, total=False):
    """
    Product specification matching IAA template table structure.
    Maps to the 14-row product table in the template.
    """
    name: str                # Row 0: Description/Product name
    wheel_formula: str       # Row 1: e.g., "6x4", "4x2"
    wheelbase: str           # Row 2: e.g., "4,000 mm"
    gvw_gcw: str             # Row 3: e.g., "37,000 kg GCW"
    range: str               # Row 4: e.g., "800 km"
    battery: str             # Row 5: e.g., "900 kWh"
    fuel_cell: str           # Row 6: e.g., "N/A" or "100 kW"
    h2_tank: str             # Row 7: e.g., "N/A" or "32 kg"
    charging: str            # Row 8: e.g., "Megacharger 1MW"
    performance: str         # Row 9: e.g., "500 kW peak"
    powertrain: str          # Row 10: e.g., "3x Electric Motors"
    sop: str                 # Row 11: Start of Production, e.g., "2024"
    markets: str             # Row 12: e.g., "NA, EU, APAC"
    application: str         # Row 13: e.g., "Long-haul trucking"


class OEMProfile(TypedDict, total=False):
    """
    Complete OEM profile for IAA template generation.
    This is the TARGET schema that ppt_generator expects.
    
    The transformation: VehicleSpecifications[] -> OEMProfile
    happens in the presentation_generator agent.
    """
    # Required
    company_name: str
    
    # Company info table (6 rows)
    company_info: IAA_CompanyInfo
    
    # Bullet list sections
    expected_highlights: List[str]
    assessment: List[str]
    technologies: List[str]
    
    # Product specifications (up to 2 products)
    products: List[IAA_ProductSpec]
    
    # Optional sections
    cooperations: List[str]
    
    # Metadata
    source_url: str
    extraction_timestamp: str
    data_quality_score: float


# =====================================================================
# SCRAPING & VALIDATION RESULTS
# =====================================================================

class ScrapingResult(TypedDict, total=False):
    """Results from scraping agent"""
    oem_name: str
    oem_url: str
    vehicles: List[VehicleSpecifications]
    total_vehicles_found: int
    extraction_timestamp: str

    # Source validation
    official_citations: List[str]
    third_party_citations: List[str]
    source_compliance_score: float

    # Raw content (markdown from Perplexity)
    raw_content: str

    # NEW: Intelligent navigation fields
    pages_crawled: int  # Number of pages crawled
    spec_urls_found: List[str]  # URLs that were identified as spec pages
    extraction_details: List[Dict[str, Any]]  # Details per page

    # Metadata
    fetched_content_length: int
    tokens_used: int
    model_used: str
    extraction_duration_seconds: float

    # Errors
    errors: List[str]
    warnings: List[str]


class QualityValidationResult(TypedDict):
    """Results from quality validator"""
    overall_quality_score: float
    passes_threshold: bool
    
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    source_quality_score: float
    
    missing_fields: List[str]
    suspicious_values: List[Dict[str, Any]]
    low_quality_vehicles: List[str]
    
    recommendation: str
    retry_suggestions: List[str]
    
    validation_timestamp: str


class PresentationResult(TypedDict, total=False):
    """Results from presentation generator"""
    presentation_path: str
    all_presentation_paths: List[str]  # NEW: All generated presentation paths
    slides_created: int
    vehicles_included: int
    oems_compared: int

    includes_charts: bool
    includes_comparison_table: bool

    generation_timestamp: str
    generation_duration_seconds: float

    errors: List[str]  # NEW: Any errors during generation


# =====================================================================
# MAIN WORKFLOW STATE
# =====================================================================

class ScrapingMode(str, Enum):
    """Scraping mode selection"""
    INTELLIGENT = "intelligent"  # Multi-page LLM-guided navigation
    PERPLEXITY = "perplexity"    # Single-page Perplexity extraction (improved)
    AUTO = "auto"                # Auto-select based on URL


class BenchmarkingState(TypedDict, total=False):
    """Complete workflow state"""

    # Conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Workflow control
    workflow_status: WorkflowStatus
    current_agent: AgentType
    retry_count: int
    total_retries_remaining: int
    scraping_mode: ScrapingMode  # NEW: Mode selection for scraper

    # Input
    oem_urls: List[str]
    
    # Results - Original vehicle-centric data
    scraping_results: Optional[List[ScrapingResult]]
    all_vehicles: Optional[List[VehicleSpecifications]]
    
    # Results - Transformed OEM profiles for presentation
    oem_profiles: Optional[List[OEMProfile]]
    
    # Validation & Output
    quality_validation: Optional[QualityValidationResult]
    presentation_result: Optional[PresentationResult]
    
    # Cost tracking
    total_tokens_used: int
    total_cost_usd: float
    cost_breakdown: Dict[str, float]
    
    # Errors
    errors: List[str]
    warnings: List[str]
    
    # Metadata
    workflow_start_time: str
    workflow_end_time: Optional[str]
    execution_duration_seconds: Optional[float]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def initialize_state(
    oem_urls: List[str],
    scraping_mode: ScrapingMode = ScrapingMode.PERPLEXITY
) -> BenchmarkingState:
    """Create initial state for new workflow"""
    try:
        from src.config.settings import settings
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.config.settings import settings

    return BenchmarkingState(
        messages=[],
        workflow_status=WorkflowStatus.INITIALIZED,
        current_agent=AgentType.SCRAPER,
        retry_count=0,
        total_retries_remaining=settings.max_retry_attempts,
        scraping_mode=scraping_mode,  # NEW: Mode selection
        oem_urls=oem_urls,
        scraping_results=None,
        all_vehicles=None,
        oem_profiles=None,
        quality_validation=None,
        presentation_result=None,
        total_tokens_used=0,
        total_cost_usd=0.0,
        cost_breakdown={},
        errors=[],
        warnings=[],
        workflow_start_time=datetime.now().isoformat(),
        workflow_end_time=None,
        execution_duration_seconds=None
    )


def get_state_summary(state: BenchmarkingState) -> str:
    """Create readable summary of current state"""
    oem_count = len(state['oem_profiles']) if state.get('oem_profiles') else 0
    return f"""
{'='*60}
WORKFLOW STATE SUMMARY
{'='*60}
Status: {state['workflow_status']}
Agent: {state['current_agent']}
Retries: {state['retry_count']}/{state['retry_count'] + state['total_retries_remaining']}

URLs: {len(state['oem_urls'])}
Vehicles: {len(state['all_vehicles']) if state['all_vehicles'] else 0}
OEM Profiles: {oem_count}
Quality: {state['quality_validation']['overall_quality_score'] if state['quality_validation'] else 'N/A'}
Presentation: {'Yes' if state['presentation_result'] else 'No'}

Cost: ${state['total_cost_usd']:.4f}
Tokens: {state['total_tokens_used']:,}

Errors: {len(state['errors'])}
Warnings: {len(state['warnings'])}
{'='*60}
"""


if __name__ == "__main__":
    """Test state initialization"""
    test_urls = [
        "https://www.tesla.com/semi",
        "https://www.man.eu"
    ]
    
    state = initialize_state(test_urls)
    print(get_state_summary(state))
    print("\nState initialized successfully!")