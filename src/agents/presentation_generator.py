"""
Presentation Generator Agent for E-Powertrain Benchmarking System

LangGraph node that generates PowerPoint presentations from validated data.
Wraps the ppt_generator tool and updates workflow state.

Usage as LangGraph node:
    from src.agents.presentation_generator import presentation_node
    graph.add_node("present", presentation_node)
"""

import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from src.state.state import (
    BenchmarkingState,
    WorkflowStatus,
    AgentType,
)
from src.tools.ppt_generator import (
    generate_all_presentations,
    generate_comparison_presentation,
    transform_scraping_result_to_oem_profile,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_TEMPLATE_PATH = "templates/IAA_Template.pptx"
DEFAULT_OUTPUT_DIR = "outputs"


# =====================================================================
# LANGGRAPH NODE FUNCTION
# =====================================================================

def presentation_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    LangGraph node for generating presentations.

    Takes validated scraping results and generates PowerPoint presentations.

    Args:
        state: Current BenchmarkingState with scraping_results and quality_validation

    Returns:
        Dict with state updates (LangGraph merges this with current state)
    """
    start_time = time.time()

    # Update status
    updates = {
        "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
        "current_agent": AgentType.PRESENTER,
    }

    # Check prerequisites
    scraping_results = state.get("scraping_results", [])
    quality_validation = state.get("quality_validation")

    if not scraping_results:
        return {
            **updates,
            "workflow_status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + ["No scraping results to generate presentation from"],
        }

    # Check if validation passed (optional - can be bypassed)
    if quality_validation and not quality_validation.get("passes_threshold", False):
        # Log warning but continue (quality check is informational)
        pass

    try:
        # Get paths from settings if available
        try:
            from src.config.settings import settings
            template_path = str(settings.ppt_template_path)
            output_dir = str(settings.output_directory)
        except Exception:
            template_path = DEFAULT_TEMPLATE_PATH
            output_dir = DEFAULT_OUTPUT_DIR

        # Check if template exists
        if not Path(template_path).exists():
            # Try alternative locations
            alt_paths = [
                "IAA_Template.pptx",
                "templates/IAA_Template.pptx",
                "../templates/IAA_Template.pptx",
            ]
            template_found = False
            for alt in alt_paths:
                if Path(alt).exists():
                    template_path = alt
                    template_found = True
                    break

            if not template_found:
                return {
                    **updates,
                    "workflow_status": WorkflowStatus.FAILED,
                    "errors": state.get("errors", []) + [
                        f"Template not found: {template_path}. "
                        f"Please ensure IAA_Template.pptx exists."
                    ],
                }

        # Generate presentations for all OEMs
        result = generate_all_presentations(
            scraping_results=scraping_results,
            template_path=template_path,
            output_dir=output_dir,
        )

        oem_profiles = result.get("oem_profiles", [])
        presentation_result = result.get("presentation_result", {})
        generation_errors = presentation_result.get("errors", [])

        duration = time.time() - start_time

        # Determine final status
        if oem_profiles and presentation_result.get("presentation_path"):
            workflow_status = WorkflowStatus.COMPLETED
        else:
            workflow_status = WorkflowStatus.FAILED

        return {
            **updates,
            "workflow_status": workflow_status,
            "oem_profiles": oem_profiles,
            "presentation_result": presentation_result,
            "errors": state.get("errors", []) + generation_errors,
            "workflow_end_time": datetime.now().isoformat(),
            "execution_duration_seconds": state.get("execution_duration_seconds", 0) + duration,
        }

    except Exception as e:
        return {
            **updates,
            "workflow_status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + [f"Presentation generation failed: {str(e)}"],
        }


# =====================================================================
# ALTERNATIVE: COMPARISON MODE
# =====================================================================

def comparison_presentation_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Alternative node that generates a SINGLE comparison presentation.

    Instead of one presentation per OEM, creates one presentation
    with all OEMs on separate slides.
    """
    start_time = time.time()

    updates = {
        "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
        "current_agent": AgentType.PRESENTER,
    }

    scraping_results = state.get("scraping_results", [])

    if not scraping_results:
        return {
            **updates,
            "workflow_status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + ["No scraping results"],
        }

    try:
        from src.config.settings import settings
        template_path = str(settings.ppt_template_path)
        output_dir = str(settings.output_directory)
    except Exception:
        template_path = DEFAULT_TEMPLATE_PATH
        output_dir = DEFAULT_OUTPUT_DIR

    try:
        result = generate_comparison_presentation(
            scraping_results=scraping_results,
            template_path=template_path,
            output_dir=output_dir,
        )

        return {
            **updates,
            "workflow_status": WorkflowStatus.COMPLETED,
            "oem_profiles": result.get("oem_profiles", []),
            "presentation_result": result.get("presentation_result", {}),
            "workflow_end_time": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            **updates,
            "workflow_status": WorkflowStatus.FAILED,
            "errors": state.get("errors", []) + [f"Comparison presentation failed: {str(e)}"],
        }


# =====================================================================
# ASYNC VERSION
# =====================================================================

async def async_presentation_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Async version of presentation node.

    Currently wraps sync implementation.
    """
    return presentation_node(state)


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

def run_presentation_generation(
    scraping_results: List[Dict],
    template_path: str = DEFAULT_TEMPLATE_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    mode: str = "individual"  # "individual" or "comparison"
) -> Dict[str, Any]:
    """
    Run presentation generation standalone (outside LangGraph).

    Args:
        scraping_results: List of ScrapingResult dicts
        template_path: Path to PowerPoint template
        output_dir: Output directory for generated files
        mode: "individual" for per-OEM files, "comparison" for single file

    Returns:
        Dict with oem_profiles and presentation_result
    """
    if mode == "comparison":
        return generate_comparison_presentation(
            scraping_results=scraping_results,
            template_path=template_path,
            output_dir=output_dir,
        )
    else:
        return generate_all_presentations(
            scraping_results=scraping_results,
            template_path=template_path,
            output_dir=output_dir,
        )


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import json

    print("Presentation Generator Agent Test")
    print("=" * 60)

    # Test with mock scraping result
    mock_scraping_results = [
        {
            "oem_name": "MAN Truck & Bus",
            "oem_url": "https://www.man.eu",
            "vehicles": [
                {
                    "vehicle_name": "MAN eTGX",
                    "oem_name": "MAN",
                    "category": "Heavy-duty Truck",
                    "powertrain_type": "BEV",
                    "source_url": "https://www.man.eu/etgx",
                    "battery_capacity_kwh": 480,
                    "motor_power_kw": 480,
                    "range_km": 600,
                    "dc_charging_kw": 375,
                    "gvw_kg": 40000,
                    "additional_specs": {
                        "wheel_formula": "6x2",
                        "wheelbase_mm": 4500,
                    },
                    "extraction_timestamp": datetime.now().isoformat(),
                }
            ],
            "total_vehicles_found": 1,
            "extraction_timestamp": datetime.now().isoformat(),
            "official_citations": ["https://www.man.eu/specs"],
            "third_party_citations": [],
            "source_compliance_score": 0.85,
            "raw_content": "...",
            "tokens_used": 1500,
            "model_used": "sonar-pro",
            "extraction_duration_seconds": 12.5,
            "errors": [],
            "warnings": [],
        }
    ]

    print("Testing with mock scraping result...")
    print(f"OEMs: {len(mock_scraping_results)}")

    # Test transformation
    from src.tools.ppt_generator import transform_scraping_result_to_oem_profile

    oem_profile = transform_scraping_result_to_oem_profile(mock_scraping_results[0])
    print(f"\nTransformed OEM Profile:")
    print(f"  Company: {oem_profile['company_name']}")
    print(f"  Products: {len(oem_profile['products'])}")
    print(f"  Highlights: {oem_profile['expected_highlights']}")

    # Full generation test (requires template)
    print("\n" + "-" * 40)
    print("To test full generation, ensure IAA_Template.pptx exists")
    print("Run: python -m src.agents.presentation_generator")
