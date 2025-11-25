"""
Scraping Agent for E-Powertrain Benchmarking System

LangGraph node that orchestrates web scraping of OEM specifications.
Wraps the EPowertrainExtractor tool and updates workflow state.

Usage as LangGraph node:
    from src.agents.scraping_agent import scraping_node
    graph.add_node("scrape", scraping_node)
"""

import time
from datetime import datetime
from typing import Dict, List, Any

from src.state.state import (
    BenchmarkingState,
    WorkflowStatus,
    AgentType,
)
from src.tools.scraper import EPowertrainExtractor, scrape_oem_urls


# =====================================================================
# LANGGRAPH NODE FUNCTION
# =====================================================================

def scraping_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    LangGraph node for scraping OEM websites.

    Takes URLs from state, scrapes each one, and returns updated state.

    Args:
        state: Current BenchmarkingState with oem_urls

    Returns:
        Dict with state updates (LangGraph merges this with current state)
    """
    start_time = time.time()

    # Update status
    updates = {
        "workflow_status": WorkflowStatus.SCRAPING,
        "current_agent": AgentType.SCRAPER,
    }

    urls = state.get("oem_urls", [])

    if not urls:
        return {
            **updates,
            "workflow_status": WorkflowStatus.SCRAPING_FAILED,
            "errors": state.get("errors", []) + ["No URLs provided for scraping"],
            "scraping_results": [],
            "all_vehicles": [],
        }

    try:
        # Use the scraper tool
        scraping_results = scrape_oem_urls(urls)

        # Aggregate all vehicles
        all_vehicles = []
        total_tokens = 0
        errors = []
        warnings = []

        for result in scraping_results:
            all_vehicles.extend(result.get("vehicles", []))
            total_tokens += result.get("tokens_used", 0)
            errors.extend(result.get("errors", []))
            warnings.extend(result.get("warnings", []))

        # Calculate cost (Perplexity sonar-pro pricing)
        # $0.003/1K input, $0.015/1K output + $0.005/request
        # Approximate: assume 50/50 split input/output
        input_tokens = total_tokens // 2
        output_tokens = total_tokens // 2
        cost = (input_tokens / 1000 * 0.003) + (output_tokens / 1000 * 0.015)
        cost += len(urls) * 0.005  # Per-request fee

        duration = time.time() - start_time

        # Determine success/failure
        if all_vehicles:
            workflow_status = WorkflowStatus.VALIDATING
        else:
            workflow_status = WorkflowStatus.SCRAPING_FAILED
            errors.append("No vehicles extracted from any URL")

        return {
            **updates,
            "workflow_status": workflow_status,
            "scraping_results": scraping_results,
            "all_vehicles": all_vehicles,
            "total_tokens_used": state.get("total_tokens_used", 0) + total_tokens,
            "total_cost_usd": state.get("total_cost_usd", 0.0) + cost,
            "cost_breakdown": {
                **state.get("cost_breakdown", {}),
                "scraping": cost,
            },
            "errors": state.get("errors", []) + errors,
            "warnings": state.get("warnings", []) + warnings,
            "execution_duration_seconds": duration,
        }

    except Exception as e:
        return {
            **updates,
            "workflow_status": WorkflowStatus.SCRAPING_FAILED,
            "errors": state.get("errors", []) + [f"Scraping failed: {str(e)}"],
            "scraping_results": [],
            "all_vehicles": [],
        }


# =====================================================================
# ASYNC VERSION (for async LangGraph workflows)
# =====================================================================

async def async_scraping_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Async version of scraping node.

    Currently wraps sync implementation, but can be extended
    for true async scraping with aiohttp.
    """
    # For now, delegate to sync version
    # TODO: Implement true async with aiohttp for parallel URL fetching
    return scraping_node(state)


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================

def run_scraping(urls: List[str]) -> Dict[str, Any]:
    """
    Run scraping standalone (outside LangGraph).

    Useful for testing or one-off extractions.

    Args:
        urls: List of OEM website URLs

    Returns:
        Dict with scraping_results and all_vehicles
    """
    from src.state.state import initialize_state

    state = initialize_state(urls)
    result = scraping_node(state)

    return {
        "scraping_results": result.get("scraping_results", []),
        "all_vehicles": result.get("all_vehicles", []),
        "total_vehicles": len(result.get("all_vehicles", [])),
        "total_cost_usd": result.get("total_cost_usd", 0),
        "errors": result.get("errors", []),
    }


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    print("Scraping Agent Test")
    print("=" * 60)

    # Test URLs
    test_urls = [
        "https://www.man.eu/global/en/truck/electric-trucks/overview.html",
    ]

    print(f"Testing with {len(test_urls)} URL(s)...")

    result = run_scraping(test_urls)

    print(f"\nResults:")
    print(f"  Vehicles found: {result['total_vehicles']}")
    print(f"  Cost: ${result['total_cost_usd']:.4f}")

    if result['errors']:
        print(f"  Errors: {result['errors']}")

    for sr in result['scraping_results']:
        print(f"\n  {sr['oem_name']}:")
        print(f"    Vehicles: {sr['total_vehicles_found']}")
        print(f"    Source score: {sr['source_compliance_score']}")
