"""
LangGraph Runtime for E-Powertrain Benchmarking System

This module wires together all agents into a LangGraph StateGraph
with conditional routing based on workflow status.

Workflow:
    START -> scrape -> validate -> [pass] -> present -> END
                          |
                          v [fail + retries]
                        scrape (retry)
                          |
                          v [fail + no retries]
                         END (failed)

Usage:
    from src.graph.runtime import create_workflow, run_benchmark

    # Quick run
    result = run_benchmark(["https://www.man.eu/...", "https://www.volvo..."])

    # Or with more control
    workflow = create_workflow()
    result = workflow.invoke(initial_state)
"""

from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.state import (
    BenchmarkingState,
    WorkflowStatus,
    AgentType,
    initialize_state,
    get_state_summary,
)
from src.agents.scraping_agent import scraping_node
from src.agents.quality_validator import validation_node
from src.agents.presentation_generator import presentation_node


# =====================================================================
# ROUTING FUNCTIONS
# =====================================================================

def route_after_scraping(state: BenchmarkingState) -> Literal["validate", "end_failed"]:
    """
    Route after scraping based on results.

    Returns:
        "validate" if vehicles were found
        "end_failed" if scraping completely failed
    """
    status = state.get("workflow_status")

    # Check for complete failure
    if status == WorkflowStatus.SCRAPING_FAILED:
        return "end_failed"

    # Check if we have any vehicles
    all_vehicles = state.get("all_vehicles", [])
    if not all_vehicles:
        return "end_failed"

    return "validate"


def route_after_validation(state: BenchmarkingState) -> Literal["present", "retry", "end_failed"]:
    """
    Route after validation based on quality score and retry count.

    Returns:
        "present" if validation passed
        "retry" if validation failed but retries available
        "end_failed" if validation failed and no retries left
    """
    validation = state.get("quality_validation")

    if not validation:
        return "end_failed"

    # Check if validation passed
    if validation.get("passes_threshold", False):
        return "present"

    # Check if retries available
    retries_remaining = state.get("total_retries_remaining", 0)
    if retries_remaining > 0:
        return "retry"

    return "end_failed"


# =====================================================================
# FAILURE HANDLER NODE
# =====================================================================

def handle_failure_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Handle workflow failure - set final status and add error summary.
    """
    from datetime import datetime

    errors = state.get("errors", [])

    # Determine failure reason
    status = state.get("workflow_status")
    if status == WorkflowStatus.SCRAPING_FAILED:
        failure_reason = "Scraping failed to extract vehicle data"
    elif status == WorkflowStatus.QUALITY_FAILED:
        failure_reason = "Quality validation failed after all retries"
    else:
        failure_reason = "Workflow failed"

    if failure_reason not in errors:
        errors.append(failure_reason)

    return {
        "workflow_status": WorkflowStatus.FAILED,
        "errors": errors,
        "workflow_end_time": datetime.now().isoformat(),
    }


# =====================================================================
# GRAPH BUILDER
# =====================================================================

def create_workflow(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """
    Create and compile the LangGraph workflow.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Create graph with state schema
    graph = StateGraph(BenchmarkingState)

    # Add nodes
    graph.add_node("scrape", scraping_node)
    graph.add_node("validate", validation_node)
    graph.add_node("present", presentation_node)
    graph.add_node("handle_failure", handle_failure_node)

    # Set entry point
    graph.set_entry_point("scrape")

    # Add conditional edges after scraping
    graph.add_conditional_edges(
        "scrape",
        route_after_scraping,
        {
            "validate": "validate",
            "end_failed": "handle_failure",
        }
    )

    # Add conditional edges after validation
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "present": "present",
            "retry": "scrape",  # Loop back for retry
            "end_failed": "handle_failure",
        }
    )

    # Terminal edges
    graph.add_edge("present", END)
    graph.add_edge("handle_failure", END)

    # Compile
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def run_benchmark(
    urls: List[str],
    thread_id: str = "default",
    verbose: bool = True
) -> BenchmarkingState:
    """
    Run the complete benchmarking workflow.

    Args:
        urls: List of OEM website URLs to scrape
        thread_id: Thread ID for checkpointing
        verbose: Print progress updates

    Returns:
        Final BenchmarkingState after workflow completion
    """
    if verbose:
        print("=" * 60)
        print("E-POWERTRAIN BENCHMARKING SYSTEM")
        print("=" * 60)
        print(f"URLs to process: {len(urls)}")
        for url in urls:
            print(f"  - {url}")
        print()

    # Initialize state
    initial_state = initialize_state(urls)

    # Create workflow
    workflow = create_workflow()

    # Run
    if verbose:
        print("Starting workflow...")
        print("-" * 60)

    config = {"configurable": {"thread_id": thread_id}}
    final_state = workflow.invoke(initial_state, config)

    if verbose:
        print("\n" + get_state_summary(final_state))

    return final_state


def stream_benchmark(
    urls: List[str],
    thread_id: str = "default"
):
    """
    Stream workflow execution, yielding state after each step.

    Useful for progress monitoring and debugging.

    Args:
        urls: List of OEM website URLs
        thread_id: Thread ID for checkpointing

    Yields:
        Dict with node name and updated state after each step
    """
    initial_state = initialize_state(urls)
    workflow = create_workflow()
    config = {"configurable": {"thread_id": thread_id}}

    for step_output in workflow.stream(initial_state, config):
        node_name = list(step_output.keys())[0]
        state = step_output[node_name]

        print(f"[{node_name}] Status: {state.get('workflow_status')}")

        if node_name == "scrape":
            vehicles = state.get("all_vehicles", [])
            print(f"  Vehicles found: {len(vehicles)}")
        elif node_name == "validate":
            validation = state.get("quality_validation", {})
            print(f"  Quality score: {validation.get('overall_quality_score', 'N/A')}")
            print(f"  Passes: {validation.get('passes_threshold', False)}")
        elif node_name == "present":
            result = state.get("presentation_result", {})
            print(f"  Presentations: {len(result.get('all_presentation_paths', []))}")

        yield step_output


def get_graph_visualization() -> str:
    """
    Get Mermaid diagram of the workflow graph.

    Returns:
        Mermaid diagram string
    """
    workflow = create_workflow()
    return workflow.get_graph().draw_mermaid()


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    """Main entry point for CLI execution."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="E-Powertrain Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.graph.runtime --urls https://www.man.eu/trucks https://www.volvo.com
  python -m src.graph.runtime --file src/inputs/urls.txt
  python -m src.graph.runtime --file src/inputs/urls.txt --stream
        """
    )

    parser.add_argument(
        "--urls",
        nargs="+",
        help="URLs to scrape (space-separated)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file containing URLs (one per line)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream progress output"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print graph visualization and exit"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Print graph visualization
    if args.graph:
        print(get_graph_visualization())
        return

    # Get URLs
    urls = []

    if args.urls:
        urls = args.urls
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        with open(file_path, "r") as f:
            urls = [line.strip() for line in f
                   if line.strip() and line.strip().startswith("http")]
    else:
        # Default: use urls.txt
        default_file = Path("src/inputs/urls.txt")
        if default_file.exists():
            with open(default_file, "r") as f:
                urls = [line.strip() for line in f
                       if line.strip() and line.strip().startswith("http")]
        else:
            print("ERROR: No URLs provided. Use --urls or --file")
            parser.print_help()
            sys.exit(1)

    if not urls:
        print("ERROR: No valid URLs found")
        sys.exit(1)

    # Run workflow
    try:
        if args.stream:
            print("Streaming workflow execution...")
            print("=" * 60)
            for step in stream_benchmark(urls):
                pass  # Progress printed in stream_benchmark
            print("=" * 60)
            print("Workflow complete!")
        else:
            result = run_benchmark(urls, verbose=not args.quiet)

            # Print final results
            if result.get("workflow_status") == WorkflowStatus.COMPLETED:
                print("\n Workflow completed successfully!")
                pres_result = result.get("presentation_result", {})
                paths = pres_result.get("all_presentation_paths", [])
                if paths:
                    print(f"\nGenerated {len(paths)} presentation(s):")
                    for path in paths:
                        print(f"  - {path}")
            else:
                print("\n Workflow failed!")
                errors = result.get("errors", [])
                if errors:
                    print("Errors:")
                    for err in errors:
                        print(f"  - {err}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
