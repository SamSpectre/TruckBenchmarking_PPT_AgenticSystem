"""
LangGraph Runtime for E-Powertrain Benchmarking System

This module wires together all agents into a LangGraph StateGraph
with conditional routing based on workflow status.

Workflow (with review enabled):
    START -> scrape -> validate -> [pass] -> review -> [approve] -> present -> END
                          |                     |
                          v [fail + retries]    v [reject]
                        scrape (retry)       END (rejected)
                          |
                          v [fail + no retries]
                         END (failed)

Workflow (without review):
    START -> scrape -> validate -> [pass] -> present -> END

Usage:
    from src.graph.runtime import create_workflow, run_benchmark

    # Quick run (with review step)
    result = run_benchmark(["https://www.man.eu/...", "https://www.volvo..."])

    # Without review step
    result = run_benchmark(urls, enable_review=False)

    # Resume after human review
    from src.graph.runtime import resume_after_review
    result = resume_after_review(thread_id, "approve", reviewer_id="user123")
"""

from typing import Dict, Any, List, Optional, Literal, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pathlib import Path

from src.state.state import (
    BenchmarkingState,
    WorkflowStatus,
    AgentType,
    ScrapingMode,
    ReviewStatus,
    initialize_state,
    get_state_summary,
)
from src.agents.scraping_agent import scraping_node
from src.agents.quality_validator import validation_node
from src.agents.presentation_generator import presentation_node
from src.agents.review_node import review_node


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


def route_after_validation(state: BenchmarkingState) -> Literal["review", "retry", "end_failed"]:
    """
    Route after validation based on quality score and retry count.

    Returns:
        "review" if validation passed (proceed to human review)
        "retry" if validation failed but retries available
        "end_failed" if validation failed and no retries left
    """
    validation = state.get("quality_validation")

    if not validation:
        return "end_failed"

    # Check if validation passed
    if validation.get("passes_threshold", False):
        return "review"  # Go to review instead of directly to present

    # Check if retries available
    retries_remaining = state.get("total_retries_remaining", 0)
    if retries_remaining > 0:
        return "retry"

    return "end_failed"


def route_after_validation_no_review(state: BenchmarkingState) -> Literal["present", "retry", "end_failed"]:
    """
    Route after validation (no review mode) - goes directly to presentation.

    Returns:
        "present" if validation passed
        "retry" if validation failed but retries available
        "end_failed" if validation failed and no retries left
    """
    validation = state.get("quality_validation")

    if not validation:
        return "end_failed"

    if validation.get("passes_threshold", False):
        return "present"

    retries_remaining = state.get("total_retries_remaining", 0)
    if retries_remaining > 0:
        return "retry"

    return "end_failed"


def route_after_review(state: BenchmarkingState) -> Literal["present", "end_rejected", "end_paused"]:
    """
    Route after review based on status.

    Returns:
        "present" if review approved (with or without edits)
        "end_rejected" if review rejected
        "end_paused" if awaiting human review (MVP: ends here, UI handles resume)
    """
    review_status = state.get("review_status")
    workflow_status = state.get("workflow_status")

    # Check if rejected
    if review_status == ReviewStatus.REJECTED.value:
        return "end_rejected"
    if workflow_status == WorkflowStatus.REVIEW_REJECTED:
        return "end_rejected"

    # Check if awaiting review (pause for human input)
    if workflow_status == WorkflowStatus.AWAITING_REVIEW:
        return "end_paused"

    # Approved or approved with edits -> proceed to presentation
    if review_status in (ReviewStatus.APPROVED.value, ReviewStatus.APPROVED_WITH_EDITS.value):
        return "present"

    # Default: pause for review
    return "end_paused"


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
    elif status == WorkflowStatus.REVIEW_REJECTED:
        failure_reason = "Data rejected during human review"
    else:
        failure_reason = "Workflow failed"

    if failure_reason not in errors:
        errors.append(failure_reason)

    return {
        "workflow_status": WorkflowStatus.FAILED,
        "errors": errors,
        "workflow_end_time": datetime.now().isoformat(),
    }


def handle_rejection_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Handle review rejection - mark workflow as rejected.
    """
    from datetime import datetime

    errors = state.get("errors", [])

    # Get rejection reason from review decision
    review_decision = state.get("review_decision", {})
    rejection_reason = review_decision.get("rejection_reason", "Data rejected during human review")

    if rejection_reason and rejection_reason not in errors:
        errors.append(rejection_reason)

    return {
        "workflow_status": WorkflowStatus.REVIEW_REJECTED,
        "errors": errors,
        "workflow_end_time": datetime.now().isoformat(),
    }


def handle_paused_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Handle workflow pause for human review.
    This is a terminal node for the MVP - the UI handles the actual review.
    """
    # Just pass through - state already has AWAITING_REVIEW status
    return {}


# =====================================================================
# CHECKPOINTER SETUP
# =====================================================================

_default_checkpointer = None


def get_sqlite_checkpointer(db_path: str = "data/checkpoints.db"):
    """
    Get SQLite checkpointer for workflow state persistence.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SqliteSaver checkpointer
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(db_path)
    except ImportError:
        # Fall back to memory saver if sqlite not available
        print("Warning: langgraph-checkpoint-sqlite not installed, using memory saver")
        return MemorySaver()


def get_default_checkpointer():
    """Get or create the default checkpointer."""
    global _default_checkpointer
    if _default_checkpointer is None:
        _default_checkpointer = get_sqlite_checkpointer()
    return _default_checkpointer


# =====================================================================
# GRAPH BUILDER
# =====================================================================

def create_workflow(
    checkpointer: Optional[Union[MemorySaver, Any]] = None,
    enable_review: bool = True
) -> StateGraph:
    """
    Create and compile the LangGraph workflow.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                      If None, uses SQLite checkpointer by default.
        enable_review: If True, include human review step (default).
                       If False, skip review and go directly to presentation.

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

    if enable_review:
        graph.add_node("review", review_node)
        graph.add_node("handle_rejection", handle_rejection_node)
        graph.add_node("handle_paused", handle_paused_node)

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
    if enable_review:
        # With review: validate -> review -> present
        graph.add_conditional_edges(
            "validate",
            route_after_validation,
            {
                "review": "review",
                "retry": "scrape",
                "end_failed": "handle_failure",
            }
        )

        # Add conditional edges after review
        graph.add_conditional_edges(
            "review",
            route_after_review,
            {
                "present": "present",
                "end_rejected": "handle_rejection",
                "end_paused": "handle_paused",
            }
        )

        # Terminal edges for review outcomes
        graph.add_edge("handle_rejection", END)
        graph.add_edge("handle_paused", END)
    else:
        # Without review: validate -> present (direct)
        graph.add_conditional_edges(
            "validate",
            route_after_validation_no_review,
            {
                "present": "present",
                "retry": "scrape",
                "end_failed": "handle_failure",
            }
        )

    # Terminal edges
    graph.add_edge("present", END)
    graph.add_edge("handle_failure", END)

    # Use provided checkpointer or default to SQLite
    if checkpointer is None:
        checkpointer = get_default_checkpointer()

    return graph.compile(checkpointer=checkpointer)


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def run_benchmark(
    urls: List[str],
    thread_id: str = "default",
    verbose: bool = True,
    mode: ScrapingMode = ScrapingMode.INTELLIGENT,
    enable_review: bool = True
) -> BenchmarkingState:
    """
    Run the complete benchmarking workflow.

    Args:
        urls: List of OEM website URLs to scrape
        thread_id: Thread ID for checkpointing
        verbose: Print progress updates
        mode: Scraping mode (default: intelligent)
        enable_review: If True, pause for human review (default)

    Returns:
        Final BenchmarkingState after workflow completion.
        If enable_review=True and validation passes, the workflow
        will pause at the review step and return with status AWAITING_REVIEW.
    """
    if verbose:
        print("=" * 60)
        print("E-POWERTRAIN BENCHMARKING SYSTEM")
        print("=" * 60)
        print(f"Mode: {mode.value}")
        print(f"Review: {'Enabled' if enable_review else 'Disabled'}")
        print(f"URLs to process: {len(urls)}")
        for url in urls:
            print(f"  - {url}")
        print()

    # Initialize state with mode
    initial_state = initialize_state(urls, scraping_mode=mode)
    # Store thread_id in state for review node
    initial_state["thread_id"] = thread_id

    # Create workflow
    workflow = create_workflow(enable_review=enable_review)

    # Run
    if verbose:
        print("Starting workflow...")
        print("-" * 60)

    config = {"configurable": {"thread_id": thread_id}}
    final_state = workflow.invoke(initial_state, config)

    if verbose:
        status = final_state.get("workflow_status")
        if status == WorkflowStatus.AWAITING_REVIEW:
            print("\n" + "=" * 60)
            print("WORKFLOW PAUSED FOR HUMAN REVIEW")
            print("=" * 60)
            print(f"Thread ID: {thread_id}")
            print(f"CSV Path: {final_state.get('review_csv_path', 'N/A')}")
            print(f"Vehicles: {len(final_state.get('all_vehicles', []))}")
            print("\nTo continue, use resume_after_review() with your decision.")
            print("=" * 60)
        else:
            print("\n" + get_state_summary(final_state))

    return final_state


def resume_after_review(
    thread_id: str,
    decision: str,
    reviewer_id: str = "unknown",
    edited_csv_path: Optional[str] = None,
    rejection_reason: Optional[str] = None,
    verbose: bool = True
) -> BenchmarkingState:
    """
    Resume workflow after human review.

    Args:
        thread_id: Thread ID of the paused workflow
        decision: "approve" or "reject"
        reviewer_id: ID of the reviewer
        edited_csv_path: Path to edited CSV (optional, for approve with edits)
        rejection_reason: Reason for rejection (required if rejecting)
        verbose: Print progress updates

    Returns:
        Final BenchmarkingState after workflow completion
    """
    if decision not in ("approve", "reject"):
        raise ValueError(f"Invalid decision: {decision}. Must be 'approve' or 'reject'")

    if decision == "reject" and not rejection_reason:
        rejection_reason = "Rejected without reason"

    if verbose:
        print("=" * 60)
        print("RESUMING WORKFLOW AFTER REVIEW")
        print("=" * 60)
        print(f"Thread ID: {thread_id}")
        print(f"Decision: {decision}")
        print(f"Reviewer: {reviewer_id}")
        if edited_csv_path:
            print(f"Edited CSV: {edited_csv_path}")
        if rejection_reason:
            print(f"Reason: {rejection_reason}")
        print()

    # Create workflow and resume
    workflow = create_workflow(enable_review=True)
    config = {"configurable": {"thread_id": thread_id}}

    # Resume with human input
    resume_value = {
        "decision": decision,
        "reviewer_id": reviewer_id,
        "edited_csv_path": edited_csv_path,
        "rejection_reason": rejection_reason,
    }

    if verbose:
        print("Resuming workflow...")
        print("-" * 60)

    # Use Command to resume with the human input
    final_state = workflow.invoke(Command(resume=resume_value), config)

    if verbose:
        print("\n" + get_state_summary(final_state))

    return final_state


def get_workflow_status(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current status of a workflow by thread ID.

    Args:
        thread_id: Thread ID to check

    Returns:
        Dictionary with workflow status info, or None if not found
    """
    try:
        workflow = create_workflow(enable_review=True)
        config = {"configurable": {"thread_id": thread_id}}
        state = workflow.get_state(config)

        if state and state.values:
            return {
                "thread_id": thread_id,
                "workflow_status": state.values.get("workflow_status"),
                "review_status": state.values.get("review_status"),
                "csv_path": state.values.get("review_csv_path"),
                "vehicle_count": len(state.values.get("all_vehicles", [])),
                "is_paused": state.values.get("workflow_status") == WorkflowStatus.AWAITING_REVIEW,
            }
    except Exception as e:
        print(f"Error getting workflow status: {e}")

    return None


def stream_benchmark(
    urls: List[str],
    thread_id: str = "default",
    mode: ScrapingMode = ScrapingMode.INTELLIGENT,
    enable_review: bool = True
):
    """
    Stream workflow execution, yielding state after each step.

    Useful for progress monitoring and debugging.

    Args:
        urls: List of OEM website URLs
        thread_id: Thread ID for checkpointing
        mode: Scraping mode (default: intelligent)
        enable_review: If True, include human review step

    Yields:
        Dict with node name and updated state after each step
    """
    initial_state = initialize_state(urls, scraping_mode=mode)
    initial_state["thread_id"] = thread_id
    workflow = create_workflow(enable_review=enable_review)
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
        elif node_name == "review":
            print(f"  CSV Path: {state.get('review_csv_path', 'N/A')}")
            print(f"  Review Status: {state.get('review_status', 'N/A')}")
            if state.get("workflow_status") == WorkflowStatus.AWAITING_REVIEW:
                print("  ** PAUSED: Waiting for human review **")
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["intelligent", "auto"],
        default="intelligent",
        help="Scraping mode: intelligent (multi-page CRAWL4AI+OpenAI), auto"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip human review step (go directly to presentation)"
    )

    args = parser.parse_args()

    # Parse mode
    mode_map = {
        "intelligent": ScrapingMode.INTELLIGENT,
        "auto": ScrapingMode.AUTO,
    }
    scraping_mode = mode_map.get(args.mode, ScrapingMode.INTELLIGENT)

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

    enable_review = not args.no_review

    # Run workflow
    try:
        if args.stream:
            print(f"Streaming workflow execution (mode: {args.mode})...")
            print(f"Review: {'Disabled' if args.no_review else 'Enabled'}")
            print("=" * 60)
            for step in stream_benchmark(urls, mode=scraping_mode, enable_review=enable_review):
                pass  # Progress printed in stream_benchmark
            print("=" * 60)
            print("Workflow complete!")
        else:
            result = run_benchmark(urls, verbose=not args.quiet, mode=scraping_mode, enable_review=enable_review)

            # Print final results
            status = result.get("workflow_status")
            if status == WorkflowStatus.COMPLETED:
                print("\n[OK] Workflow completed successfully!")
                pres_result = result.get("presentation_result", {})
                paths = pres_result.get("all_presentation_paths", [])
                if paths:
                    print(f"\nGenerated {len(paths)} presentation(s):")
                    for path in paths:
                        print(f"  - {path}")
            elif status == WorkflowStatus.AWAITING_REVIEW:
                # Paused for review - not an error
                print("\n[PAUSED] Workflow paused for human review")
                print(f"Thread ID: {result.get('thread_id', 'default')}")
                print(f"CSV Path: {result.get('review_csv_path', 'N/A')}")
                print("\nTo continue, use the UI Review tab or call resume_after_review()")
            elif status == WorkflowStatus.REVIEW_REJECTED:
                print("\n[REJECTED] Data rejected during human review")
                errors = result.get("errors", [])
                if errors:
                    print("Reason:")
                    for err in errors:
                        print(f"  - {err}")
                sys.exit(1)
            else:
                print("\n[FAILED] Workflow failed!")
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
