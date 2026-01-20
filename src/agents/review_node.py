"""
Review Node for E-Powertrain Benchmarking System

Human-in-the-loop review step using LangGraph's interrupt pattern.
Pauses workflow for CSV export, human editing, and approval.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from langgraph.types import interrupt, Command

from src.state.state import (
    BenchmarkingState,
    WorkflowStatus,
    AgentType,
    ReviewStatus,
    ReviewDecision,
    VehicleSpecifications,
)
from src.services.csv_export_service import CSVExportService, get_csv_export_service
from src.services.csv_import_service import CSVImportService, get_csv_import_service
from src.services.audit_service import AuditService, get_audit_service

logger = logging.getLogger(__name__)


def review_node(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Review node that prepares data for human review.

    For the MVP, this node:
    1. Exports vehicles to CSV
    2. Creates audit session
    3. Returns AWAITING_REVIEW status (pause handled at UI level)

    The actual human review and resume is handled by the UI calling
    process_review_decision() directly.

    Args:
        state: Current workflow state

    Returns:
        Updated state dictionary with AWAITING_REVIEW status
    """
    logger.info("REVIEW NODE: Preparing for human review")

    # Check if this is a resume (review already approved)
    review_status = state.get("review_status")
    if review_status in (ReviewStatus.APPROVED.value, ReviewStatus.APPROVED_WITH_EDITS.value):
        logger.info("REVIEW NODE: Review already approved, proceeding to presentation")
        return {
            "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
        }

    # Get services
    csv_export = get_csv_export_service()
    audit = get_audit_service()

    # Get vehicles for review
    all_vehicles = state.get("all_vehicles", [])
    if not all_vehicles:
        logger.warning("No vehicles to review")
        return {
            "workflow_status": WorkflowStatus.REVIEW_REJECTED,
            "errors": state.get("errors", []) + ["No vehicles available for review"],
        }

    # Get thread ID for file naming
    thread_id = state.get("thread_id", f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Get quality score
    quality_validation = state.get("quality_validation", {})
    quality_score = quality_validation.get("overall_quality_score", 0.0)

    # Step 1: Export vehicles to CSV
    csv_path = None
    try:
        csv_path, export_metadata = csv_export.export_vehicles(
            vehicles=all_vehicles,
            thread_id=thread_id,
            full_export=False  # Use default columns for cleaner view
        )
        logger.info(f"Exported {len(all_vehicles)} vehicles to {csv_path}")
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        # Continue without CSV - UI will handle export

    # Step 2: Create audit session
    session_id = None
    try:
        session_id = audit.create_session(
            thread_id=thread_id,
            csv_export_path=csv_path or "N/A",
            vehicle_count=len(all_vehicles),
            quality_score=quality_score
        )
        logger.info(f"Created audit session {session_id}")
    except Exception as e:
        logger.error(f"Audit session creation failed: {e}")

    # Step 3: Create pre-review snapshot for rollback/comparison
    pre_review_vehicles = [dict(v) for v in all_vehicles]

    # Return state update - workflow will pause here (no interrupt needed for MVP)
    # The UI handles the pause by checking for AWAITING_REVIEW status
    logger.info("REVIEW NODE: Returning AWAITING_REVIEW status")

    return {
        "workflow_status": WorkflowStatus.AWAITING_REVIEW,
        "current_agent": AgentType.REVIEWER,
        "review_status": ReviewStatus.PENDING.value,
        "review_csv_path": csv_path,
        "pre_review_vehicles": pre_review_vehicles,
        "review_session_id": session_id,
        "thread_id": thread_id,
    }


def _process_approval(
    state: BenchmarkingState,
    human_input: Dict[str, Any],
    session_id: Optional[int],
    pre_review_vehicles: List[Dict[str, Any]],
    csv_path: str,
    thread_id: str
) -> Dict[str, Any]:
    """Process an approval decision, optionally with edits."""
    csv_import = get_csv_import_service()
    audit = get_audit_service()

    reviewer_id = human_input.get("reviewer_id", "unknown")
    edited_csv_path = human_input.get("edited_csv_path")

    all_vehicles = state.get("all_vehicles", [])
    changes_made = []
    review_status = ReviewStatus.APPROVED

    # Check if edited CSV was provided
    if edited_csv_path:
        logger.info(f"Processing edited CSV: {edited_csv_path}")

        try:
            # Import and validate edited CSV
            imported_vehicles, changes, validation_errors = csv_import.import_csv(
                filepath=edited_csv_path,
                original_vehicles=pre_review_vehicles
            )

            # Log validation errors as warnings
            if validation_errors:
                warnings = state.get("warnings", [])
                for err in validation_errors:
                    warnings.append(f"Row {err.row_id}, {err.field}: {err.error}")
                logger.warning(f"Import had {len(validation_errors)} validation warnings")

            if changes:
                # Convert FieldChange objects to dictionaries for audit
                changes_made = [
                    {
                        "row_id": c.row_id,
                        "vehicle_name": c.vehicle_name,
                        "field": c.field,
                        "field_display_name": c.field_display_name,
                        "original_value": c.original_value,
                        "new_value": c.new_value,
                        "change_type": c.change_type,
                    }
                    for c in changes
                ]
                review_status = ReviewStatus.APPROVED_WITH_EDITS
                all_vehicles = imported_vehicles
                logger.info(f"Applied {len(changes)} edits from CSV")

                # Log edits to audit trail
                if session_id:
                    audit.log_edits(session_id, changes_made)

        except Exception as e:
            logger.error(f"CSV import failed: {e}")
            return {
                "workflow_status": WorkflowStatus.REVIEW_REJECTED,
                "errors": state.get("errors", []) + [f"CSV import failed: {str(e)}"],
            }

    # Complete audit session
    if session_id:
        audit.complete_session(
            session_id=session_id,
            status=review_status.value,
            reviewer_id=reviewer_id,
            csv_import_path=edited_csv_path
        )

    # Create review decision record
    review_decision = ReviewDecision(
        status=review_status.value,
        reviewer_id=reviewer_id,
        reviewed_at=datetime.now().isoformat(),
        original_vehicle_count=len(pre_review_vehicles),
        edited_vehicle_count=len(all_vehicles),
        changes_made=changes_made,
        rejection_reason=None,
        csv_export_path=csv_path,
        csv_import_path=edited_csv_path
    )

    logger.info(f"Review approved by {reviewer_id} with status: {review_status.value}")

    return {
        "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
        "review_status": review_status.value,
        "review_decision": review_decision,
        "all_vehicles": all_vehicles,
    }


def _process_rejection(
    state: BenchmarkingState,
    session_id: Optional[int],
    reviewer_id: str,
    rejection_reason: Optional[str]
) -> Dict[str, Any]:
    """Process a rejection decision."""
    audit = get_audit_service()

    # Complete audit session
    if session_id:
        audit.complete_session(
            session_id=session_id,
            status=ReviewStatus.REJECTED.value,
            reviewer_id=reviewer_id,
            rejection_reason=rejection_reason
        )

    # Create review decision record
    review_decision = ReviewDecision(
        status=ReviewStatus.REJECTED.value,
        reviewer_id=reviewer_id,
        reviewed_at=datetime.now().isoformat(),
        original_vehicle_count=len(state.get("all_vehicles", [])),
        edited_vehicle_count=0,
        changes_made=[],
        rejection_reason=rejection_reason,
        csv_export_path=state.get("review_csv_path", ""),
        csv_import_path=None
    )

    reason_msg = f" Reason: {rejection_reason}" if rejection_reason else ""
    logger.info(f"Review rejected by {reviewer_id}.{reason_msg}")

    return {
        "workflow_status": WorkflowStatus.REVIEW_REJECTED,
        "review_status": ReviewStatus.REJECTED.value,
        "review_decision": review_decision,
        "errors": state.get("errors", []) + [f"Review rejected by {reviewer_id}.{reason_msg}"],
    }


def get_review_status_summary(state: BenchmarkingState) -> Dict[str, Any]:
    """
    Get a summary of the current review status for UI display.

    Args:
        state: Current workflow state

    Returns:
        Summary dictionary for UI
    """
    status = state.get("workflow_status")
    review_status = state.get("review_status")
    review_decision = state.get("review_decision")

    is_paused = status == WorkflowStatus.AWAITING_REVIEW
    is_reviewing = status == WorkflowStatus.REVIEWING
    is_rejected = status == WorkflowStatus.REVIEW_REJECTED or review_status == ReviewStatus.REJECTED.value

    summary = {
        "is_paused": is_paused,
        "is_reviewing": is_reviewing,
        "is_rejected": is_rejected,
        "workflow_status": status.value if status else None,
        "review_status": review_status,
        "csv_path": state.get("review_csv_path"),
        "thread_id": state.get("thread_id"),
        "session_id": state.get("review_session_id"),
        "vehicle_count": len(state.get("all_vehicles", [])),
        "quality_score": None,
    }

    # Add quality score if available
    validation = state.get("quality_validation")
    if validation:
        summary["quality_score"] = validation.get("overall_quality_score")

    # Add decision details if available
    if review_decision:
        summary["reviewer_id"] = review_decision.get("reviewer_id")
        summary["changes_count"] = len(review_decision.get("changes_made", []))
        summary["rejection_reason"] = review_decision.get("rejection_reason")

    return summary


# For manual testing
if __name__ == "__main__":
    print("Review node module loaded successfully.")
    print("This module requires a running LangGraph workflow to test.")
    print("Use the runtime to test the full review workflow.")
