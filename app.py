"""
E-Powertrain Benchmarking System - Gradio Frontend

Professional web interface for extracting electric vehicle specifications
from OEM websites and generating PowerPoint presentations.

Includes human-in-the-loop review workflow with CSV export/import.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.graph.runtime import run_benchmark, resume_after_review, get_workflow_status
from src.state.state import ScrapingMode, WorkflowStatus
from src.services.csv_export_service import get_csv_export_service
from src.services.csv_import_service import get_csv_import_service
from src.services.audit_service import get_audit_service


# =============================================================================
# GLOBAL STATE FOR REVIEW WORKFLOW
# =============================================================================

# Store current review state (in production, use proper session management)
_review_state: Dict[str, Any] = {
    "thread_id": None,
    "csv_path": None,
    "vehicles": [],
    "quality_score": 0.0,
    "is_paused": False,
    "edited_csv_path": None,
    "session_id": None,
}


# =============================================================================
# SAMPLE OEM URLs
# =============================================================================

SAMPLE_OEMS = {
    "MAN Truck & Bus": "https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html",
    "Mercedes-Benz Trucks": "https://www.mercedes-benz-trucks.com/de_DE/models/eactros-600.html",
    "Volvo Trucks": "https://www.volvotrucks.com/en-en/trucks/volvo-fh-electric.html",
    "Scania": "https://www.scania.com/group/en/home/products-and-services/trucks/battery-electric-vehicles.html",
    "DAF Trucks": "https://www.daf.com/en/trucks/electric-trucks",
}


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_extraction(urls_text: str, mode: str, enable_review: bool, progress=gr.Progress()):
    """Main extraction function with optional human review step."""
    global _review_state

    urls = [u.strip() for u in urls_text.strip().split('\n') if u.strip()]

    if not urls:
        return (
            "**Error:** No URLs provided. Please enter at least one OEM URL.",
            pd.DataFrame(),
            [],
            ""
        )

    try:
        progress(0.1, desc="Starting extraction...")
        scraping_mode = ScrapingMode(mode)

        # Generate thread ID for this run
        thread_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        progress(0.3, desc="Fetching and extracting data...")
        result = run_benchmark(
            urls,
            thread_id=thread_id,
            mode=scraping_mode,
            verbose=True,
            enable_review=enable_review
        )

        progress(0.9, desc="Processing results...")

        # Get data
        all_vehicles = result.get('all_vehicles', [])
        quality = result.get('quality_validation', {}) or {}
        quality_score = quality.get('overall_quality_score', 0)

        # Calculate metrics
        oem_names = list(set(v.get('oem_name', 'Unknown') for v in all_vehicles))
        avg_completeness = sum(v.get('data_completeness_score', 0) for v in all_vehicles) / len(all_vehicles) if all_vehicles else 0

        # Status
        status = result.get('workflow_status', 'unknown')
        if hasattr(status, 'value'):
            status = status.value

        if status == 'completed':
            # Reset review state
            _review_state = {
                "thread_id": None,
                "csv_path": None,
                "vehicles": [],
                "quality_score": 0.0,
                "is_paused": False,
                "edited_csv_path": None,
                "session_id": None,
            }
            status_msg = f"""### Extraction Complete

- **Vehicles Found:** {len(all_vehicles)}
- **OEMs Processed:** {len(oem_names)} ({', '.join(oem_names)})
- **Quality Score:** {quality_score*100:.0f}%
- **Avg Completeness:** {avg_completeness*100:.0f}%
"""
        elif status == 'awaiting_review':
            # Export CSV directly here to ensure it's available
            csv_path = None
            session_id = None
            try:
                csv_export_svc = get_csv_export_service()
                csv_path, _ = csv_export_svc.export_vehicles(
                    vehicles=all_vehicles,
                    thread_id=thread_id,
                    full_export=False
                )

                # Create audit session
                try:
                    audit_svc = get_audit_service()
                    session_id = audit_svc.create_session(
                        thread_id=thread_id,
                        csv_export_path=csv_path,
                        vehicle_count=len(all_vehicles),
                        quality_score=quality_score
                    )
                except Exception as audit_err:
                    print(f"Audit session creation failed: {audit_err}")

            except Exception as csv_err:
                csv_path = result.get('review_csv_path')
                print(f"CSV export in UI failed, using workflow path: {csv_err}")

            # Store review state for the Review tab
            _review_state = {
                "thread_id": result.get('thread_id', thread_id),
                "csv_path": csv_path,
                "vehicles": all_vehicles,
                "quality_score": quality_score,
                "is_paused": True,
                "edited_csv_path": None,
                "session_id": session_id,
            }
            status_msg = f"""### Extraction Paused - Awaiting Review

- **Vehicles Found:** {len(all_vehicles)}
- **OEMs Processed:** {len(oem_names)} ({', '.join(oem_names)})
- **Quality Score:** {quality_score*100:.0f}%
- **Avg Completeness:** {avg_completeness*100:.0f}%
- **CSV Ready:** {csv_path}

**Next Step:** Go to the **Review & Approve** tab to review the data, make edits in Excel if needed, and approve or reject.
"""
        else:
            # Reset review state on failure
            _review_state = {
                "thread_id": None,
                "csv_path": None,
                "vehicles": [],
                "quality_score": 0.0,
                "is_paused": False,
                "edited_csv_path": None,
                "session_id": None,
            }
            errors = result.get('errors', [])
            error_text = ', '.join(errors) if errors else 'Unknown error'
            status_msg = f"### Extraction Failed\n\n**Error:** {error_text}"

        # Create dataframe
        df = format_vehicle_table(all_vehicles)

        # Get files
        files = get_presentation_files(result)

        # Create JSON output
        json_output = json.dumps({
            "extraction_timestamp": datetime.now().isoformat(),
            "total_vehicles": len(all_vehicles),
            "oems_processed": oem_names,
            "quality_score": quality_score,
            "vehicles": all_vehicles
        }, indent=2, default=str)

        progress(1.0, desc="Complete!")
        return status_msg, df, files, json_output

    except Exception as e:
        return f"### Error\n\n{str(e)}", pd.DataFrame(), [], ""


def format_vehicle_table(vehicles: list) -> pd.DataFrame:
    """Convert vehicle list to DataFrame."""
    if not vehicles:
        return pd.DataFrame()

    rows = []
    for v in vehicles:
        # Format battery
        battery = v.get('battery_capacity_kwh')
        battery_min = v.get('battery_capacity_min_kwh')
        if battery and battery_min and battery != battery_min:
            battery_str = f"{battery_min}-{battery}"
        elif battery:
            battery_str = str(battery)
        else:
            battery_str = "-"

        # Format motor
        motor = v.get('motor_power_kw')
        motor_min = v.get('motor_power_min_kw')
        if motor and motor_min and motor != motor_min:
            motor_str = f"{motor_min}-{motor}"
        elif motor:
            motor_str = str(motor)
        else:
            motor_str = "-"

        # Format charging
        mcs = v.get('mcs_charging_kw')
        dc = v.get('dc_charging_kw')
        charging_str = f"{mcs} MCS" if mcs else (f"{dc} DC" if dc else "-")

        # Format weight
        gcw = v.get('gcw_kg')
        gvw = v.get('gvw_kg')
        weight_str = f"{gcw:,} GCW" if gcw else (f"{gvw:,} GVW" if gvw else "-")

        rows.append({
            'OEM': v.get('oem_name', '-'),
            'Vehicle': v.get('vehicle_name', '-'),
            'Battery (kWh)': battery_str,
            'Range (km)': v.get('range_km') or '-',
            'Motor (kW)': motor_str,
            'Charging (kW)': charging_str,
            'Weight (kg)': weight_str,
            'Completeness': f"{v.get('data_completeness_score', 0)*100:.0f}%"
        })

    return pd.DataFrame(rows)


def get_presentation_files(result: dict) -> list:
    """Extract presentation file paths."""
    pres_result = result.get('presentation_result', {}) or {}
    paths = pres_result.get('all_presentation_paths', [])
    return [p for p in paths if Path(p).exists()]


def add_sample_url(current_text: str, oem_name: str) -> str:
    """Add sample URL to input."""
    url = SAMPLE_OEMS.get(oem_name, "")
    if not url:
        return current_text

    current_urls = [u.strip() for u in current_text.strip().split('\n') if u.strip()]
    if url not in current_urls:
        current_urls.append(url)
    return '\n'.join(current_urls)


def export_csv(json_data: str) -> Optional[str]:
    """Export to CSV."""
    if not json_data:
        return None
    try:
        data = json.loads(json_data)
        vehicles = data.get('vehicles', [])
        if not vehicles:
            return None
        df = pd.DataFrame(vehicles)
        output_path = f"outputs/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        Path("outputs").mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path
    except Exception:
        return None


# =============================================================================
# REVIEW WORKFLOW FUNCTIONS
# =============================================================================

def get_review_status() -> Tuple[str, pd.DataFrame, Optional[str]]:
    """Get current review status for display."""
    global _review_state

    if not _review_state.get("is_paused"):
        return (
            "### No Active Review\n\nRun an extraction with review enabled to start the review workflow.",
            pd.DataFrame(),
            None
        )

    vehicles = _review_state.get("vehicles", [])
    quality_score = _review_state.get("quality_score", 0)
    thread_id = _review_state.get("thread_id", "N/A")
    csv_path = _review_state.get("csv_path")

    status_msg = f"""### Review in Progress

- **Thread ID:** {thread_id}
- **Vehicles:** {len(vehicles)}
- **Quality Score:** {quality_score*100:.0f}%
- **CSV Path:** {csv_path or 'N/A'}

**Instructions:**
1. Download the CSV file below
2. Open in Excel and review/edit the data
3. Upload the edited CSV (optional)
4. Click Approve or Reject to continue
"""

    # Create preview table
    df = format_vehicle_table(vehicles)

    return status_msg, df, csv_path


def download_review_csv() -> Optional[str]:
    """Get the review CSV path for download."""
    global _review_state
    csv_path = _review_state.get("csv_path")
    if csv_path and Path(csv_path).exists():
        return csv_path
    return None


def upload_edited_csv(file_obj) -> Tuple[str, pd.DataFrame]:
    """Process uploaded edited CSV and show changes."""
    global _review_state

    if file_obj is None:
        return "No file uploaded.", pd.DataFrame()

    try:
        import_service = get_csv_import_service()
        original_vehicles = _review_state.get("vehicles", [])

        # Import and detect changes
        imported_vehicles, changes, errors = import_service.import_csv(
            filepath=file_obj.name,
            original_vehicles=original_vehicles
        )

        # Store the edited file path
        _review_state["edited_csv_path"] = file_obj.name

        # Build summary
        if changes:
            change_summary = import_service.get_change_summary(changes)
            summary_msg = f"""### Changes Detected

- **Total Changes:** {change_summary['total_changes']}
- **Vehicles Modified:** {change_summary['vehicles_modified']}
- **Fields Modified:** {change_summary['fields_modified']}

**Changes by Type:**
"""
            for change_type, count in change_summary['by_type'].items():
                if count > 0:
                    summary_msg += f"- {change_type.title()}: {count}\n"
        else:
            summary_msg = "### No Changes Detected\n\nThe uploaded CSV matches the original data."

        # Show validation errors if any
        if errors:
            summary_msg += "\n\n**Validation Warnings:**\n"
            for err in errors[:5]:  # Show first 5 errors
                summary_msg += f"- Row {err.row_id}, {err.field}: {err.error}\n"
            if len(errors) > 5:
                summary_msg += f"- ... and {len(errors) - 5} more\n"

        # Create preview of imported data
        df = format_vehicle_table(imported_vehicles)

        return summary_msg, df

    except Exception as e:
        return f"### Error Processing CSV\n\n{str(e)}", pd.DataFrame()


def approve_review(reviewer_id: str) -> Tuple[str, pd.DataFrame, List[str]]:
    """Approve the review and generate presentations."""
    global _review_state

    if not _review_state.get("is_paused"):
        return "### Error\n\nNo active review to approve.", pd.DataFrame(), []

    if not reviewer_id.strip():
        return "### Error\n\nPlease enter a Reviewer ID.", pd.DataFrame(), []

    thread_id = _review_state.get("thread_id")
    edited_csv_path = _review_state.get("edited_csv_path")
    vehicles = _review_state.get("vehicles", [])
    quality_score = _review_state.get("quality_score", 0)

    try:
        # If edited CSV was uploaded, import the changes
        changes_count = 0
        if edited_csv_path:
            try:
                import_service = get_csv_import_service()
                imported_vehicles, changes, errors = import_service.import_csv(
                    filepath=edited_csv_path,
                    original_vehicles=vehicles
                )
                if changes:
                    vehicles = imported_vehicles
                    changes_count = len(changes)

                    # Log to audit
                    audit_service = get_audit_service()
                    session_id = _review_state.get("session_id")
                    if session_id:
                        changes_list = [
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
                        audit_service.log_edits(session_id, changes_list)
            except Exception as import_err:
                print(f"Warning: Could not import edited CSV: {import_err}")

        # Generate presentations directly
        from src.agents.presentation_generator import presentation_node
        from src.state.state import BenchmarkingState, WorkflowStatus, ReviewStatus

        # Create a minimal state for presentation generation
        presentation_state = {
            "all_vehicles": vehicles,
            "quality_validation": {"overall_quality_score": quality_score},
            "workflow_status": WorkflowStatus.GENERATING_PRESENTATION,
            "review_status": ReviewStatus.APPROVED_WITH_EDITS.value if changes_count > 0 else ReviewStatus.APPROVED.value,
            "errors": [],
            "warnings": [],
        }

        # Run presentation generator
        result = presentation_node(presentation_state)

        # Get presentation files
        pres_result = result.get("presentation_result", {})
        paths = pres_result.get("all_presentation_paths", [])
        files = [p for p in paths if Path(p).exists()]

        # Complete audit session
        try:
            audit_service = get_audit_service()
            session_id = _review_state.get("session_id")
            if session_id:
                audit_service.complete_session(
                    session_id=session_id,
                    status=ReviewStatus.APPROVED_WITH_EDITS.value if changes_count > 0 else ReviewStatus.APPROVED.value,
                    reviewer_id=reviewer_id.strip(),
                    csv_import_path=edited_csv_path
                )
        except Exception as audit_err:
            print(f"Warning: Audit completion failed: {audit_err}")

        # Reset review state
        _review_state = {
            "thread_id": None,
            "csv_path": None,
            "vehicles": [],
            "quality_score": 0.0,
            "is_paused": False,
            "edited_csv_path": None,
        }

        return (
            f"""### Workflow Complete

- **Review Status:** Approved{' with edits' if changes_count > 0 else ''}
- **Reviewer:** {reviewer_id}
- **Changes Applied:** {changes_count}
- **Presentations Generated:** {len(files)}
""",
            format_vehicle_table(vehicles),
            files
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"### Error\n\n{str(e)}", pd.DataFrame(), []


def reject_review(reviewer_id: str, rejection_reason: str) -> str:
    """Reject the review and end workflow."""
    global _review_state

    if not _review_state.get("is_paused"):
        return "### Error\n\nNo active review to reject."

    if not reviewer_id.strip():
        return "### Error\n\nPlease enter a Reviewer ID."

    if not rejection_reason.strip():
        return "### Error\n\nPlease provide a rejection reason."

    try:
        # Complete audit session with rejection
        try:
            audit_service = get_audit_service()
            session_id = _review_state.get("session_id")
            if session_id:
                audit_service.complete_session(
                    session_id=session_id,
                    status="rejected",
                    reviewer_id=reviewer_id.strip(),
                    rejection_reason=rejection_reason.strip()
                )
        except Exception as audit_err:
            print(f"Warning: Audit completion failed: {audit_err}")

        # Reset review state
        _review_state = {
            "thread_id": None,
            "csv_path": None,
            "vehicles": [],
            "quality_score": 0.0,
            "is_paused": False,
            "edited_csv_path": None,
        }

        return f"""### Review Rejected

- **Reviewer:** {reviewer_id}
- **Reason:** {rejection_reason}

The workflow has been stopped. Run a new extraction to start over.
"""

    except Exception as e:
        return f"### Error\n\n{str(e)}"


# =============================================================================
# GRADIO APP
# =============================================================================

def create_app():
    """Create Gradio application."""

    with gr.Blocks(title="E-Powertrain Benchmarking") as app:

        gr.Markdown("""
        # E-Powertrain Benchmarking System
        Extract electric vehicle specifications from OEM websites and generate presentations.
        """)

        with gr.Tabs():
            # TAB 1: Extraction
            with gr.Tab("Extract Data"):
                with gr.Row():
                    with gr.Column(scale=2):
                        urls_input = gr.Textbox(
                            label="OEM URLs (one per line)",
                            placeholder="Enter URLs to electric vehicle specification pages...",
                            lines=5
                        )

                        gr.Markdown("**Quick Add:**")
                        with gr.Row():
                            for oem in list(SAMPLE_OEMS.keys()):
                                btn = gr.Button(oem, size="sm")
                                btn.click(
                                    fn=lambda t, o=oem: add_sample_url(t, o),
                                    inputs=[urls_input],
                                    outputs=[urls_input]
                                )

                    with gr.Column(scale=1):
                        mode_dropdown = gr.Dropdown(
                            choices=[
                                ("Intelligent Extraction", "intelligent"),
                            ],
                            value="intelligent",
                            label="Extraction Mode"
                        )

                        enable_review_checkbox = gr.Checkbox(
                            label="Enable Human Review",
                            value=True,
                            info="Pause for review before generating presentations"
                        )

                        run_btn = gr.Button("Start Extraction", variant="primary", size="lg")

                status_output = gr.Markdown(value="*Ready to extract*")

                gr.Markdown("### Results")
                results_table = gr.Dataframe(
                    headers=["OEM", "Vehicle", "Battery (kWh)", "Range (km)",
                             "Motor (kW)", "Charging (kW)", "Weight (kg)", "Completeness"],
                    interactive=False
                )

                gr.Markdown("### Downloads")
                files_output = gr.Files(label="Generated Presentations")

            # TAB 2: Review & Approve
            with gr.Tab("Review & Approve") as review_tab:
                gr.Markdown("### Human Review Workflow")
                gr.Markdown("Review extracted data, make edits in Excel if needed, then approve or reject.")

                review_status_output = gr.Markdown(value="*Click 'Refresh Status' or run an extraction with review enabled*")

                with gr.Row():
                    refresh_review_btn = gr.Button("Refresh Status", variant="secondary")

                gr.Markdown("### Data Preview")
                review_table = gr.Dataframe(
                    headers=["OEM", "Vehicle", "Battery (kWh)", "Range (km)",
                             "Motor (kW)", "Charging (kW)", "Weight (kg)", "Completeness"],
                    interactive=False
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Step 1: Download CSV")
                        review_csv_file = gr.File(label="Review CSV (click to download)", interactive=False)
                        download_csv_btn = gr.Button("Load CSV for Download", variant="secondary")

                    with gr.Column():
                        gr.Markdown("#### Step 2: Upload Edited CSV (Optional)")
                        upload_csv_input = gr.File(
                            label="Upload Edited CSV",
                            file_types=[".csv"]
                        )
                        upload_status = gr.Markdown(value="")

                gr.Markdown("### Step 3: Decision")
                with gr.Row():
                    reviewer_id_input = gr.Textbox(
                        label="Reviewer ID",
                        placeholder="Enter your name or ID",
                        scale=2
                    )

                with gr.Row():
                    with gr.Column():
                        approve_btn = gr.Button("Approve", variant="primary")
                    with gr.Column():
                        reject_btn = gr.Button("Reject", variant="stop")

                rejection_reason_input = gr.Textbox(
                    label="Rejection Reason (required for rejection)",
                    placeholder="Enter reason for rejection...",
                    visible=True
                )

                decision_output = gr.Markdown(value="")
                decision_files = gr.Files(label="Generated Presentations", visible=True)

                # Review tab event handlers

                # Auto-refresh when tab is selected
                review_tab.select(
                    fn=get_review_status,
                    outputs=[review_status_output, review_table, review_csv_file]
                )

                refresh_review_btn.click(
                    fn=get_review_status,
                    outputs=[review_status_output, review_table, review_csv_file]
                )

                download_csv_btn.click(
                    fn=download_review_csv,
                    outputs=[review_csv_file]
                )

                upload_csv_input.change(
                    fn=upload_edited_csv,
                    inputs=[upload_csv_input],
                    outputs=[upload_status, review_table]
                )

                approve_btn.click(
                    fn=approve_review,
                    inputs=[reviewer_id_input],
                    outputs=[decision_output, review_table, decision_files]
                )

                reject_btn.click(
                    fn=reject_review,
                    inputs=[reviewer_id_input, rejection_reason_input],
                    outputs=[decision_output]
                )

            # TAB 3: Export
            with gr.Tab("Export Data"):
                json_output = gr.Code(label="JSON Data", language="json", lines=15)

                with gr.Row():
                    export_btn = gr.Button("Export as CSV")
                    export_file = gr.File(label="Download")

                export_btn.click(fn=export_csv, inputs=[json_output], outputs=[export_file])

            # TAB 4: Help
            with gr.Tab("Help"):
                gr.Markdown("""
                ## How to Use

                ### Standard Workflow (with Review)
                1. **Enter URLs**: Paste OEM electric vehicle specification page URLs
                2. **Start Extraction**: Keep "Enable Human Review" checked
                3. **Go to Review Tab**: Once extraction pauses, switch to "Review & Approve"
                4. **Download CSV**: Get the extracted data in Excel-compatible format
                5. **Edit in Excel**: Make corrections or additions as needed
                6. **Upload Edited CSV**: Upload your edited version (optional)
                7. **Approve or Reject**: Complete the review to generate presentations

                ### Quick Workflow (without Review)
                1. Uncheck "Enable Human Review" before extraction
                2. Presentations generate automatically after validation

                ## Supported OEMs
                - MAN Truck & Bus
                - Mercedes-Benz Trucks
                - Volvo Trucks
                - Scania
                - DAF Trucks
                - And more...

                ## Data Fields Extracted
                - Battery capacity (kWh)
                - Motor power (kW)
                - Range (km)
                - Charging speed (DC/MCS)
                - Weight (GVW/GCW)
                - Available configurations
                """)

        # Main click handler
        run_btn.click(
            fn=process_extraction,
            inputs=[urls_input, mode_dropdown, enable_review_checkbox],
            outputs=[status_output, results_table, files_output, json_output]
        )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import socket

    def find_free_port(start=7860, end=7900):
        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 7860

    port = find_free_port()
    print(f"\n{'='*50}")
    print(f"E-Powertrain Benchmarking System")
    print(f"http://127.0.0.1:{port}")
    print(f"{'='*50}\n")

    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=port, share=False)
