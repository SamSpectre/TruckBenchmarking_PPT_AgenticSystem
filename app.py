"""
E-Powertrain Benchmarking System - Gradio Frontend

Professional web interface for extracting electric vehicle specifications
from OEM websites and generating PowerPoint presentations.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import List, Optional

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.graph.runtime import run_benchmark
from src.state.state import ScrapingMode


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

def process_extraction(urls_text: str, mode: str, progress=gr.Progress()):
    """Main extraction function."""
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

        progress(0.3, desc="Fetching and extracting data...")
        result = run_benchmark(urls, mode=scraping_mode, verbose=True)

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
            status_msg = f"""### Extraction Complete

- **Vehicles Found:** {len(all_vehicles)}
- **OEMs Processed:** {len(oem_names)} ({', '.join(oem_names)})
- **Quality Score:** {quality_score*100:.0f}%
- **Avg Completeness:** {avg_completeness*100:.0f}%
"""
        else:
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
# GRADIO APP
# =============================================================================

def create_app():
    """Create Gradio application."""

    with gr.Blocks() as app:

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

            # TAB 2: Export
            with gr.Tab("Export Data"):
                json_output = gr.Code(label="JSON Data", language="json", lines=15)

                with gr.Row():
                    export_btn = gr.Button("Export as CSV")
                    export_file = gr.File(label="Download")

                export_btn.click(fn=export_csv, inputs=[json_output], outputs=[export_file])

            # TAB 3: Help
            with gr.Tab("Help"):
                gr.Markdown("""
                ## How to Use

                1. **Enter URLs**: Paste OEM electric vehicle specification page URLs (entry pages)
                2. **Start Extraction**: The system will automatically discover and crawl all spec pages
                3. **Review Results**: View extracted vehicle data in the table
                4. **Download**: Get PowerPoint presentations and export data as CSV/JSON

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
            inputs=[urls_input, mode_dropdown],
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
