"""
E-Powertrain Benchmarking System - Gradio Frontend

A professional web interface for extracting electric vehicle specifications
from OEM websites and generating PowerPoint presentations.
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import sys

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.graph.runtime import run_benchmark
from src.state.state import ScrapingMode


def process_urls(urls_text: str, mode: str):
    """
    Main processing function called by Gradio.

    Args:
        urls_text: Newline-separated URLs
        mode: "perplexity", "intelligent", or "auto"

    Returns:
        Tuple of (status_html, vehicles_dataframe, list_of_file_paths)
    """
    # 1. Parse and validate URLs
    urls = [u.strip() for u in urls_text.strip().split('\n') if u.strip()]
    if not urls:
        return "<span style='color: red;'>Error: No URLs provided</span>", None, []

    try:
        # 2. Run the benchmarking workflow
        scraping_mode = ScrapingMode(mode)
        result = run_benchmark(urls, mode=scraping_mode, verbose=True)

        # 3. Format status message
        status = format_status(result)

        # 4. Create vehicles dataframe
        df = format_vehicle_table(result.get('all_vehicles', []))

        # 5. Get presentation file paths
        files = get_presentation_files(result)

        return status, df, files

    except Exception as e:
        error_msg = f"<span style='color: red;'><b>Error:</b> {str(e)}</span>"
        return error_msg, None, []


def format_status(result: dict) -> str:
    """Format workflow status as HTML"""
    status = result.get('workflow_status', 'unknown')
    if hasattr(status, 'value'):
        status = status.value

    vehicles = len(result.get('all_vehicles', []))
    quality = result.get('quality_validation', {}) or {}
    score = quality.get('overall_quality_score', 0) * 100

    if status == 'completed':
        return f"""
        <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border: 1px solid #c3e6cb;'>
            <b style='color: #155724;'>Completed</b> - {vehicles} vehicles extracted, Quality Score: {score:.1f}%
        </div>
        """
    else:
        errors = result.get('errors', [])
        error_text = ', '.join(errors) if errors else 'Unknown error'
        return f"""
        <div style='padding: 10px; background-color: #f8d7da; border-radius: 5px; border: 1px solid #f5c6cb;'>
            <b style='color: #721c24;'>Failed</b> - {error_text}
        </div>
        """


def format_vehicle_table(vehicles: list) -> pd.DataFrame:
    """Convert vehicle list to pandas DataFrame for display"""
    if not vehicles:
        return pd.DataFrame()

    rows = []
    for v in vehicles:
        rows.append({
            'OEM': v.get('oem_name', 'N/A'),
            'Vehicle': v.get('vehicle_name', 'N/A'),
            'Battery (kWh)': v.get('battery_capacity_kwh') or 'N/A',
            'Range (km)': v.get('range_km') or 'N/A',
            'Motor (kW)': v.get('motor_power_kw') or 'N/A',
            'DC Charging (kW)': v.get('dc_charging_kw') or 'N/A',
            'GVW (kg)': v.get('gvw_kg') or 'N/A',
            'Completeness': f"{(v.get('data_completeness_score', 0) * 100):.0f}%"
        })

    return pd.DataFrame(rows)


def get_presentation_files(result: dict) -> list:
    """Extract presentation file paths for download"""
    pres_result = result.get('presentation_result', {}) or {}
    paths = pres_result.get('all_presentation_paths', [])
    # Filter to only existing files
    return [p for p in paths if Path(p).exists()]


# No default URLs - users will provide their own
DEFAULT_URLS = ""

# Build Gradio Interface
with gr.Blocks() as app:

    gr.Markdown("""
    # E-Powertrain Benchmarking System

    Extract electric vehicle specifications from OEM websites and generate IAA-format presentations.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            urls_input = gr.Textbox(
                label="Enter OEM URLs (one per line)",
                placeholder="Enter URLs to electric vehicle specification pages, e.g.:\nhttps://www.man.eu/global/en/truck/electric-trucks/overview.html\nhttps://www.volvotrucks.com/en-en/trucks/electric.html",
                lines=5,
                value=DEFAULT_URLS
            )
        with gr.Column(scale=1):
            mode_dropdown = gr.Dropdown(
                choices=[
                    ("Fast (Single Page)", "perplexity"),
                    ("Deep (Multi-page)", "intelligent"),
                    ("Auto", "auto")
                ],
                value="perplexity",
                label="Extraction Mode",
                info="Fast mode extracts from single page. Deep mode crawls multiple pages for more complete data."
            )
            run_btn = gr.Button("Run Benchmarking", variant="primary", size="lg")

    status_output = gr.HTML(label="Status", elem_classes=["status-box"])

    gr.Markdown("### Vehicle Specifications")
    table_output = gr.Dataframe(
        label="Extracted Vehicles",
        headers=["OEM", "Vehicle", "Battery (kWh)", "Range (km)",
                 "Motor (kW)", "DC Charging (kW)", "GVW (kg)", "Completeness"],
        interactive=False,
        wrap=True
    )

    gr.Markdown("### Download Presentations")
    files_output = gr.Files(label="Generated Presentations")

    # Connect the button to the processing function
    run_btn.click(
        fn=process_urls,
        inputs=[urls_input, mode_dropdown],
        outputs=[status_output, table_output, files_output]
    )

    gr.Markdown("""
    ---
    **Tips:**
    - Use **Fast mode** for quick extraction from pages with all specs visible
    - Use **Deep mode** when specs are spread across multiple pages (higher quality, slower)
    - **Completeness** shows data confidence (higher = more fields extracted from source)
    """)


if __name__ == "__main__":
    import socket

    def find_free_port(start=7860, end=7900):
        """Find a free port in the given range"""
        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 7860  # fallback

    port = find_free_port()
    print(f"\n{'='*50}")
    print(f"E-Powertrain Benchmarking System")
    print(f"Access at: http://127.0.0.1:{port}")
    print(f"{'='*50}\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False
    )
