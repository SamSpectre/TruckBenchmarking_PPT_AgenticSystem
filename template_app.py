"""
Template Management App - Plug-and-Play PowerPoint Template System

Separate Gradio UI for managing PowerPoint templates.
Run alongside or independently from the main benchmarking app.

Usage:
    python template_app.py
    # Opens at http://127.0.0.1:7870 (or next available port)
"""

import sys
import os
import json
import socket
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from src.tools.template_analyzer import TemplateAnalyzer
from src.agents.template_analysis_agent import TemplateAnalysisAgent, VEHICLE_DATA_SCHEMA
from src.config.template_registry import TemplateRegistry
from src.tools.dynamic_ppt_generator import DynamicOEMPresentationGenerator


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

analyzer = TemplateAnalyzer()
agent = TemplateAnalysisAgent()
registry = TemplateRegistry()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_free_port(start: int = 7870, end: int = 7900) -> int:
    """Find a free port for the server."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return start


def get_available_templates():
    """Get list of available template names for dropdown."""
    templates = registry.list_templates()
    return [t["name"] for t in templates if "error" not in t]


# =============================================================================
# GRADIO FUNCTIONS
# =============================================================================

def analyze_template(file_obj):
    """Analyze an uploaded template file."""
    if file_obj is None:
        return "Please upload a PowerPoint template (.pptx)", None, None

    try:
        # Get the file path
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

        # Analyze the template
        full_analysis = analyzer.analyze(file_path)
        mapping_analysis = analyzer.analyze_for_mapping(file_path)

        # Format summary
        summary = f"""## Template Analysis

**Name**: {full_analysis['template_name']}
**Hash**: {full_analysis['template_hash']}
**Slides**: {full_analysis['total_slides']}

### Summary
- Total shapes: {full_analysis['summary']['total_shapes']}
- Tables: {full_analysis['summary']['total_tables']}
- Placeholders: {full_analysis['summary']['total_placeholders']}
- Text boxes: {full_analysis['summary']['total_text_boxes']}

### Mappable Elements
"""
        for elem in mapping_analysis['mappable_elements'][:10]:
            content = elem.get('current_content', '')[:30]
            summary += f"- **Shape {elem['shape_id']}** ({elem['type']}): {content}\n"

        if len(mapping_analysis['mappable_elements']) > 10:
            summary += f"\n... and {len(mapping_analysis['mappable_elements']) - 10} more"

        return summary, json.dumps(mapping_analysis, indent=2), file_path

    except Exception as e:
        return f"Error analyzing template: {str(e)}", None, None


def generate_mapping(analysis_json: str, use_llm: bool):
    """Generate mapping from analysis."""
    if not analysis_json:
        return "Please analyze a template first", None

    try:
        analysis = json.loads(analysis_json)

        if use_llm and os.getenv("OPENAI_API_KEY"):
            mapping = agent.generate_mapping(analysis)
            method = "LLM-generated"
        else:
            mapping = agent.generate_mapping_manual(analysis)
            method = "Rule-based"

        # Multi-slide info
        multi_slide = mapping.get('supports_multi_slide', False)
        pagination_rules = mapping.get('pagination_rules', {})
        version = mapping.get('version', '1.0.0')

        summary = f"""## Generated Mapping ({method})

**Template**: {mapping.get('template_name', 'unknown')}
**Version**: {version}
**Multi-Slide Support**: {'Yes' if multi_slide else 'No'}
**Shape Mappings**: {len(mapping.get('shape_mappings', []))}
**Unmapped Shapes**: {len(mapping.get('unmapped_shapes', []))}
"""
        if pagination_rules:
            summary += "\n### Pagination Rules\n"
            for field, rules in pagination_rules.items():
                items_per = rules.get('items_per_slide', 2)
                summary += f"- **{field}**: {items_per} items/slide (overflow: {rules.get('overflow_behavior', 'N/A')})\n"

        summary += "\n### Mappings\n"
        for m in mapping.get('shape_mappings', [])[:10]:
            field = m.get('data_field', 'N/A')
            slide_idx = m.get('slide_index', 0)
            repeatable = m.get('repeatable_for', '')
            rep_indicator = f" [repeatable: {repeatable}]" if repeatable else ""
            summary += f"- Shape {m['shape_id']} (slide {slide_idx}, {m['type']}): `{field}`{rep_indicator}\n"

        return summary, json.dumps(mapping, indent=2)

    except json.JSONDecodeError:
        return "Invalid JSON in analysis", None
    except Exception as e:
        return f"Error generating mapping: {str(e)}", None


def save_mapping(mapping_json: str, template_name: str):
    """Save mapping to registry."""
    if not mapping_json:
        return "No mapping to save"
    if not template_name:
        return "Please enter a template name"

    try:
        mapping = json.loads(mapping_json)
        result = registry.save_mapping(template_name, mapping, overwrite=True)

        if result["success"]:
            return f"Saved successfully to: {result['path']}"
        else:
            return f"Error: {result['error']}"

    except json.JSONDecodeError:
        return "Invalid JSON in mapping"
    except Exception as e:
        return f"Error saving: {str(e)}"


def load_existing_mapping(template_name: str):
    """Load an existing mapping from registry."""
    if not template_name:
        return "Select a template", None

    mapping = registry.load_mapping(template_name)

    if mapping is None:
        return f"Template '{template_name}' not found", None

    # Multi-slide info
    multi_slide = mapping.get('supports_multi_slide', False)
    pagination_rules = mapping.get('pagination_rules', {})

    summary = f"""## Loaded: {template_name}

**Hash**: {mapping.get('template_hash', 'N/A')}
**Version**: {mapping.get('version', '1.0.0')}
**Multi-Slide Support**: {'Yes' if multi_slide else 'No'}
**Mappings**: {len(mapping.get('shape_mappings', []))}
"""
    if pagination_rules:
        summary += "\n### Pagination Rules\n"
        for field, rules in pagination_rules.items():
            items_per = rules.get('items_per_slide', 2)
            summary += f"- **{field}**: {items_per} items/slide\n"

    return summary, json.dumps(mapping, indent=2)


def test_mapping(mapping_json: str, template_path: str):
    """Test a mapping by generating a sample presentation with multi-slide support."""
    if not mapping_json:
        return "No mapping to test", None

    # Use IAA template if no template uploaded
    if not template_path or not Path(template_path).exists():
        template_path = "templates/IAA_Template.pptx"

    try:
        mapping = json.loads(mapping_json)

        # Create test data with 5 products to test multi-slide overflow
        test_data = {
            "oem_info": {
                "oem_name": "Test OEM (Multi-Slide Test)",
                "country": "Germany",
                "website": "www.test-oem.com",
                "category": "OEM - Commercial Trucks"
            },
            "products": [
                {
                    "name": "Test Vehicle X1",
                    "wheel_formula": "6x2",
                    "wheelbase": "4,500 mm",
                    "gvw_gcw": "40,000 kg GVW",
                    "range": "400 km",
                    "battery": "500 kWh",
                    "fuel_cell": "N/A",
                    "h2_tank": "N/A",
                    "charging": "350 kW DC",
                    "performance": "350 kW",
                    "powertrain": "BEV",
                    "sop": "2025",
                    "markets": "EU",
                    "application": "Heavy-duty Truck"
                },
                {
                    "name": "Test Vehicle X2",
                    "wheel_formula": "6x4",
                    "wheelbase": "4,800 mm",
                    "gvw_gcw": "44,000 kg GVW",
                    "range": "500 km",
                    "battery": "600 kWh",
                    "fuel_cell": "N/A",
                    "h2_tank": "N/A",
                    "charging": "400 kW DC",
                    "performance": "400 kW",
                    "powertrain": "BEV",
                    "sop": "2025",
                    "markets": "EU, NA",
                    "application": "Heavy-duty Truck"
                },
                {
                    "name": "Test Vehicle X3 Long Range",
                    "wheel_formula": "6x2",
                    "wheelbase": "5,000 mm",
                    "gvw_gcw": "40,000 kg GVW",
                    "range": "700 km",
                    "battery": "800 kWh",
                    "fuel_cell": "N/A",
                    "h2_tank": "N/A",
                    "charging": "450 kW DC",
                    "performance": "450 kW",
                    "powertrain": "BEV",
                    "sop": "2026",
                    "markets": "Global",
                    "application": "Heavy-duty Truck"
                },
                {
                    "name": "Test Bus City 12m",
                    "wheel_formula": "4x2",
                    "wheelbase": "5,900 mm",
                    "gvw_gcw": "19,000 kg GVW",
                    "range": "300 km",
                    "battery": "350 kWh",
                    "fuel_cell": "N/A",
                    "h2_tank": "N/A",
                    "charging": "150 kW DC",
                    "performance": "250 kW",
                    "powertrain": "BEV",
                    "sop": "2024",
                    "markets": "EU",
                    "application": "City Bus"
                },
                {
                    "name": "Test Bus Coach 13m",
                    "wheel_formula": "6x2",
                    "wheelbase": "6,500 mm",
                    "gvw_gcw": "24,000 kg GVW",
                    "range": "400 km",
                    "battery": "450 kWh",
                    "fuel_cell": "N/A",
                    "h2_tank": "N/A",
                    "charging": "250 kW DC",
                    "performance": "300 kW",
                    "powertrain": "BEV",
                    "sop": "2025",
                    "markets": "EU, APAC",
                    "application": "Coach"
                }
            ],
            "computed_fields": {
                "expected_highlights": [
                    "5 vehicle portfolio test",
                    "Multi-slide overflow test",
                    "Battery capacity: up to 800 kWh",
                    "Range: up to 700 km"
                ],
                "assessment": [
                    "Multi-slide template test",
                    "All shapes populated",
                    "Overflow slides generated"
                ],
                "technologies": [
                    "Li-ion Battery",
                    "DC Fast Charging",
                    "High-Power Charging (450kW)"
                ]
            }
        }

        # Generate using dynamic generator
        from src.tools.dynamic_ppt_generator import DynamicPPTGenerator

        generator = DynamicPPTGenerator(mapping)
        output_path = f"outputs/template_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"

        result = generator.generate(template_path, test_data, output_path)

        if result["success"]:
            slides_info = f"Slides: {result.get('slides_created', 1)} (overflow: {result.get('overflow_slides', 0)})"
            return f"Test successful!\nOutput: {result['presentation_path']}\n{slides_info}\nShapes populated: {result['shapes_populated']}", output_path
        else:
            return f"Test failed: {result.get('error', 'Unknown error')}", None

    except json.JSONDecodeError:
        return "Invalid JSON in mapping", None
    except Exception as e:
        return f"Error: {str(e)}", None


def refresh_templates():
    """Refresh the template dropdown."""
    return gr.Dropdown(choices=get_available_templates())


# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Template Manager") as app:
        gr.Markdown("# PowerPoint Template Manager")
        gr.Markdown("Analyze, map, and manage PowerPoint templates for the benchmarking system.")

        with gr.Tabs():
            # =====================
            # TAB 1: Analyze & Map
            # =====================
            with gr.Tab("Analyze & Map"):
                with gr.Row():
                    with gr.Column(scale=1):
                        template_upload = gr.File(
                            label="Upload PowerPoint Template",
                            file_types=[".pptx"],
                            type="filepath"
                        )
                        analyze_btn = gr.Button("1. Analyze Template", variant="primary")

                        gr.Markdown("---")

                        use_llm = gr.Checkbox(
                            label="Use LLM for mapping (requires OpenAI API key)",
                            value=bool(os.getenv("OPENAI_API_KEY"))
                        )
                        generate_btn = gr.Button("2. Generate Mapping")

                        gr.Markdown("---")

                        template_name_input = gr.Textbox(
                            label="Template Name",
                            placeholder="my_custom_template"
                        )
                        save_btn = gr.Button("3. Save Mapping", variant="primary")
                        save_status = gr.Textbox(label="Save Status", interactive=False)

                    with gr.Column(scale=2):
                        analysis_summary = gr.Markdown(label="Analysis Summary")
                        analysis_json = gr.Code(
                            label="Analysis JSON",
                            language="json",
                            lines=15
                        )
                        template_path_state = gr.State(value=None)

                with gr.Row():
                    with gr.Column():
                        mapping_summary = gr.Markdown(label="Mapping Summary")
                        mapping_json = gr.Code(
                            label="Mapping JSON (editable)",
                            language="json",
                            lines=20
                        )

                # Event handlers
                analyze_btn.click(
                    fn=analyze_template,
                    inputs=[template_upload],
                    outputs=[analysis_summary, analysis_json, template_path_state]
                )

                generate_btn.click(
                    fn=generate_mapping,
                    inputs=[analysis_json, use_llm],
                    outputs=[mapping_summary, mapping_json]
                )

                save_btn.click(
                    fn=save_mapping,
                    inputs=[mapping_json, template_name_input],
                    outputs=[save_status]
                )

            # =====================
            # TAB 2: Manage Templates
            # =====================
            with gr.Tab("Manage Templates"):
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        label="Select Template",
                        choices=get_available_templates(),
                        interactive=True
                    )
                    refresh_btn = gr.Button("Refresh List")

                load_btn = gr.Button("Load Template", variant="primary")

                with gr.Row():
                    with gr.Column():
                        loaded_summary = gr.Markdown()
                    with gr.Column():
                        loaded_json = gr.Code(
                            label="Mapping JSON",
                            language="json",
                            lines=25
                        )

                # Event handlers
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_available_templates()),
                    outputs=[template_dropdown]
                )

                load_btn.click(
                    fn=load_existing_mapping,
                    inputs=[template_dropdown],
                    outputs=[loaded_summary, loaded_json]
                )

            # =====================
            # TAB 3: Test Template
            # =====================
            with gr.Tab("Test Template"):
                gr.Markdown("Test a mapping by generating a sample presentation.")

                with gr.Row():
                    with gr.Column():
                        test_template_dropdown = gr.Dropdown(
                            label="Select Template Mapping",
                            choices=get_available_templates()
                        )
                        load_for_test_btn = gr.Button("Load for Testing")

                        test_template_upload = gr.File(
                            label="Template File (optional - uses IAA if empty)",
                            file_types=[".pptx"],
                            type="filepath"
                        )

                    with gr.Column():
                        test_mapping_json = gr.Code(
                            label="Mapping to Test",
                            language="json",
                            lines=15
                        )

                test_btn = gr.Button("Run Test", variant="primary")
                test_result = gr.Textbox(label="Test Result", lines=3)
                test_output = gr.File(label="Generated Test File")

                # Event handlers
                load_for_test_btn.click(
                    fn=lambda t: (registry.load_mapping(t) and json.dumps(registry.load_mapping(t), indent=2)) or "",
                    inputs=[test_template_dropdown],
                    outputs=[test_mapping_json]
                )

                test_btn.click(
                    fn=test_mapping,
                    inputs=[test_mapping_json, test_template_upload],
                    outputs=[test_result, test_output]
                )

            # =====================
            # TAB 4: Data Schema
            # =====================
            with gr.Tab("Data Schema"):
                gr.Markdown("## Available Data Fields")
                gr.Markdown("These fields can be mapped to template elements:")
                gr.Code(
                    value=json.dumps(VEHICLE_DATA_SCHEMA, indent=2),
                    language="json",
                    label="Vehicle Data Schema",
                    lines=40
                )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Template Management App")
    print("=" * 60)

    port = find_free_port(7870, 7900)
    print(f"\nStarting server at: http://127.0.0.1:{port}")

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False
    )
