"""
Template Analysis Agent for Plug-and-Play PowerPoint System

Uses LLM to intelligently map template structure to vehicle data fields.
Generates JSON mapping configuration that can be used by the dynamic generator.
Supports multi-slide templates with slide_index and pagination_rules.

This is a standalone module - does NOT modify existing agents.

Version History:
- 1.0.0: Initial release
- 2.0.0: Added multi-slide support with slide_index and pagination_rules
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


# Vehicle data schema - defines what fields are available for mapping
VEHICLE_DATA_SCHEMA = {
    "oem_info": {
        "oem_name": {"type": "str", "description": "OEM/Manufacturer name (e.g., MAN, Volvo)"},
        "country": {"type": "str", "description": "OEM headquarters country"},
        "website": {"type": "str", "description": "OEM website domain"},
        "category": {"type": "str", "description": "OEM category (e.g., OEM - Commercial Trucks)"},
    },
    "vehicle_specs": {
        "vehicle_name": {"type": "str", "description": "Vehicle model name"},
        "powertrain_type": {"type": "str", "description": "Powertrain type (BEV, FCEV, Hybrid)"},
        "category": {"type": "str", "description": "Vehicle category (Heavy-duty Truck, Bus, etc.)"},
        "battery_capacity_kwh": {"type": "float", "description": "Battery capacity in kWh"},
        "range_km": {"type": "float", "description": "Range in kilometers"},
        "motor_power_kw": {"type": "float", "description": "Motor power in kW"},
        "motor_torque_nm": {"type": "float", "description": "Motor torque in Nm"},
        "dc_charging_kw": {"type": "float", "description": "DC charging power in kW"},
        "charging_time_minutes": {"type": "float", "description": "Charging time in minutes"},
        "gvw_kg": {"type": "float", "description": "Gross Vehicle Weight in kg"},
        "payload_capacity_kg": {"type": "float", "description": "Payload capacity in kg"},
    },
    "additional_specs": {
        "wheel_formula": {"type": "str", "description": "Wheel configuration (4x2, 6x2, etc.)"},
        "wheelbase_mm": {"type": "float", "description": "Wheelbase in mm"},
        "fuel_cell_kw": {"type": "float", "description": "Fuel cell power in kW (for FCEV)"},
        "h2_tank_kg": {"type": "float", "description": "Hydrogen tank capacity in kg"},
        "start_of_production": {"type": "str", "description": "Start of production year"},
        "markets": {"type": "str", "description": "Available markets"},
    },
    "computed_fields": {
        "expected_highlights": {"type": "list", "description": "List of key highlights/features"},
        "assessment": {"type": "list", "description": "List of assessment points"},
        "technologies": {"type": "list", "description": "List of technologies used"},
    }
}


class TemplateAnalysisAgent:
    """
    LLM agent that analyzes template structure and generates field mappings.

    Usage:
        agent = TemplateAnalysisAgent()
        mapping = agent.generate_mapping(template_analysis)
        print(json.dumps(mapping, indent=2))
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the agent with an LLM model.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(model=self.model, temperature=0)
            except ImportError:
                raise ImportError("langchain_openai required. Install with: pip install langchain-openai")
        return self._client

    def generate_mapping(
        self,
        template_analysis: Dict[str, Any],
        custom_data_schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a mapping configuration from template analysis.

        Args:
            template_analysis: Output from TemplateAnalyzer.analyze_for_mapping()
            custom_data_schema: Optional custom data schema (defaults to VEHICLE_DATA_SCHEMA)

        Returns:
            Mapping configuration JSON
        """
        data_schema = custom_data_schema or VEHICLE_DATA_SCHEMA

        prompt = self._build_prompt(template_analysis, data_schema)

        try:
            response = self.client.invoke(prompt)
            mapping = self._parse_response(response.content)

            # Add metadata
            mapping["template_name"] = template_analysis.get("template_name", "unknown")
            mapping["template_hash"] = template_analysis.get("template_hash", "")
            mapping["generated_at"] = datetime.now().isoformat()
            mapping["model_used"] = self.model

            return mapping

        except Exception as e:
            return {
                "error": str(e),
                "template_name": template_analysis.get("template_name", "unknown"),
                "shape_mappings": [],
                "generated_at": datetime.now().isoformat()
            }

    def _build_prompt(self, template_analysis: Dict, data_schema: Dict) -> str:
        """Build the prompt for the LLM with multi-slide support."""
        return f"""You are an expert at analyzing PowerPoint templates and mapping them to data schemas.

## TASK
Analyze this PowerPoint template structure and create a mapping configuration that maps each template element to the appropriate data field from the vehicle data schema.

## TEMPLATE STRUCTURE
```json
{json.dumps(template_analysis, indent=2)}
```

## AVAILABLE DATA FIELDS
```json
{json.dumps(data_schema, indent=2)}
```

## MAPPING RULES
1. For TEXT elements: Map to a single data field
2. For TABLE elements: Map each row to a data field
3. For PLACEHOLDER elements: Usually title, footer, or slide number
4. Use "format" to specify how values should be formatted (e.g., "{{value}} kWh")
5. Use "default" for fallback values when data is missing
6. For lists (highlights, assessment, technologies): Use "type": "bullet_list"
7. IMPORTANT: Include "slide_index" for each shape (0 for first slide, 1 for second, etc.)
8. For product tables that may have many products, add "repeatable_for": "products" to enable multi-slide pagination

## OUTPUT FORMAT
Return ONLY valid JSON (no markdown, no explanation) with this structure:
{{
    "version": "2.0.0",
    "supports_multi_slide": true,
    "pagination_rules": {{
        "products": {{
            "items_per_slide": 2,
            "overflow_behavior": "duplicate_slide",
            "source_slide_index": 0,
            "shape_id": 8,
            "description": "Duplicate slide when more than 2 products"
        }}
    }},
    "shape_mappings": [
        {{
            "shape_id": 2,
            "shape_name": "Title placeholder",
            "slide_index": 0,
            "type": "text",
            "data_field": "oem_info.oem_name",
            "format": "{{value}}",
            "default": "Unknown OEM"
        }},
        {{
            "shape_id": 7,
            "shape_name": "Company Info Table",
            "slide_index": 0,
            "type": "table",
            "row_mappings": [
                {{"row_index": 0, "col_index": 1, "data_field": "oem_info.country", "format": "{{value}}", "default": ""}},
                {{"row_index": 1, "col_index": 1, "data_field": "oem_info.website", "format": "{{value}}", "default": ""}}
            ]
        }},
        {{
            "shape_id": 8,
            "shape_name": "Products Table",
            "slide_index": 0,
            "type": "table",
            "repeatable_for": "products",
            "row_mappings": [
                {{"row_index": 0, "data_field": "products[].name", "format": "{{value}}", "default": ""}}
            ],
            "product_columns": [1, 2],
            "max_products": 2
        }},
        {{
            "shape_id": 25,
            "shape_name": "Highlights",
            "slide_index": 0,
            "type": "bullet_list",
            "data_field": "computed_fields.expected_highlights",
            "max_items": 4
        }}
    ],
    "unmapped_shapes": [
        {{"shape_id": 6, "reason": "Embedded OLE object - cannot map"}}
    ]
}}

Analyze the template and generate the mapping with multi-slide support. Return ONLY the JSON."""

    def _parse_response(self, response_text: str) -> Dict:
        """Parse LLM response and extract JSON."""
        # Try direct JSON parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object pattern
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return error if nothing works
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response_text[:500],
            "shape_mappings": []
        }

    def generate_mapping_manual(self, template_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a basic mapping without LLM (rule-based fallback).
        Includes multi-slide support with slide_index and pagination_rules.

        Useful when LLM is unavailable or for simple templates.
        """
        mapping = {
            "template_name": template_analysis.get("template_name", "unknown"),
            "template_hash": template_analysis.get("template_hash", ""),
            "version": "2.0.0",
            "supports_multi_slide": True,
            "generated_at": datetime.now().isoformat(),
            "generation_method": "rule_based",
            "pagination_rules": {
                "products": {
                    "items_per_slide": 2,
                    "overflow_behavior": "duplicate_slide",
                    "source_slide_index": 0,
                    "description": "Duplicate slide when more than 2 products"
                }
            },
            "shape_mappings": [],
            "unmapped_shapes": []
        }

        # Track if we found a products table for pagination_rules
        products_table_shape_id = None

        for element in template_analysis.get("mappable_elements", []):
            shape_id = element["shape_id"]
            shape_type = element["type"]
            content = element.get("current_content", "").lower()
            name = element.get("shape_name", "").lower()
            slide_index = element.get("slide_index", 0)

            # Rule-based mapping
            shape_mapping = {
                "shape_id": shape_id,
                "shape_name": element.get("shape_name", ""),
                "slide_index": slide_index,
                "type": "unknown"
            }

            # Text/Placeholder mapping
            if shape_type in ["PLACEHOLDER", "TEXT_BOX"]:
                if "title" in name or shape_id == 2:
                    shape_mapping["type"] = "text"
                    shape_mapping["data_field"] = "oem_info.oem_name"
                elif "highlight" in content or "highlight" in name:
                    shape_mapping["type"] = "bullet_list"
                    shape_mapping["data_field"] = "computed_fields.expected_highlights"
                elif "assessment" in content or "assessment" in name:
                    shape_mapping["type"] = "bullet_list"
                    shape_mapping["data_field"] = "computed_fields.assessment"
                elif "technolog" in content or "technolog" in name:
                    shape_mapping["type"] = "bullet_list"
                    shape_mapping["data_field"] = "computed_fields.technologies"
                else:
                    mapping["unmapped_shapes"].append({
                        "shape_id": shape_id,
                        "reason": f"Could not determine mapping for: {name}"
                    })
                    continue

            # Table mapping
            elif shape_type == "TABLE":
                table_info = element.get("table_structure", {})
                shape_mapping["type"] = "table"
                shape_mapping["row_mappings"] = []

                row_labels = table_info.get("row_labels", [])

                # Check if this looks like a products table (multiple columns, vehicle-related labels)
                is_products_table = False
                num_cols = table_info.get("dimensions", {}).get("cols", 0)
                if num_cols >= 3:
                    product_related_labels = ["name", "model", "vehicle", "product", "battery", "range", "power"]
                    if any(any(label.lower().find(pl) >= 0 for pl in product_related_labels) for label in row_labels):
                        is_products_table = True

                if is_products_table:
                    shape_mapping["repeatable_for"] = "products"
                    shape_mapping["product_columns"] = [1, 2]
                    shape_mapping["max_products"] = 2
                    products_table_shape_id = shape_id
                    # Update pagination rules with actual shape_id
                    mapping["pagination_rules"]["products"]["shape_id"] = shape_id

                for row_idx, label in enumerate(row_labels):
                    label_lower = label.lower()
                    field = None

                    # Map common labels to fields
                    if "country" in label_lower:
                        field = "oem_info.country"
                    elif "website" in label_lower:
                        field = "oem_info.website"
                    elif "category" in label_lower:
                        field = "oem_info.category"
                    elif "name" in label_lower and "vehicle" not in label_lower:
                        field = "products[].name" if is_products_table else "vehicle_specs.vehicle_name"
                    elif "battery" in label_lower:
                        field = "products[].battery" if is_products_table else "vehicle_specs.battery_capacity_kwh"
                    elif "range" in label_lower:
                        field = "products[].range" if is_products_table else "vehicle_specs.range_km"
                    elif "power" in label_lower or "performance" in label_lower:
                        field = "products[].performance" if is_products_table else "vehicle_specs.motor_power_kw"
                    elif "charg" in label_lower:
                        field = "products[].charging" if is_products_table else "vehicle_specs.dc_charging_kw"
                    elif "gvw" in label_lower or "weight" in label_lower:
                        field = "products[].gvw_gcw" if is_products_table else "vehicle_specs.gvw_kg"
                    elif "wheel" in label_lower:
                        field = "products[].wheel_formula" if is_products_table else "additional_specs.wheel_formula"

                    if field:
                        row_mapping = {
                            "row_index": row_idx,
                            "data_field": field,
                            "format": "{value}",
                            "default": "N/A"
                        }
                        if not is_products_table:
                            row_mapping["col_index"] = 1  # Assume value in second column
                        shape_mapping["row_mappings"].append(row_mapping)

            else:
                mapping["unmapped_shapes"].append({
                    "shape_id": shape_id,
                    "reason": f"Unsupported shape type: {shape_type}"
                })
                continue

            if shape_mapping.get("type") != "unknown":
                mapping["shape_mappings"].append(shape_mapping)

        return mapping


def get_data_schema() -> Dict:
    """Return the vehicle data schema for reference."""
    return VEHICLE_DATA_SCHEMA


# =============================================================================
# CLI for standalone testing
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('src/agents/template_analysis_agent.py', ''))

    from src.tools.template_analyzer import TemplateAnalyzer

    print("=" * 60)
    print("Template Analysis Agent")
    print("=" * 60)

    # Default to IAA template
    template_path = sys.argv[1] if len(sys.argv) > 1 else "templates/IAA_Template.pptx"

    print(f"\nAnalyzing template: {template_path}")

    # Step 1: Analyze template
    analyzer = TemplateAnalyzer()
    template_analysis = analyzer.analyze_for_mapping(template_path)

    print(f"Found {len(template_analysis['mappable_elements'])} mappable elements")

    # Step 2: Generate mapping (try LLM first, fall back to rules)
    agent = TemplateAnalysisAgent()

    use_llm = os.getenv("OPENAI_API_KEY") is not None
    print(f"\nUsing {'LLM' if use_llm else 'rule-based'} mapping generation...")

    if use_llm:
        try:
            mapping = agent.generate_mapping(template_analysis)
        except Exception as e:
            print(f"LLM failed ({e}), falling back to rule-based...")
            mapping = agent.generate_mapping_manual(template_analysis)
    else:
        mapping = agent.generate_mapping_manual(template_analysis)

    # Show results
    print(f"\n--- Generated Mapping ---")
    print(f"Template: {mapping.get('template_name')}")
    print(f"Shape mappings: {len(mapping.get('shape_mappings', []))}")
    print(f"Unmapped shapes: {len(mapping.get('unmapped_shapes', []))}")

    if mapping.get("shape_mappings"):
        print(f"\nMappings:")
        for m in mapping["shape_mappings"][:5]:
            print(f"  Shape {m['shape_id']} ({m['type']}): {m.get('data_field', 'multiple fields')}")

    # Save mapping
    output_path = f"src/config/template_schemas/{mapping['template_name']}_mapping.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {output_path}")
