"""
Template Registry for Plug-and-Play PowerPoint System

Manages saved template mapping configurations.
Provides CRUD operations for template mappings.

This is a standalone module.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil


class TemplateRegistry:
    """
    Registry for managing PowerPoint template mappings.

    Usage:
        registry = TemplateRegistry()
        templates = registry.list_templates()
        mapping = registry.load_mapping("iaa_template")
        registry.save_mapping("custom_template", mapping_dict)
    """

    def __init__(self, config_dir: str = "src/config/template_schemas"):
        """
        Initialize the registry.

        Args:
            config_dir: Directory where mapping JSON files are stored
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available template mappings.

        Returns:
            List of template info dicts with name, hash, created_at, etc.
        """
        templates = []

        for path in self.config_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)

                templates.append({
                    "name": path.stem,
                    "file_path": str(path),
                    "template_name": mapping.get("template_name", path.stem),
                    "template_hash": mapping.get("template_hash", ""),
                    "description": mapping.get("description", ""),
                    "version": mapping.get("version", "1.0.0"),
                    "created_at": mapping.get("created_at", ""),
                    "shape_mappings_count": len(mapping.get("shape_mappings", [])),
                    "unmapped_shapes_count": len(mapping.get("unmapped_shapes", []))
                })
            except (json.JSONDecodeError, IOError) as e:
                templates.append({
                    "name": path.stem,
                    "file_path": str(path),
                    "error": str(e)
                })

        return sorted(templates, key=lambda x: x.get("name", ""))

    def get_template_names(self) -> List[str]:
        """Get list of template names only."""
        return [t["name"] for t in self.list_templates() if "error" not in t]

    def load_mapping(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a template mapping by name.

        Args:
            template_name: Name of the template (without .json extension)

        Returns:
            Mapping dict or None if not found
        """
        path = self.config_dir / f"{template_name}.json"

        if not path.exists():
            # Try with _mapping suffix
            path = self.config_dir / f"{template_name}_mapping.json"

        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def save_mapping(
        self,
        template_name: str,
        mapping: Dict[str, Any],
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Save a template mapping.

        Args:
            template_name: Name for the template
            mapping: Mapping configuration dict
            overwrite: Whether to overwrite existing mapping

        Returns:
            Result dict with success status and path
        """
        # Sanitize template name
        safe_name = "".join(c for c in template_name if c.isalnum() or c in "_-").lower()
        path = self.config_dir / f"{safe_name}.json"

        if path.exists() and not overwrite:
            return {
                "success": False,
                "error": f"Template '{safe_name}' already exists. Use overwrite=True to replace.",
                "existing_path": str(path)
            }

        # Add/update metadata
        mapping["template_name"] = template_name
        if "created_at" not in mapping:
            mapping["created_at"] = datetime.now().isoformat()
        mapping["updated_at"] = datetime.now().isoformat()

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)

            return {
                "success": True,
                "path": str(path),
                "template_name": safe_name
            }
        except IOError as e:
            return {
                "success": False,
                "error": str(e)
            }

    def delete_mapping(self, template_name: str) -> Dict[str, Any]:
        """
        Delete a template mapping.

        Args:
            template_name: Name of the template to delete

        Returns:
            Result dict with success status
        """
        path = self.config_dir / f"{template_name}.json"

        if not path.exists():
            return {
                "success": False,
                "error": f"Template '{template_name}' not found"
            }

        try:
            path.unlink()
            return {
                "success": True,
                "deleted": template_name
            }
        except IOError as e:
            return {
                "success": False,
                "error": str(e)
            }

    def duplicate_mapping(
        self,
        source_name: str,
        new_name: str
    ) -> Dict[str, Any]:
        """
        Duplicate an existing template mapping.

        Args:
            source_name: Name of template to copy
            new_name: Name for the new template

        Returns:
            Result dict with success status
        """
        mapping = self.load_mapping(source_name)

        if mapping is None:
            return {
                "success": False,
                "error": f"Source template '{source_name}' not found"
            }

        # Update metadata for new template
        mapping["template_name"] = new_name
        mapping["created_at"] = datetime.now().isoformat()
        mapping["description"] = f"Copy of {source_name}"

        return self.save_mapping(new_name, mapping)

    def validate_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a template mapping structure.

        Args:
            mapping: Mapping dict to validate

        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []

        # Required fields
        if "shape_mappings" not in mapping:
            errors.append("Missing 'shape_mappings' field")
        elif not isinstance(mapping["shape_mappings"], list):
            errors.append("'shape_mappings' must be a list")
        else:
            for i, sm in enumerate(mapping["shape_mappings"]):
                if "shape_id" not in sm:
                    errors.append(f"shape_mappings[{i}]: Missing 'shape_id'")
                if "type" not in sm:
                    warnings.append(f"shape_mappings[{i}]: Missing 'type', will default to 'text'")

        # Optional but recommended fields
        if "template_name" not in mapping:
            warnings.append("Missing 'template_name' - will use filename")
        if "template_hash" not in mapping:
            warnings.append("Missing 'template_hash' - cannot verify template version")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "shape_count": len(mapping.get("shape_mappings", []))
        }

    def get_mapping_path(self, template_name: str) -> Optional[str]:
        """Get the file path for a template mapping."""
        path = self.config_dir / f"{template_name}.json"
        return str(path) if path.exists() else None

    def import_mapping(self, file_path: str, new_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Import a mapping from an external JSON file.

        Args:
            file_path: Path to JSON file to import
            new_name: Optional new name (defaults to filename)

        Returns:
            Result dict
        """
        source = Path(file_path)

        if not source.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            with open(source, 'r', encoding='utf-8') as f:
                mapping = json.load(f)

            name = new_name or source.stem
            return self.save_mapping(name, mapping)

        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}

    def export_mapping(self, template_name: str, output_path: str) -> Dict[str, Any]:
        """
        Export a mapping to an external location.

        Args:
            template_name: Template to export
            output_path: Destination path

        Returns:
            Result dict
        """
        mapping = self.load_mapping(template_name)

        if mapping is None:
            return {"success": False, "error": f"Template '{template_name}' not found"}

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)

            return {"success": True, "exported_to": output_path}

        except IOError as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Template Registry")
    print("=" * 60)

    registry = TemplateRegistry()

    print(f"\nConfig directory: {registry.config_dir}")
    print(f"\nAvailable templates:")

    templates = registry.list_templates()

    if not templates:
        print("  (none)")
    else:
        for t in templates:
            if "error" in t:
                print(f"  - {t['name']}: ERROR - {t['error']}")
            else:
                print(f"  - {t['name']}")
                print(f"      Hash: {t['template_hash']}")
                print(f"      Mappings: {t['shape_mappings_count']}")
                print(f"      Unmapped: {t['unmapped_shapes_count']}")
