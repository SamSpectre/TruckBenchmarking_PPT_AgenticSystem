"""
CSV Import Service for E-Powertrain Benchmarking System

Imports edited CSV files with validation and change detection.
Compares imported data against original to identify human edits.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error in the imported data."""
    row_id: int
    field: str
    value: Any
    error: str
    severity: str  # "error" or "warning"


@dataclass
class FieldChange:
    """Represents a change made to a field during review."""
    row_id: int
    vehicle_name: str
    field: str
    field_display_name: str
    original_value: Any
    new_value: Any
    change_type: str  # "modified", "added", "removed"


class CSVImportService:
    """
    Service for importing edited CSV files.

    Features:
    - Data type validation
    - Range validation for numeric fields
    - Change detection against original data
    - Detailed error reporting
    """

    # Reverse mapping: CSV header -> internal field
    # Supports both legacy format and new client-friendly format
    HEADER_TO_FIELD = {
        # Legacy format
        "Row ID": "row_id",
        "Vehicle Name": "vehicle_name",
        "OEM": "oem_name",
        "Category": "category",
        "Powertrain Type": "powertrain_type",
        "Battery (kWh)": "battery_capacity_kwh",
        "Battery Min (kWh)": "battery_capacity_min_kwh",
        "Battery Voltage (V)": "battery_voltage_v",
        "Battery Voltage Min (V)": "battery_voltage_min_v",
        "Battery Chemistry": "battery_chemistry",
        "Motor Power (kW)": "motor_power_kw",
        "Motor Power Min (kW)": "motor_power_min_kw",
        "Motor Torque (Nm)": "motor_torque_nm",
        "Motor Torque Min (Nm)": "motor_torque_min_nm",
        "Range (km)": "range_km",
        "Range Min (km)": "range_min_km",
        "Energy Consumption (kWh/100km)": "energy_consumption_kwh_per_100km",
        "DC Charging (kW)": "dc_charging_kw",
        "DC Charging Min (kW)": "dc_charging_min_kw",
        "MCS Charging (kW)": "mcs_charging_kw",
        "MCS Charging Min (kW)": "mcs_charging_min_kw",
        "Charging Time (min)": "charging_time_minutes",
        "Charging Time Max (min)": "charging_time_max_minutes",
        "GVW (kg)": "gvw_kg",
        "GVW Min (kg)": "gvw_min_kg",
        "GCW (kg)": "gcw_kg",
        "GCW Min (kg)": "gcw_min_kg",
        "Payload (kg)": "payload_capacity_kg",
        "Payload Min (kg)": "payload_capacity_min_kg",
        "Configurations": "available_configurations",
        "Source URL": "source_url",
        "Completeness Score": "data_completeness_score",
        "Extracted At": "extraction_timestamp",
        # New client-friendly format
        "No.": "row_id",
        "Vehicle Model": "vehicle_name",
        "Vehicle Category": "category",
        "Battery Capacity": "battery_capacity_combined",  # Combined field
        "Battery Voltage": "battery_voltage_combined",
        "Motor Power": "motor_power_combined",
        "Range": "range_combined",
        "DC Fast Charging": "dc_charging_combined",
        "MCS Charging": "mcs_charging_combined",
        "Gross Vehicle Weight (GVW)": "gvw_kg",
        "Gross Combination Weight (GCW)": "gcw_kg",
        "Payload Capacity": "payload_capacity_kg",
        "Data Quality": "data_completeness_score",
        "Source": "source_url",
    }

    # Fields that contain combined min-max values
    COMBINED_FIELDS = {
        "battery_capacity_combined": ("battery_capacity_min_kwh", "battery_capacity_kwh", "kWh"),
        "battery_voltage_combined": ("battery_voltage_min_v", "battery_voltage_v", "V"),
        "motor_power_combined": ("motor_power_min_kw", "motor_power_kw", "kW"),
        "range_combined": ("range_min_km", "range_km", "km"),
        "dc_charging_combined": ("dc_charging_min_kw", "dc_charging_kw", "kW"),
        "mcs_charging_combined": ("mcs_charging_min_kw", "mcs_charging_kw", "kW"),
    }

    # Field type mapping
    NUMERIC_FIELDS = {
        "battery_capacity_kwh", "battery_capacity_min_kwh",
        "battery_voltage_v", "battery_voltage_min_v",
        "motor_power_kw", "motor_power_min_kw",
        "motor_torque_nm", "motor_torque_min_nm",
        "range_km", "range_min_km",
        "energy_consumption_kwh_per_100km",
        "dc_charging_kw", "dc_charging_min_kw",
        "mcs_charging_kw", "mcs_charging_min_kw",
        "charging_time_minutes", "charging_time_max_minutes",
        "gvw_kg", "gvw_min_kg",
        "gcw_kg", "gcw_min_kg",
        "payload_capacity_kg", "payload_capacity_min_kg",
        "data_completeness_score",
    }

    # Validation ranges for numeric fields (field -> (min, max))
    VALIDATION_RANGES = {
        "battery_capacity_kwh": (10, 2000),
        "battery_capacity_min_kwh": (10, 2000),
        "battery_voltage_v": (100, 1500),
        "battery_voltage_min_v": (100, 1500),
        "motor_power_kw": (50, 1500),
        "motor_power_min_kw": (50, 1500),
        "motor_torque_nm": (100, 50000),
        "motor_torque_min_nm": (100, 50000),
        "range_km": (50, 1500),
        "range_min_km": (50, 1500),
        "energy_consumption_kwh_per_100km": (0.5, 5.0),
        "dc_charging_kw": (20, 1000),
        "dc_charging_min_kw": (20, 1000),
        "mcs_charging_kw": (100, 5000),
        "mcs_charging_min_kw": (100, 5000),
        "charging_time_minutes": (5, 1440),
        "charging_time_max_minutes": (5, 1440),
        "gvw_kg": (2000, 100000),
        "gvw_min_kg": (2000, 100000),
        "gcw_kg": (2000, 200000),
        "gcw_min_kg": (2000, 200000),
        "payload_capacity_kg": (500, 80000),
        "payload_capacity_min_kg": (500, 80000),
        "data_completeness_score": (0, 1),
    }

    def __init__(self):
        """Initialize CSV import service."""
        pass

    def import_csv(
        self,
        filepath: str,
        original_vehicles: Optional[List[Dict[str, Any]]] = None,
        strict_validation: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[FieldChange], List[ValidationError]]:
        """
        Import CSV file and validate data.

        Supports both legacy format and new client-friendly format.

        Args:
            filepath: Path to the CSV file to import
            original_vehicles: Original vehicle data for change detection
            strict_validation: If True, reject on any validation error

        Returns:
            Tuple of (imported_vehicles, changes, validation_errors)
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        vehicles = []
        validation_errors = []

        # Read all rows first to handle different formats
        with open(path, "r", encoding="utf-8-sig") as f:
            all_rows = list(csv.reader(f))

        if not all_rows:
            raise ValueError("CSV file is empty")

        # Find the header row (skip metadata rows in client format)
        header_row_idx = 0
        headers = None
        for idx, row in enumerate(all_rows):
            # Check if this row looks like a header (has expected column names)
            if any(h in self.HEADER_TO_FIELD for h in row):
                header_row_idx = idx
                headers = row
                break

        if headers is None:
            # Try first row as headers (legacy format)
            headers = all_rows[0]
            header_row_idx = 0

        # Create a dict reader starting from data rows
        data_rows = all_rows[header_row_idx + 1:]

        # Filter out footer rows (Notes, empty rows, etc.)
        data_rows = [row for row in data_rows if row and not self._is_footer_row(row)]

        # Validate headers
        unknown_headers = set(headers) - set(self.HEADER_TO_FIELD.keys()) - {"", "Notes:", "Notes"}
        if unknown_headers:
            logger.warning(f"Unknown CSV headers will be ignored: {unknown_headers}")

        # Process data rows
        for row_num, row_values in enumerate(data_rows, start=header_row_idx + 2):
            # Create dict from row values
            row_dict = {}
            for i, value in enumerate(row_values):
                if i < len(headers):
                    row_dict[headers[i]] = value

            vehicle, errors = self._parse_row(row_dict, row_num)
            if vehicle:  # Only add if we got valid data
                vehicles.append(vehicle)
            validation_errors.extend(errors)

        # Detect changes if original data provided
        changes = []
        if original_vehicles:
            changes = self._detect_changes(original_vehicles, vehicles)

        logger.info(
            f"Imported {len(vehicles)} vehicles, "
            f"{len(changes)} changes detected, "
            f"{len(validation_errors)} validation issues"
        )

        return vehicles, changes, validation_errors

    def _is_footer_row(self, row: List[str]) -> bool:
        """Check if a row is a footer row (notes, empty, etc.)."""
        if not row:
            return True
        first_cell = row[0].strip().lower() if row[0] else ""
        # Skip notes and empty rows
        if first_cell in ("", "notes:", "notes", "-"):
            return True
        if first_cell.startswith("-"):
            return True
        return False

    def _parse_row(
        self,
        row: Dict[str, str],
        row_num: int
    ) -> Tuple[Dict[str, Any], List[ValidationError]]:
        """
        Parse a single CSV row into a vehicle dictionary.

        Handles both legacy format and client-friendly combined fields.

        Args:
            row: CSV row as dictionary
            row_num: Row number for error reporting

        Returns:
            Tuple of (vehicle_dict, validation_errors)
        """
        vehicle = {}
        errors = []

        for header, value in row.items():
            if header not in self.HEADER_TO_FIELD:
                continue

            field = self.HEADER_TO_FIELD[header]

            # Skip row_id - it's for tracking only
            if field == "row_id":
                try:
                    vehicle["row_id"] = int(value)
                except (ValueError, TypeError):
                    vehicle["row_id"] = row_num - 1
                continue

            # Handle combined fields (e.g., "400-560 kWh")
            if field in self.COMBINED_FIELDS:
                min_field, max_field, unit = self.COMBINED_FIELDS[field]
                min_val, max_val = self._parse_combined_value(value)
                if min_val is not None:
                    vehicle[min_field] = min_val
                if max_val is not None:
                    vehicle[max_field] = max_val
                continue

            # Parse value based on field type
            parsed_value, error = self._parse_value(field, value, row_num)

            if error:
                errors.append(error)

            if parsed_value is not None:
                vehicle[field] = parsed_value
                # Validate range
                range_error = self._validate_range(field, parsed_value, row_num)
                if range_error:
                    errors.append(range_error)

        return vehicle, errors

    def _parse_combined_value(self, value: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse a combined range value like "400-560 kWh" into min and max.

        Returns:
            Tuple of (min_value, max_value)
        """
        if not value or value.strip() in ("-", "N/A", "n/a", ""):
            return None, None

        # Remove unit and commas
        value = value.strip()
        for unit in ("kWh", "kW", "km", "V", "kg", "Nm", "min"):
            value = value.replace(unit, "")
        value = value.replace(",", "").strip()

        # Check for range format "min-max"
        if "-" in value and not value.startswith("-"):
            parts = value.split("-")
            if len(parts) == 2:
                try:
                    min_val = float(parts[0].strip())
                    max_val = float(parts[1].strip())
                    return min_val, max_val
                except ValueError:
                    pass

        # Single value
        try:
            val = float(value)
            return None, val  # Single value goes to max field
        except ValueError:
            return None, None

    def _parse_value(
        self,
        field: str,
        value: str,
        row_num: int
    ) -> Tuple[Any, Optional[ValidationError]]:
        """
        Parse a string value into the appropriate type.

        Args:
            field: Field name
            value: String value from CSV
            row_num: Row number for error reporting

        Returns:
            Tuple of (parsed_value, error_or_none)
        """
        # Handle empty values
        value = value.strip() if value else ""
        if not value or value.lower() in ("", "n/a", "na", "-", "none"):
            return None, None

        # Handle list fields
        if field == "available_configurations":
            configs = [c.strip() for c in value.split(",") if c.strip()]
            return configs if configs else None, None

        # Handle percentage fields
        if field == "data_completeness_score":
            try:
                # Handle both "85%" and "0.85" formats
                if value.endswith("%"):
                    return float(value.rstrip("%")) / 100, None
                val = float(value)
                # If value > 1, assume it's a percentage
                if val > 1:
                    return val / 100, None
                return val, None
            except ValueError:
                return None, ValidationError(
                    row_id=row_num,
                    field=field,
                    value=value,
                    error=f"Invalid percentage: {value}",
                    severity="warning"
                )

        # Handle numeric fields
        if field in self.NUMERIC_FIELDS:
            # Clean up value: remove units, commas, extra whitespace
            clean_value = value
            for unit in ("kWh", "kW", "km", "V", "kg", "Nm", "min", "t"):
                clean_value = clean_value.replace(unit, "")
            clean_value = clean_value.replace(",", "").strip()

            try:
                return float(clean_value), None
            except ValueError:
                return None, ValidationError(
                    row_id=row_num,
                    field=field,
                    value=value,
                    error=f"Invalid number: {value}",
                    severity="error"
                )

        # String fields
        return value, None

    def _validate_range(
        self,
        field: str,
        value: Any,
        row_num: int
    ) -> Optional[ValidationError]:
        """
        Validate that a numeric value is within expected range.

        Args:
            field: Field name
            value: Parsed numeric value
            row_num: Row number for error reporting

        Returns:
            ValidationError if out of range, None otherwise
        """
        if field not in self.VALIDATION_RANGES:
            return None

        if value is None:
            return None

        try:
            num_value = float(value)
            min_val, max_val = self.VALIDATION_RANGES[field]

            if num_value < min_val or num_value > max_val:
                return ValidationError(
                    row_id=row_num,
                    field=field,
                    value=value,
                    error=f"Value {num_value} outside expected range [{min_val}, {max_val}]",
                    severity="warning"
                )
        except (ValueError, TypeError):
            pass

        return None

    def _detect_changes(
        self,
        original_vehicles: List[Dict[str, Any]],
        imported_vehicles: List[Dict[str, Any]]
    ) -> List[FieldChange]:
        """
        Detect changes between original and imported data.

        Args:
            original_vehicles: Original vehicle data
            imported_vehicles: Imported vehicle data

        Returns:
            List of FieldChange objects describing modifications
        """
        changes = []

        # Create lookup by row_id
        original_by_id = {}
        for idx, vehicle in enumerate(original_vehicles, start=1):
            original_by_id[vehicle.get("row_id", idx)] = vehicle

        for imported in imported_vehicles:
            row_id = imported.get("row_id", 0)
            original = original_by_id.get(row_id)

            if not original:
                # New vehicle added (rare case)
                changes.append(FieldChange(
                    row_id=row_id,
                    vehicle_name=imported.get("vehicle_name", "Unknown"),
                    field="vehicle",
                    field_display_name="Vehicle",
                    original_value=None,
                    new_value=imported.get("vehicle_name"),
                    change_type="added"
                ))
                continue

            vehicle_name = imported.get("vehicle_name") or original.get("vehicle_name", "Unknown")

            # Compare fields
            all_fields = set(original.keys()) | set(imported.keys())
            for field in all_fields:
                if field in ("row_id", "additional_specs", "raw_table_data"):
                    continue

                original_value = original.get(field)
                imported_value = imported.get(field)

                # Normalize for comparison
                if self._values_differ(original_value, imported_value):
                    # Get display name
                    display_name = self._get_display_name(field)

                    # Determine change type
                    if original_value is None and imported_value is not None:
                        change_type = "added"
                    elif original_value is not None and imported_value is None:
                        change_type = "removed"
                    else:
                        change_type = "modified"

                    changes.append(FieldChange(
                        row_id=row_id,
                        vehicle_name=vehicle_name,
                        field=field,
                        field_display_name=display_name,
                        original_value=original_value,
                        new_value=imported_value,
                        change_type=change_type
                    ))

        return changes

    def _values_differ(self, val1: Any, val2: Any) -> bool:
        """Check if two values are different (with tolerance for floats)."""
        if val1 is None and val2 is None:
            return False
        if val1 is None or val2 is None:
            return True

        # Handle float comparison with tolerance
        try:
            f1, f2 = float(val1), float(val2)
            return abs(f1 - f2) > 0.01
        except (ValueError, TypeError):
            pass

        return str(val1) != str(val2)

    def _get_display_name(self, field: str) -> str:
        """Get human-readable display name for a field."""
        # Reverse lookup
        for header, internal_field in self.HEADER_TO_FIELD.items():
            if internal_field == field:
                return header
        return field.replace("_", " ").title()

    def validate_csv_structure(self, filepath: str) -> Tuple[bool, List[str]]:
        """
        Validate CSV file structure without fully parsing.

        Args:
            filepath: Path to the CSV file

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            path = Path(filepath)
            if not path.exists():
                return False, ["File does not exist"]

            if path.suffix.lower() != ".csv":
                issues.append("File does not have .csv extension")

            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                headers = next(reader, None)

                if not headers:
                    return False, ["File is empty or has no headers"]

                # Check for required columns
                required = {"Row ID", "Vehicle Name", "OEM"}
                found = set(headers)
                missing = required - found
                if missing:
                    issues.append(f"Missing required columns: {missing}")

                # Check row count
                row_count = sum(1 for _ in reader)
                if row_count == 0:
                    issues.append("No data rows found")

        except csv.Error as e:
            return False, [f"CSV parsing error: {str(e)}"]
        except UnicodeDecodeError as e:
            return False, [f"Encoding error (try saving as UTF-8): {str(e)}"]
        except Exception as e:
            return False, [f"Error reading file: {str(e)}"]

        return len(issues) == 0, issues

    def get_change_summary(self, changes: List[FieldChange]) -> Dict[str, Any]:
        """
        Generate a summary of changes for display.

        Args:
            changes: List of FieldChange objects

        Returns:
            Summary dictionary
        """
        if not changes:
            return {
                "total_changes": 0,
                "vehicles_modified": 0,
                "fields_modified": 0,
                "by_type": {},
                "by_vehicle": {},
            }

        by_type = {"modified": 0, "added": 0, "removed": 0}
        by_vehicle = {}

        for change in changes:
            by_type[change.change_type] = by_type.get(change.change_type, 0) + 1

            if change.vehicle_name not in by_vehicle:
                by_vehicle[change.vehicle_name] = []
            by_vehicle[change.vehicle_name].append({
                "field": change.field_display_name,
                "original": change.original_value,
                "new": change.new_value,
                "type": change.change_type,
            })

        return {
            "total_changes": len(changes),
            "vehicles_modified": len(by_vehicle),
            "fields_modified": len(set(c.field for c in changes)),
            "by_type": by_type,
            "by_vehicle": by_vehicle,
        }


# Module-level instance
_default_service: Optional[CSVImportService] = None


def get_csv_import_service() -> CSVImportService:
    """Get the default CSV import service instance."""
    global _default_service
    if _default_service is None:
        _default_service = CSVImportService()
    return _default_service


def import_csv_with_validation(
    filepath: str,
    original_vehicles: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], List[FieldChange], List[ValidationError]]:
    """
    Convenience function to import CSV with validation.

    Args:
        filepath: Path to CSV file
        original_vehicles: Original data for change detection

    Returns:
        Tuple of (vehicles, changes, errors)
    """
    service = get_csv_import_service()
    return service.import_csv(filepath, original_vehicles)


if __name__ == "__main__":
    # Test the import service
    print("=" * 60)
    print("CSV IMPORT SERVICE TEST")
    print("=" * 60)

    # First, create a test CSV using the export service
    from csv_export_service import CSVExportService

    test_vehicles = [
        {
            "vehicle_name": "MAN eTGX 4x2",
            "oem_name": "MAN",
            "battery_capacity_kwh": 480,
            "motor_power_kw": 400,
            "range_km": 500,
        },
        {
            "vehicle_name": "MAN eTGS 6x2",
            "oem_name": "MAN",
            "battery_capacity_kwh": 560,
            "motor_power_kw": 320,
            "range_km": 700,
        },
    ]

    # Export
    export_service = CSVExportService()
    filepath, _ = export_service.export_vehicles(test_vehicles, "import_test")
    print(f"\nCreated test CSV: {filepath}")

    # Import
    import_service = CSVImportService()
    imported, changes, errors = import_service.import_csv(filepath, test_vehicles)

    print(f"\nImported {len(imported)} vehicles")
    print(f"Changes detected: {len(changes)}")
    print(f"Validation errors: {len(errors)}")

    if errors:
        print("\nValidation issues:")
        for err in errors:
            print(f"  Row {err.row_id}, {err.field}: {err.error}")

    # Test structure validation
    valid, issues = import_service.validate_csv_structure(filepath)
    print(f"\nStructure valid: {valid}")
    if issues:
        print(f"Issues: {issues}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
