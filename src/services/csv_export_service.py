"""
CSV Export Service for E-Powertrain Benchmarking System

Exports vehicle data to CSV format for human review in Excel.
Generates structured, client-friendly CSVs with clear formatting.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CSVExportService:
    """
    Service for exporting vehicle data to CSV format.

    Features:
    - Client-friendly column headers and formatting
    - Combined range values (e.g., "400-560 kWh" instead of separate columns)
    - Row IDs for tracking edits
    - Professional filename format
    - Summary header with metadata
    """

    # Client-friendly column definitions for the simplified export
    # Format: (header_name, extraction_function)
    CLIENT_COLUMNS = [
        ("No.", lambda v, i: i),
        ("OEM", lambda v, i: v.get("oem_name") or "-"),
        ("Vehicle Model", lambda v, i: v.get("vehicle_name") or "-"),
        ("Vehicle Category", lambda v, i: v.get("category") or "-"),
        ("Battery Capacity", lambda v, i: _format_range_with_unit(
            v.get("battery_capacity_min_kwh"),
            v.get("battery_capacity_kwh"),
            "kWh"
        )),
        ("Battery Voltage", lambda v, i: _format_range_with_unit(
            v.get("battery_voltage_min_v"),
            v.get("battery_voltage_v"),
            "V"
        )),
        ("Motor Power", lambda v, i: _format_range_with_unit(
            v.get("motor_power_min_kw"),
            v.get("motor_power_kw"),
            "kW"
        )),
        ("Range", lambda v, i: _format_range_with_unit(
            v.get("range_min_km"),
            v.get("range_km"),
            "km"
        )),
        ("DC Fast Charging", lambda v, i: _format_range_with_unit(
            v.get("dc_charging_min_kw"),
            v.get("dc_charging_kw"),
            "kW"
        )),
        ("MCS Charging", lambda v, i: _format_range_with_unit(
            v.get("mcs_charging_min_kw"),
            v.get("mcs_charging_kw"),
            "kW"
        )),
        ("Gross Vehicle Weight (GVW)", lambda v, i: _format_weight(v.get("gvw_kg"))),
        ("Gross Combination Weight (GCW)", lambda v, i: _format_weight(v.get("gcw_kg"))),
        ("Payload Capacity", lambda v, i: _format_weight(v.get("payload_capacity_kg"))),
        ("Configurations", lambda v, i: ", ".join(v.get("available_configurations", [])) if v.get("available_configurations") else "-"),
        ("Data Quality", lambda v, i: _format_percentage(v.get("data_completeness_score"))),
        ("Source", lambda v, i: v.get("source_url") or "-"),
    ]

    # Legacy column mapping for backward compatibility
    COLUMN_MAPPING = {
        "row_id": ("Row ID", lambda x: x),
        "vehicle_name": ("Vehicle Name", lambda x: x or ""),
        "oem_name": ("OEM", lambda x: x or ""),
        "category": ("Category", lambda x: x or ""),
        "powertrain_type": ("Powertrain Type", lambda x: x or ""),
        "battery_capacity_kwh": ("Battery (kWh)", lambda x: _format_number(x)),
        "battery_capacity_min_kwh": ("Battery Min (kWh)", lambda x: _format_number(x)),
        "battery_voltage_v": ("Battery Voltage (V)", lambda x: _format_number(x)),
        "battery_voltage_min_v": ("Battery Voltage Min (V)", lambda x: _format_number(x)),
        "battery_chemistry": ("Battery Chemistry", lambda x: x or ""),
        "motor_power_kw": ("Motor Power (kW)", lambda x: _format_number(x)),
        "motor_power_min_kw": ("Motor Power Min (kW)", lambda x: _format_number(x)),
        "motor_torque_nm": ("Motor Torque (Nm)", lambda x: _format_number(x)),
        "motor_torque_min_nm": ("Motor Torque Min (Nm)", lambda x: _format_number(x)),
        "range_km": ("Range (km)", lambda x: _format_number(x)),
        "range_min_km": ("Range Min (km)", lambda x: _format_number(x)),
        "energy_consumption_kwh_per_100km": ("Energy Consumption (kWh/100km)", lambda x: _format_number(x)),
        "dc_charging_kw": ("DC Charging (kW)", lambda x: _format_number(x)),
        "dc_charging_min_kw": ("DC Charging Min (kW)", lambda x: _format_number(x)),
        "mcs_charging_kw": ("MCS Charging (kW)", lambda x: _format_number(x)),
        "mcs_charging_min_kw": ("MCS Charging Min (kW)", lambda x: _format_number(x)),
        "charging_time_minutes": ("Charging Time (min)", lambda x: _format_number(x)),
        "charging_time_max_minutes": ("Charging Time Max (min)", lambda x: _format_number(x)),
        "gvw_kg": ("GVW (kg)", lambda x: _format_number(x)),
        "gvw_min_kg": ("GVW Min (kg)", lambda x: _format_number(x)),
        "gcw_kg": ("GCW (kg)", lambda x: _format_number(x)),
        "gcw_min_kg": ("GCW Min (kg)", lambda x: _format_number(x)),
        "payload_capacity_kg": ("Payload (kg)", lambda x: _format_number(x)),
        "payload_capacity_min_kg": ("Payload Min (kg)", lambda x: _format_number(x)),
        "available_configurations": ("Configurations", lambda x: ", ".join(x) if x else ""),
        "source_url": ("Source URL", lambda x: x or ""),
        "data_completeness_score": ("Completeness Score", lambda x: _format_percentage(x)),
        "extraction_timestamp": ("Extracted At", lambda x: x or ""),
    }

    # Default columns for export (subset of all columns for cleaner view)
    DEFAULT_EXPORT_COLUMNS = [
        "row_id",
        "vehicle_name",
        "oem_name",
        "category",
        "battery_capacity_kwh",
        "battery_capacity_min_kwh",
        "motor_power_kw",
        "motor_power_min_kw",
        "range_km",
        "range_min_km",
        "dc_charging_kw",
        "dc_charging_min_kw",
        "gvw_kg",
        "gcw_kg",
        "payload_capacity_kg",
        "source_url",
        "data_completeness_score",
    ]

    # Full export includes all columns
    FULL_EXPORT_COLUMNS = list(COLUMN_MAPPING.keys())

    def __init__(self, output_dir: str = "outputs/reviews"):
        """
        Initialize CSV export service.

        Args:
            output_dir: Directory for CSV output files
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CSV export directory: {self.output_dir.absolute()}")

    def export_vehicles(
        self,
        vehicles: List[Dict[str, Any]],
        thread_id: str,
        full_export: bool = False,
        custom_columns: Optional[List[str]] = None,
        client_format: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Export vehicles to CSV file.

        Args:
            vehicles: List of vehicle dictionaries
            thread_id: Workflow thread ID for file naming
            full_export: If True, include all columns; otherwise use default subset
            custom_columns: Optional list of specific columns to export
            client_format: If True, use client-friendly format with combined columns

        Returns:
            Tuple of (csv_file_path, export_metadata)
        """
        if not vehicles:
            raise ValueError("No vehicles to export")

        # Generate client-friendly filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H%M")

        # Get OEM names for filename
        oem_names = list(set(v.get("oem_name", "Unknown") for v in vehicles if v.get("oem_name")))
        if len(oem_names) == 1:
            oem_part = oem_names[0].replace(" ", "_")
        elif len(oem_names) <= 3:
            oem_part = "_".join(o.replace(" ", "") for o in oem_names[:3])
        else:
            oem_part = f"{len(oem_names)}_OEMs"

        filename = f"EV_Specs_{oem_part}_{date_str}_{time_str}.csv"
        filepath = self.output_dir / filename

        # Use client-friendly format by default
        if client_format and not custom_columns and not full_export:
            rows_written = self._write_client_csv(filepath, vehicles)
            column_names = [col[0] for col in self.CLIENT_COLUMNS]
        else:
            # Legacy format
            if custom_columns:
                columns = ["row_id"] + [c for c in custom_columns if c != "row_id"]
            elif full_export:
                columns = self.FULL_EXPORT_COLUMNS
            else:
                columns = self.DEFAULT_EXPORT_COLUMNS

            columns = [c for c in columns if c in self.COLUMN_MAPPING]
            vehicles_with_ids = self._add_row_ids(vehicles)
            rows_written = self._write_csv(filepath, vehicles_with_ids, columns)
            column_names = [self.COLUMN_MAPPING[c][0] for c in columns]

        # Create metadata
        metadata = {
            "file_path": str(filepath.absolute()),
            "filename": filename,
            "thread_id": thread_id,
            "vehicle_count": len(vehicles),
            "rows_written": rows_written,
            "columns_exported": len(column_names),
            "column_names": column_names,
            "export_timestamp": datetime.now().isoformat(),
            "full_export": full_export,
            "client_format": client_format,
        }

        logger.info(f"Exported {rows_written} vehicles to {filepath}")
        return str(filepath.absolute()), metadata

    def _write_client_csv(
        self,
        filepath: Path,
        vehicles: List[Dict[str, Any]]
    ) -> int:
        """
        Write vehicles to CSV in client-friendly format.

        Features:
        - Summary header with metadata
        - Combined range columns (e.g., "400-560 kWh")
        - Professional formatting
        """
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # Write summary header
            writer.writerow(["E-POWERTRAIN VEHICLE SPECIFICATIONS"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"])
            writer.writerow([f"Total Vehicles: {len(vehicles)}"])

            # Get unique OEMs
            oem_names = list(set(v.get("oem_name", "Unknown") for v in vehicles if v.get("oem_name")))
            writer.writerow([f"OEMs: {', '.join(oem_names)}"])

            # Empty row before data
            writer.writerow([])

            # Write column headers
            headers = [col[0] for col in self.CLIENT_COLUMNS]
            writer.writerow(headers)

            # Write data rows
            rows_written = 0
            for idx, vehicle in enumerate(vehicles, start=1):
                row = []
                for header, extractor in self.CLIENT_COLUMNS:
                    try:
                        value = extractor(vehicle, idx)
                        row.append(value)
                    except Exception:
                        row.append("-")
                writer.writerow(row)
                rows_written += 1

            # Footer
            writer.writerow([])
            writer.writerow(["Notes:"])
            writer.writerow(["- Values shown as ranges indicate min-max specifications"])
            writer.writerow(["- 'Data Quality' indicates completeness of extracted data"])
            writer.writerow(["- Edit values as needed, then upload for review"])

        return rows_written

    def _add_row_ids(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add sequential row IDs to vehicles for change tracking."""
        result = []
        for idx, vehicle in enumerate(vehicles, start=1):
            vehicle_copy = dict(vehicle)
            vehicle_copy["row_id"] = idx
            result.append(vehicle_copy)
        return result

    def _write_csv(
        self,
        filepath: Path,
        vehicles: List[Dict[str, Any]],
        columns: List[str]
    ) -> int:
        """
        Write vehicles to CSV file.

        Args:
            filepath: Output file path
            vehicles: List of vehicle dictionaries with row IDs
            columns: List of column keys to export

        Returns:
            Number of rows written
        """
        # Get headers
        headers = [self.COLUMN_MAPPING[col][0] for col in columns]

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # Write header row
            writer.writerow(headers)

            # Write data rows
            rows_written = 0
            for vehicle in vehicles:
                row = []
                for col in columns:
                    value = vehicle.get(col)
                    formatter = self.COLUMN_MAPPING[col][1]
                    formatted_value = formatter(value)
                    row.append(formatted_value)
                writer.writerow(row)
                rows_written += 1

        return rows_written

    def get_export_preview(
        self,
        vehicles: List[Dict[str, Any]],
        max_rows: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get a preview of the data that would be exported.

        Args:
            vehicles: List of vehicle dictionaries
            max_rows: Maximum rows to include in preview

        Returns:
            List of dictionaries with formatted values for preview
        """
        preview = []
        vehicles_with_ids = self._add_row_ids(vehicles[:max_rows])

        for vehicle in vehicles_with_ids:
            row = {}
            for col in self.DEFAULT_EXPORT_COLUMNS:
                header = self.COLUMN_MAPPING[col][0]
                value = vehicle.get(col)
                formatter = self.COLUMN_MAPPING[col][1]
                row[header] = formatter(value)
            preview.append(row)

        return preview

    def list_exports(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List existing CSV exports.

        Args:
            thread_id: Optional filter by thread ID

        Returns:
            List of export file metadata
        """
        exports = []
        pattern = f"review_{thread_id}_*.csv" if thread_id else "review_*.csv"

        for filepath in self.output_dir.glob(pattern):
            stat = filepath.stat()
            exports.append({
                "filename": filepath.name,
                "path": str(filepath.absolute()),
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        # Sort by creation time, newest first
        exports.sort(key=lambda x: x["created_at"], reverse=True)
        return exports

    def delete_export(self, filepath: str) -> bool:
        """
        Delete an export file.

        Args:
            filepath: Path to the CSV file to delete

        Returns:
            True if deleted, False if file not found
        """
        path = Path(filepath)
        if path.exists() and path.is_file():
            path.unlink()
            logger.info(f"Deleted export: {filepath}")
            return True
        return False


# Helper functions for formatting
def _format_number(value: Any) -> str:
    """Format numeric value for CSV export."""
    if value is None:
        return ""
    try:
        num = float(value)
        # Use integer format if whole number
        if num == int(num):
            return str(int(num))
        return f"{num:.1f}"
    except (ValueError, TypeError):
        return str(value)


def _format_percentage(value: Any) -> str:
    """Format percentage value for CSV export."""
    if value is None:
        return "-"
    try:
        num = float(value)
        return f"{num * 100:.0f}%"
    except (ValueError, TypeError):
        return str(value)


def _format_range_with_unit(min_val: Any, max_val: Any, unit: str) -> str:
    """
    Format a min-max range with unit.

    Examples:
        - (400, 560, "kWh") -> "400-560 kWh"
        - (None, 500, "km") -> "500 km"
        - (None, None, "kW") -> "-"
    """
    min_num = _parse_number(min_val)
    max_num = _parse_number(max_val)

    if min_num is None and max_num is None:
        return "-"

    if min_num is not None and max_num is not None:
        if min_num == max_num:
            return f"{_format_int(max_num)} {unit}"
        return f"{_format_int(min_num)}-{_format_int(max_num)} {unit}"

    if max_num is not None:
        return f"{_format_int(max_num)} {unit}"

    if min_num is not None:
        return f"{_format_int(min_num)}+ {unit}"

    return "-"


def _format_weight(value: Any) -> str:
    """Format weight value with thousands separator."""
    num = _parse_number(value)
    if num is None:
        return "-"
    # Format with thousands separator
    return f"{int(num):,} kg"


def _parse_number(value: Any) -> Optional[float]:
    """Parse a value to float, returning None if not valid."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _format_int(value: float) -> str:
    """Format a number as integer with thousands separator if large."""
    int_val = int(value)
    if int_val >= 1000:
        return f"{int_val:,}"
    return str(int_val)


# Module-level instance for convenience
_default_service: Optional[CSVExportService] = None


def get_csv_export_service() -> CSVExportService:
    """Get the default CSV export service instance."""
    global _default_service
    if _default_service is None:
        _default_service = CSVExportService()
    return _default_service


def export_vehicles_to_csv(
    vehicles: List[Dict[str, Any]],
    thread_id: str,
    full_export: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to export vehicles to CSV.

    Args:
        vehicles: List of vehicle dictionaries
        thread_id: Workflow thread ID
        full_export: If True, include all columns

    Returns:
        Tuple of (csv_file_path, export_metadata)
    """
    service = get_csv_export_service()
    return service.export_vehicles(vehicles, thread_id, full_export)


if __name__ == "__main__":
    # Test the export service
    test_vehicles = [
        {
            "vehicle_name": "MAN eTGX 4x2",
            "oem_name": "MAN",
            "category": "Long-haul Truck",
            "battery_capacity_kwh": 480,
            "motor_power_kw": 400,
            "range_km": 500,
            "dc_charging_kw": 375,
            "gvw_kg": 18000,
            "source_url": "https://www.man.eu/trucks",
            "data_completeness_score": 0.85,
        },
        {
            "vehicle_name": "MAN eTGS 6x2",
            "oem_name": "MAN",
            "category": "Distribution Truck",
            "battery_capacity_kwh": 560,
            "battery_capacity_min_kwh": 400,
            "motor_power_kw": 320,
            "range_km": 700,
            "range_min_km": 400,
            "dc_charging_kw": 375,
            "gvw_kg": 26000,
            "source_url": "https://www.man.eu/trucks",
            "data_completeness_score": 0.92,
        },
    ]

    print("=" * 60)
    print("CSV EXPORT SERVICE TEST")
    print("=" * 60)

    service = CSVExportService()

    # Test export
    filepath, metadata = service.export_vehicles(test_vehicles, "test_thread_001")

    print(f"\nExported to: {filepath}")
    print(f"Metadata: {metadata}")

    # Test preview
    preview = service.get_export_preview(test_vehicles)
    print(f"\nPreview ({len(preview)} rows):")
    for row in preview:
        print(f"  {row['Vehicle Name']}: Battery={row['Battery (kWh)']} kWh, Range={row['Range (km)']} km")

    # Test list exports
    exports = service.list_exports()
    print(f"\nExisting exports: {len(exports)}")
    for exp in exports[:3]:
        print(f"  {exp['filename']}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
