#!/usr/bin/env python3
r"""
Volvo Trucks E2E Extraction Test
================================

Run: venv\Scripts\python.exe run_volvo_extraction.py
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    print("=" * 70)
    print("VOLVO TRUCKS - E2E EXTRACTION TEST")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Volvo URL from urls.txt
    volvo_url = "https://www.volvotrucks.com/en-en/trucks/electric.html"

    print(f"URL: {volvo_url}")
    print("Mode: CRAWL4AI + OpenAI (Intelligent Mode)")
    print()

    try:
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print("Starting complete workflow...")
        print("-" * 50)
        print()

        start_time = time.time()

        # Run the complete workflow
        final_state = run_benchmark(
            urls=[volvo_url],
            verbose=True,
            mode=ScrapingMode.INTELLIGENT
        )

        duration = time.time() - start_time

        print()
        print("=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)

        # Results summary
        status = final_state.get("workflow_status")
        vehicles = final_state.get("all_vehicles", [])
        validation = final_state.get("quality_validation", {})
        presentation = final_state.get("presentation_result", {})

        print(f"\nStatus: {status}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Vehicles extracted: {len(vehicles)}")
        print(f"Quality score: {validation.get('overall_quality_score', 0):.2%}")

        if vehicles:
            print("\n" + "-" * 50)
            print("EXTRACTED VEHICLES:")
            print("-" * 50)
            for v in vehicles:
                name = v.get('vehicle_name', 'Unknown')
                battery = v.get('battery_capacity_kwh', '-')
                battery_min = v.get('battery_capacity_min_kwh')
                range_km = v.get('range_km', '-')
                motor = v.get('motor_power_kw', '-')
                completeness = v.get('data_completeness_score', 0)

                battery_str = f"{battery_min}-{battery}" if battery_min and battery_min != battery else str(battery)

                print(f"\n  {name}")
                print(f"    Battery: {battery_str} kWh")
                print(f"    Range: {range_km} km")
                print(f"    Motor Power: {motor} kW")
                print(f"    Completeness: {completeness:.0%}")

        # Check presentation
        pres_paths = presentation.get("all_presentation_paths", [])
        if pres_paths:
            print("\n" + "-" * 50)
            print("GENERATED PRESENTATIONS:")
            print("-" * 50)
            for path in pres_paths:
                if Path(path).exists():
                    size = Path(path).stat().st_size / 1024
                    print(f"  {Path(path).name} ({size:.1f} KB)")

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
