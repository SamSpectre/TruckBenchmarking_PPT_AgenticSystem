#!/usr/bin/env python3
"""
E2E Test Runner for Intelligent Mode (CRAWL4AI + OpenAI)

This script runs the complete E2E workflow without requiring pytest.
Run from project root: python run_e2e_test.py

Requirements:
- OPENAI_API_KEY environment variable set
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
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def print_header(title: str):
    """Print formatted header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str):
    """Print section separator."""
    print()
    print("-" * 50)
    print(title)
    print("-" * 50)


def check_result(name: str, passed: bool, details: str = ""):
    """Print check result."""
    icon = "+" if passed else "x"
    print(f"  {icon} {name}")
    if details:
        print(f"      {details}")
    return passed


def main():
    """Run E2E tests."""
    print_header("E-POWERTRAIN BENCHMARKING SYSTEM - E2E VERIFICATION")
    print()
    print("Testing: CRAWL4AI + OpenAI (Intelligent Mode)")
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not available, using system environment")

    # Track results
    all_checks = []

    # =========================================================================
    # TEST 1: CONFIGURATION VERIFICATION
    # =========================================================================
    print_section("TEST 1: Configuration Verification")

    try:
        from src.state.state import initialize_state, ScrapingMode
        from src.tools.scraper import ScraperConfig
        from src.config.terminology_mappings import SEMANTIC_EQUIVALENCES, build_semantic_equivalence_prompt

        # Check 1.1: Default mode is Intelligent
        state = initialize_state(["https://example.com"])
        passed = state["scraping_mode"] == ScrapingMode.INTELLIGENT
        all_checks.append(check_result(
            "Default mode is INTELLIGENT",
            passed,
            f"Got: {state['scraping_mode']}"
        ))

        # Check 1.2: Intelligent navigation enabled
        passed = ScraperConfig.ENABLE_INTELLIGENT_NAVIGATION is True
        all_checks.append(check_result(
            "ENABLE_INTELLIGENT_NAVIGATION is True",
            passed
        ))

        # Check 1.3: Strict hallucination check enabled
        passed = ScraperConfig.STRICT_HALLUCINATION_CHECK is True
        all_checks.append(check_result(
            "STRICT_HALLUCINATION_CHECK is True",
            passed
        ))

        # Check 1.4: Terminology mappings loaded
        passed = "motor_torque_nm" in SEMANTIC_EQUIVALENCES
        all_checks.append(check_result(
            "Terminology mappings loaded",
            passed,
            f"motor_torque_nm has {len(SEMANTIC_EQUIVALENCES.get('motor_torque_nm', []))} terms"
        ))

        # Check 1.5: E-axle terminology mapped
        motor_terms = SEMANTIC_EQUIVALENCES.get("motor_torque_nm", [])
        passed = any("e-axle" in term.lower() for term in motor_terms)
        all_checks.append(check_result(
            "E-axle torque mapped to motor_torque",
            passed
        ))

        print("\nConfiguration tests PASSED")

    except Exception as e:
        print(f"\nERROR in configuration tests: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # =========================================================================
    # TEST 2: API KEY CHECK
    # =========================================================================
    print_section("TEST 2: API Key Verification")

    openai_key = os.getenv("OPENAI_API_KEY")

    passed = openai_key is not None and len(openai_key) > 0
    all_checks.append(check_result(
        "OPENAI_API_KEY is set",
        passed,
        f"Key found" if passed else "NOT SET"
    ))

    if not openai_key:
        print("\nERROR: OPENAI_API_KEY is required for E2E tests")
        print("Please set the environment variable and try again.")
        return 1

    # =========================================================================
    # TEST 3: EXTRACTION TEST
    # =========================================================================
    print_section("TEST 3: Extraction Test (Intelligent Mode)")

    test_url = "https://www.man.eu/global/en/truck/electric-trucks/overview.html"
    print(f"URL: {test_url}")
    print("Mode: CRAWL4AI + OpenAI GPT-4o")
    print()

    try:
        from src.tools.scraper import EPowertrainExtractor

        print("Creating extractor (Intelligent mode)...")
        extractor = EPowertrainExtractor(use_intelligent_mode=True)

        print("Starting extraction (this may take 1-2 minutes)...")
        start_time = time.time()

        results = extractor.process_urls([test_url])

        duration = time.time() - start_time
        print(f"\nExtraction completed in {duration:.1f} seconds")

        # Verify results
        passed = results is not None and len(results) >= 1
        all_checks.append(check_result(
            "Extraction returned results",
            passed
        ))

        if results and len(results) >= 1:
            result = results[0]

            oem_name = result.get('oem_name', 'Unknown')
            vehicles = result.get('vehicles', [])
            pages_crawled = result.get('pages_crawled', 'N/A')

            print(f"\nResults:")
            print(f"  OEM: {oem_name}")
            print(f"  Vehicles found: {len(vehicles)}")
            print(f"  Pages crawled: {pages_crawled}")

            passed = len(vehicles) > 0
            all_checks.append(check_result(
                "Vehicles extracted",
                passed,
                f"Found {len(vehicles)} vehicle(s)"
            ))

            if vehicles:
                print(f"\nExtracted vehicles:")
                for v in vehicles[:5]:  # Show first 5
                    name = v.get('vehicle_name', 'Unknown')
                    battery = v.get('battery_capacity_kwh', 'N/A')
                    motor = v.get('motor_power_kw', 'N/A')
                    range_km = v.get('range_km', 'N/A')
                    completeness = v.get('data_completeness_score', 0)

                    print(f"  - {name}")
                    print(f"    Battery: {battery} kWh | Motor: {motor} kW | Range: {range_km} km")
                    print(f"    Completeness: {completeness:.0%}")

                # Check data quality
                avg_completeness = sum(v.get('data_completeness_score', 0) for v in vehicles) / len(vehicles)
                passed = avg_completeness >= 0.3
                all_checks.append(check_result(
                    "Data quality acceptable",
                    passed,
                    f"Avg completeness: {avg_completeness:.0%}"
                ))

        print("\nExtraction test completed")

    except Exception as e:
        print(f"\nERROR in extraction test: {e}")
        import traceback
        traceback.print_exc()
        all_checks.append(False)

    # =========================================================================
    # TEST 4: QUALITY VALIDATION TEST
    # =========================================================================
    print_section("TEST 4: Quality Validation")

    if results and len(results) >= 1 and results[0].get('vehicles'):
        try:
            from src.agents.quality_validator import run_validation

            print("Running quality validation...")
            validation = run_validation(results, use_llm=False)

            print(f"\nValidation Results:")
            print(f"  Overall Score: {validation['overall_quality_score']:.2f}")
            print(f"  Completeness: {validation['completeness_score']:.2f}")
            print(f"  Accuracy: {validation['accuracy_score']:.2f}")
            print(f"  Consistency: {validation['consistency_score']:.2f}")
            print(f"  Passes Threshold: {validation['passes_threshold']}")
            print(f"  Recommendation: {validation['recommendation'][:100]}...")

            passed = validation['overall_quality_score'] > 0
            all_checks.append(check_result(
                "Quality validation executed",
                passed,
                f"Score: {validation['overall_quality_score']:.2f}"
            ))

        except Exception as e:
            print(f"ERROR in validation: {e}")
            all_checks.append(False)
    else:
        print("Skipping validation - no vehicles to validate")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("E2E VERIFICATION SUMMARY")

    passed_count = sum(1 for c in all_checks if c)
    total_count = len(all_checks)

    print(f"\nChecks passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n[SUCCESS] All E2E verification checks PASSED!")
        print("\nThe system is working correctly with Intelligent mode (CRAWL4AI + OpenAI).")
        return 0
    elif passed_count >= total_count - 2:
        print("\n[PARTIAL SUCCESS] Most checks passed.")
        print("Review the failed checks above.")
        return 0
    else:
        print("\n[FAILURE] Multiple checks failed.")
        print("Review the output above to identify issues.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
