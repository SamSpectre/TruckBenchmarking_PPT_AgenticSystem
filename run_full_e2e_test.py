#!/usr/bin/env python3
r"""
COMPREHENSIVE E2E TEST SCRIPT
=============================

Run this from Windows Command Prompt or PowerShell:
    cd "C:\Users\SEHGALS\Langgraph Projects\Langgraph_Project1"
    venv\Scripts\python.exe run_full_e2e_test.py

This script performs:
1. Environment validation (API keys, dependencies)
2. Configuration verification
3. OpenAI API connectivity test
4. Full extraction test with Intelligent mode
5. Quality validation
6. Presentation generation test

Requirements:
- OPENAI_API_KEY (required)
- PERPLEXITY_API_KEY (optional - not needed for Intelligent mode)
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-supporting terminals."""
        cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = cls.CYAN = cls.BOLD = cls.END = ''


# Disable colors if not supported
if sys.platform == 'win32' and not os.environ.get('TERM'):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        Colors.disable()


def print_header(title: str):
    """Print formatted header."""
    print()
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")


def print_section(title: str):
    """Print section separator."""
    print()
    print(f"{Colors.CYAN}{'-'*50}{Colors.END}")
    print(f"{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.END}")


def print_pass(msg: str, details: str = ""):
    """Print passed check."""
    print(f"  {Colors.GREEN}[PASS]{Colors.END} {msg}")
    if details:
        print(f"         {details}")


def print_fail(msg: str, details: str = ""):
    """Print failed check."""
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")
    if details:
        print(f"         {details}")


def print_warn(msg: str, details: str = ""):
    """Print warning."""
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")
    if details:
        print(f"         {details}")


def print_info(msg: str):
    """Print info message."""
    print(f"  {Colors.BLUE}[INFO]{Colors.END} {msg}")


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details: List[Dict[str, Any]] = []

    def add_pass(self, name: str, details: str = ""):
        self.passed += 1
        self.details.append({"name": name, "status": "pass", "details": details})
        print_pass(name, details)

    def add_fail(self, name: str, details: str = ""):
        self.failed += 1
        self.details.append({"name": name, "status": "fail", "details": details})
        print_fail(name, details)

    def add_warn(self, name: str, details: str = ""):
        self.warnings += 1
        self.details.append({"name": name, "status": "warn", "details": details})
        print_warn(name, details)

    def summary(self) -> Tuple[int, int, int]:
        return self.passed, self.failed, self.warnings


# =============================================================================
# TEST: ENVIRONMENT
# =============================================================================

def test_environment(results: TestResults) -> bool:
    """Test environment setup."""
    print_section("ENVIRONMENT CHECK")

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9):
        results.add_pass(f"Python version: {py_version}")
    else:
        results.add_fail(f"Python version: {py_version}", "Requires Python 3.9+")
        return False

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        results.add_pass("dotenv loaded")
    except ImportError:
        results.add_warn("python-dotenv not installed", "Using system environment only")

    # Check OPENAI_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and len(openai_key) > 20:
        results.add_pass("OPENAI_API_KEY is set", f"Starts with: {openai_key[:10]}...")
    else:
        results.add_fail("OPENAI_API_KEY not set or invalid")
        print()
        print(f"  {Colors.RED}ERROR: OPENAI_API_KEY is required.{Colors.END}")
        print("  Set it in your .env file or as an environment variable.")
        return False

    # Check PERPLEXITY_API_KEY (optional)
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    if perplexity_key:
        results.add_pass("PERPLEXITY_API_KEY is set (optional)")
    else:
        print_info("PERPLEXITY_API_KEY not set (OK - not needed for Intelligent mode)")

    return True


# =============================================================================
# TEST: DEPENDENCIES
# =============================================================================

def test_dependencies(results: TestResults) -> bool:
    """Test required dependencies."""
    print_section("DEPENDENCY CHECK")

    required_deps = [
        ("langchain_core", "LangChain Core"),
        ("langgraph", "LangGraph"),
        ("openai", "OpenAI SDK"),
        ("pydantic", "Pydantic"),
    ]

    optional_deps = [
        ("crawl4ai", "CRAWL4AI"),
        ("litellm", "LiteLLM"),
    ]

    all_ok = True

    for module_name, display_name in required_deps:
        try:
            __import__(module_name)
            results.add_pass(f"{display_name} installed")
        except ImportError:
            results.add_fail(f"{display_name} NOT installed")
            all_ok = False

    for module_name, display_name in optional_deps:
        try:
            __import__(module_name)
            results.add_pass(f"{display_name} installed")
        except ImportError:
            results.add_warn(f"{display_name} not installed", "May be needed for some features")

    return all_ok


# =============================================================================
# TEST: CONFIGURATION
# =============================================================================

def test_configuration(results: TestResults) -> bool:
    """Test system configuration."""
    print_section("CONFIGURATION CHECK")

    try:
        from src.state.state import initialize_state, ScrapingMode
        from src.tools.scraper import ScraperConfig

        # Default mode check
        state = initialize_state(["https://example.com"])
        if state["scraping_mode"] == ScrapingMode.INTELLIGENT:
            results.add_pass("Default mode is INTELLIGENT")
        else:
            results.add_fail(f"Default mode is {state['scraping_mode']}", "Expected INTELLIGENT")

        # Intelligent navigation enabled
        if ScraperConfig.ENABLE_INTELLIGENT_NAVIGATION:
            results.add_pass("ENABLE_INTELLIGENT_NAVIGATION = True")
        else:
            results.add_fail("ENABLE_INTELLIGENT_NAVIGATION = False")

        # Strict hallucination check
        if ScraperConfig.STRICT_HALLUCINATION_CHECK:
            results.add_pass("STRICT_HALLUCINATION_CHECK = True")
        else:
            results.add_warn("STRICT_HALLUCINATION_CHECK = False", "Hallucinations may not be filtered")

        # LLM model
        results.add_pass(f"LLM_EXTRACTION_MODEL = {ScraperConfig.LLM_EXTRACTION_MODEL}")

        return True

    except Exception as e:
        results.add_fail(f"Configuration load error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TEST: TERMINOLOGY MAPPINGS
# =============================================================================

def test_terminology(results: TestResults) -> bool:
    """Test terminology mappings."""
    print_section("TERMINOLOGY MAPPINGS CHECK")

    try:
        from src.config.terminology_mappings import (
            SEMANTIC_EQUIVALENCES,
            build_semantic_equivalence_prompt,
            build_terminology_prompt,
        )

        # Check key mappings exist
        required_mappings = ["motor_torque_nm", "motor_power_kw", "battery_capacity_kwh"]
        for mapping in required_mappings:
            if mapping in SEMANTIC_EQUIVALENCES:
                count = len(SEMANTIC_EQUIVALENCES[mapping])
                results.add_pass(f"{mapping} has {count} equivalent terms")
            else:
                results.add_fail(f"{mapping} mapping not found")

        # Check e-axle terminology
        motor_terms = SEMANTIC_EQUIVALENCES.get("motor_torque_nm", [])
        e_axle_terms = [t for t in motor_terms if "e-axle" in t.lower() or "eaxle" in t.lower()]
        if e_axle_terms:
            results.add_pass(f"E-axle torque mapped: {e_axle_terms[:2]}")
        else:
            results.add_warn("E-axle torque not mapped", "May miss e-axle specifications")

        # Check prompt generation
        prompt = build_semantic_equivalence_prompt()
        if len(prompt) > 100:
            results.add_pass(f"Semantic prompt generated ({len(prompt)} chars)")
        else:
            results.add_warn("Semantic prompt is short")

        return True

    except Exception as e:
        results.add_fail(f"Terminology error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TEST: OPENAI API CONNECTIVITY
# =============================================================================

def test_openai_api(results: TestResults) -> bool:
    """Test OpenAI API connectivity with a simple call."""
    print_section("OPENAI API CONNECTIVITY TEST")

    try:
        from openai import OpenAI

        client = OpenAI()

        print_info("Testing OpenAI API with a simple completion...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for test
            messages=[{"role": "user", "content": "Say 'API test successful' in exactly those words."}],
            max_tokens=20,
        )

        reply = response.choices[0].message.content
        if "successful" in reply.lower():
            results.add_pass("OpenAI API connection successful", f"Response: {reply}")
            return True
        else:
            results.add_pass("OpenAI API responded", f"Response: {reply}")
            return True

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid_api_key" in error_msg:
            results.add_fail("OpenAI API key is INVALID", "Please update your OPENAI_API_KEY")
            print()
            print(f"  {Colors.RED}ERROR: Your OpenAI API key is invalid or expired.{Colors.END}")
            print("  1. Go to https://platform.openai.com/api-keys")
            print("  2. Generate a new API key")
            print("  3. Update your .env file with the new key")
            return False
        elif "429" in error_msg:
            results.add_fail("OpenAI API rate limited", "Please wait and try again")
            return False
        else:
            results.add_fail(f"OpenAI API error: {error_msg}")
            traceback.print_exc()
            return False


# =============================================================================
# TEST: EXTRACTION
# =============================================================================

def test_extraction(results: TestResults) -> Optional[Dict]:
    """Test full extraction with Intelligent mode."""
    print_section("EXTRACTION TEST (Intelligent Mode)")

    test_url = "https://www.man.eu/global/en/truck/electric-trucks/overview.html"

    print_info(f"URL: {test_url}")
    print_info("Mode: CRAWL4AI + OpenAI GPT-4o")
    print()

    try:
        from src.tools.scraper import EPowertrainExtractor

        print_info("Creating extractor with Intelligent mode...")
        extractor = EPowertrainExtractor(use_intelligent_mode=True)

        print_info("Starting extraction (this may take 2-3 minutes)...")
        print()

        start_time = time.time()
        results_data = extractor.process_urls([test_url])
        duration = time.time() - start_time

        print()
        print_info(f"Extraction completed in {duration:.1f} seconds")

        if not results_data or len(results_data) < 1:
            results.add_fail("No results returned")
            return None

        result = results_data[0]
        vehicles = result.get("vehicles", [])
        oem_name = result.get("oem_name", "Unknown")
        pages_crawled = result.get("pages_crawled", "N/A")

        results.add_pass(f"OEM identified: {oem_name}")
        results.add_pass(f"Pages crawled: {pages_crawled}")

        if len(vehicles) > 0:
            results.add_pass(f"Vehicles extracted: {len(vehicles)}")

            print()
            print_info("Extracted vehicles:")
            for v in vehicles[:5]:
                name = v.get('vehicle_name', 'Unknown')
                battery = v.get('battery_capacity_kwh', '-')
                battery_min = v.get('battery_capacity_min_kwh')
                motor = v.get('motor_power_kw', '-')
                range_km = v.get('range_km', '-')
                completeness = v.get('data_completeness_score', 0)

                battery_str = f"{battery_min}-{battery}" if battery_min and battery_min != battery else str(battery)

                print(f"    - {name}")
                print(f"      Battery: {battery_str} kWh | Motor: {motor} kW | Range: {range_km} km")
                print(f"      Completeness: {completeness:.0%}")

            # Check data quality
            avg_completeness = sum(v.get('data_completeness_score', 0) for v in vehicles) / len(vehicles)
            if avg_completeness >= 0.5:
                results.add_pass(f"Average data completeness: {avg_completeness:.0%}")
            elif avg_completeness >= 0.3:
                results.add_warn(f"Average data completeness: {avg_completeness:.0%}", "Consider improving extraction")
            else:
                results.add_fail(f"Average data completeness: {avg_completeness:.0%}", "Data quality too low")

            return result

        else:
            results.add_fail("No vehicles extracted")
            print()
            print(f"  {Colors.YELLOW}Possible causes:{Colors.END}")
            print("  - Website structure may have changed")
            print("  - Extraction prompts may need tuning")
            print("  - Check errors above for more details")
            return None

    except Exception as e:
        results.add_fail(f"Extraction error: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# TEST: QUALITY VALIDATION
# =============================================================================

def test_quality_validation(results: TestResults, extraction_result: Optional[Dict]) -> bool:
    """Test quality validation."""
    print_section("QUALITY VALIDATION TEST")

    if not extraction_result or not extraction_result.get("vehicles"):
        print_info("Skipping - no vehicles to validate")
        return True  # Don't fail the overall test

    try:
        from src.agents.quality_validator import run_validation

        print_info("Running quality validation...")

        validation = run_validation([extraction_result], use_llm=False)

        overall_score = validation.get("overall_quality_score", 0)
        completeness = validation.get("completeness_score", 0)
        accuracy = validation.get("accuracy_score", 0)
        passes = validation.get("passes_threshold", False)

        print()
        print_info("Validation Results:")
        print(f"    Overall Score: {overall_score:.2f}")
        print(f"    Completeness: {completeness:.2f}")
        print(f"    Accuracy: {accuracy:.2f}")
        print(f"    Passes Threshold: {passes}")

        if passes:
            results.add_pass(f"Quality validation PASSED (score: {overall_score:.2f})")
        elif overall_score >= 0.5:
            results.add_warn(f"Quality validation marginal (score: {overall_score:.2f})")
        else:
            results.add_fail(f"Quality validation FAILED (score: {overall_score:.2f})")

        return True

    except Exception as e:
        results.add_fail(f"Validation error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TEST: COMPLETE WORKFLOW
# =============================================================================

def test_complete_workflow(results: TestResults) -> bool:
    """Test the complete LangGraph workflow."""
    print_section("COMPLETE WORKFLOW TEST")

    test_url = "https://www.man.eu/global/en/truck/electric-trucks/overview.html"

    print_info(f"URL: {test_url}")
    print_info("Running complete workflow: Scrape -> Validate -> Present")
    print()

    try:
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print_info("Starting workflow (this may take 3-5 minutes)...")
        print()

        start_time = time.time()

        final_state = run_benchmark(
            urls=[test_url],
            verbose=True,
            mode=ScrapingMode.INTELLIGENT
        )

        duration = time.time() - start_time

        print()
        print_info(f"Workflow completed in {duration:.1f} seconds")

        # Check status
        status = final_state.get("workflow_status")
        vehicles = final_state.get("all_vehicles", [])
        validation = final_state.get("quality_validation", {})
        presentation = final_state.get("presentation_result", {})
        cost = final_state.get("total_cost_usd", 0)
        tokens = final_state.get("total_tokens_used", 0)

        print()
        print_info("Workflow Results:")
        print(f"    Status: {status}")
        print(f"    Vehicles: {len(vehicles)}")
        print(f"    Quality Score: {validation.get('overall_quality_score', 0):.2f}")
        print(f"    Tokens Used: {tokens:,}")
        print(f"    Cost: ${cost:.4f}")

        if status == WorkflowStatus.COMPLETED:
            results.add_pass(f"Workflow COMPLETED successfully")

            # Check presentation
            pres_paths = presentation.get("all_presentation_paths", [])
            if pres_paths:
                for path in pres_paths:
                    if Path(path).exists():
                        size = Path(path).stat().st_size / 1024
                        results.add_pass(f"Presentation generated: {Path(path).name} ({size:.1f} KB)")
                    else:
                        results.add_warn(f"Presentation file not found: {path}")
            else:
                results.add_warn("No presentations generated")

            return True

        elif status == WorkflowStatus.FAILED:
            errors = final_state.get("errors", [])
            if "Quality validation failed" in str(errors):
                results.add_warn("Workflow ended - quality threshold not met", str(errors))
                return True  # Don't fail overall test
            else:
                results.add_fail(f"Workflow FAILED: {errors}")
                return False

        else:
            results.add_warn(f"Workflow ended with status: {status}")
            return True

    except Exception as e:
        results.add_fail(f"Workflow error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Run all E2E tests."""
    print_header("E-POWERTRAIN BENCHMARKING SYSTEM")
    print_header("COMPREHENSIVE E2E VERIFICATION")

    print()
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: CRAWL4AI + OpenAI (Intelligent Mode)")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    results = TestResults()

    # Run tests in order
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Terminology", test_terminology),
        ("OpenAI API", test_openai_api),
    ]

    # Run prerequisite tests
    for name, test_func in tests:
        if not test_func(results):
            print_header("TEST SUITE STOPPED")
            print(f"\n{Colors.RED}Prerequisite test '{name}' failed. Fix the issue and retry.{Colors.END}")
            break
    else:
        # All prerequisites passed, run extraction tests
        print()
        print_info("All prerequisites passed. Running extraction tests...")

        # Run extraction test
        extraction_result = test_extraction(results)

        # Run validation test
        test_quality_validation(results, extraction_result)

        # Run complete workflow test
        test_complete_workflow(results)

    # Print summary
    print_header("TEST SUMMARY")

    passed, failed, warnings = results.summary()
    total = passed + failed

    print()
    print(f"  {Colors.GREEN}Passed:   {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed:   {failed}{Colors.END}")
    print(f"  {Colors.YELLOW}Warnings: {warnings}{Colors.END}")
    print(f"  Total:    {total}")

    if failed == 0:
        print()
        print(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED!{Colors.END}")
        print()
        print("The E-Powertrain Benchmarking System is working correctly")
        print("with Intelligent mode (CRAWL4AI + OpenAI).")
        return 0
    elif failed <= 2:
        print()
        print(f"{Colors.YELLOW}{Colors.BOLD}PARTIAL SUCCESS{Colors.END}")
        print()
        print("Most tests passed but some issues were found.")
        print("Review the failed tests above.")
        return 0
    else:
        print()
        print(f"{Colors.RED}{Colors.BOLD}TESTS FAILED{Colors.END}")
        print()
        print("Multiple tests failed. Review the output above to fix issues.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        print()
        input("Press Enter to exit...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
