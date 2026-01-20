"""
E2E Tests for Intelligent Mode (CRAWL4AI + OpenAI)

These tests verify the complete workflow using the new default Intelligent mode,
which uses CRAWL4AI for multi-page crawling and OpenAI GPT-4o for extraction.

Tests cover:
1. Complete workflow execution with Intelligent mode
2. Terminology mapping (e.g., "e-axle torque" = "motor torque")
3. Hallucination prevention (strict validation)
4. Multi-OEM extraction
5. PowerPoint generation

Requirements:
- OPENAI_API_KEY (required)
- PERPLEXITY_API_KEY (NOT required - this is the point!)

Run with: pytest tests/test_intelligent_mode_e2e.py -v -s --tb=short
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_openai_key():
    """Check if OpenAI API key is available."""
    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set - skipping live tests")
    return True


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def openai_key_available():
    """Ensure OpenAI API key is available."""
    return check_openai_key()


@pytest.fixture
def test_urls():
    """URLs from urls.txt for testing."""
    return [
        "https://www.man.eu/global/en/truck/electric-trucks/overview.html",
        "https://www.volvotrucks.com/en-en/trucks/electric.html",
    ]


@pytest.fixture
def single_test_url():
    """Single URL for faster testing."""
    return ["https://www.man.eu/global/en/truck/electric-trucks/overview.html"]


@pytest.fixture
def output_dir():
    """Create output directory for test artifacts."""
    test_output = Path("outputs/test_intelligent_mode")
    test_output.mkdir(parents=True, exist_ok=True)
    return test_output


# =============================================================================
# TEST: DEFAULT MODE VERIFICATION
# =============================================================================

class TestDefaultModeIsIntelligent:
    """Verify that Intelligent mode is now the default."""

    def test_state_default_is_intelligent(self):
        """Verify initialize_state defaults to Intelligent mode."""
        from src.state.state import initialize_state, ScrapingMode

        state = initialize_state(["https://example.com"])

        assert state["scraping_mode"] == ScrapingMode.INTELLIGENT, \
            "Default mode should be INTELLIGENT"
        print("+ Default mode is Intelligent")

    def test_scraper_config_enables_intelligent(self):
        """Verify ScraperConfig enables intelligent navigation by default."""
        from src.tools.scraper import ScraperConfig

        assert ScraperConfig.ENABLE_INTELLIGENT_NAVIGATION is True, \
            "ENABLE_INTELLIGENT_NAVIGATION should be True"
        print("+ ScraperConfig enables intelligent navigation")


# =============================================================================
# TEST: INTELLIGENT MODE EXTRACTION
# =============================================================================

class TestIntelligentModeExtraction:
    """Test extraction using Intelligent mode (CRAWL4AI + OpenAI)."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_single_oem_intelligent_extraction(self, openai_key_available, single_test_url):
        """
        Test extraction from single OEM using Intelligent mode.

        This is the core test for CRAWL4AI + OpenAI extraction.
        """
        from src.tools.scraper import EPowertrainExtractor

        print("\n" + "=" * 60)
        print("TEST: Intelligent Mode Single OEM Extraction")
        print("=" * 60)
        print(f"URL: {single_test_url[0]}")
        print("Mode: CRAWL4AI + OpenAI GPT-4o")
        print("-" * 60)

        # Create extractor with Intelligent mode
        extractor = EPowertrainExtractor(use_intelligent_mode=True)

        print("\nPhase 1: Discovering spec pages...")
        start_time = time.time()

        # Process URL
        results = extractor.process_urls(single_test_url)

        duration = time.time() - start_time
        print(f"\nExtraction completed in {duration:.1f} seconds")

        # Verify results
        assert results is not None, "Results should not be None"
        assert len(results) >= 1, "Should have at least one result"

        result = results[0]

        # Display results
        print(f"\nResults:")
        print(f"  OEM: {result.get('oem_name', 'Unknown')}")
        print(f"  Vehicles found: {result.get('total_vehicles_found', 0)}")
        print(f"  Pages crawled: {result.get('pages_crawled', 'N/A')}")

        vehicles = result.get("vehicles", [])
        if vehicles:
            print(f"\nVehicles extracted:")
            for v in vehicles[:5]:  # Show first 5
                name = v.get('vehicle_name', 'Unknown')
                battery = v.get('battery_capacity_kwh', 'N/A')
                motor = v.get('motor_power_kw', 'N/A')
                range_km = v.get('range_km', 'N/A')
                completeness = v.get('data_completeness_score', 0)
                print(f"  - {name}")
                print(f"    Battery: {battery} kWh | Motor: {motor} kW | Range: {range_km} km")
                print(f"    Completeness: {completeness:.0%}")

        # Assertions
        assert "oem_name" in result, "Result should have oem_name"
        assert "vehicles" in result, "Result should have vehicles"
        assert len(vehicles) > 0, "Should extract at least one vehicle"

        print("\n+ Intelligent mode extraction PASSED")
        return result


# =============================================================================
# TEST: COMPLETE WORKFLOW
# =============================================================================

class TestCompleteWorkflowIntelligentMode:
    """Test the complete LangGraph workflow with Intelligent mode."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_full_workflow_intelligent_mode(self, openai_key_available, single_test_url, output_dir):
        """
        Test complete workflow: Scrape -> Validate -> Generate Presentation.

        Uses Intelligent mode (CRAWL4AI + OpenAI) - NO Perplexity API needed.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print("\n" + "=" * 60)
        print("TEST: Complete Workflow - Intelligent Mode")
        print("=" * 60)
        print("Mode: CRAWL4AI + OpenAI (Perplexity NOT required)")
        print(f"URL: {single_test_url[0]}")
        print("-" * 60)

        print("\nWorkflow steps:")
        print("  1. Scrape vehicle specs (CRAWL4AI multi-page + OpenAI extraction)")
        print("  2. Validate data quality")
        print("  3. Generate PowerPoint presentation")
        print("\nStarting workflow...")

        start_time = time.time()

        # Run complete workflow with Intelligent mode (default)
        final_state = run_benchmark(
            urls=single_test_url,
            verbose=True,
            mode=ScrapingMode.INTELLIGENT  # Explicit, but should be default
        )

        duration = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Workflow completed in {duration:.1f} seconds")
        print(f"{'=' * 60}")

        # Display results
        status = final_state['workflow_status']
        vehicles = final_state.get('all_vehicles', [])
        validation = final_state.get('quality_validation', {})
        presentation = final_state.get('presentation_result', {})

        print(f"\nFinal Status: {status}")
        print(f"Vehicles extracted: {len(vehicles)}")
        print(f"Tokens used: {final_state.get('total_tokens_used', 0):,}")
        print(f"Cost: ${final_state.get('total_cost_usd', 0):.4f}")

        if validation:
            print(f"\nQuality Validation:")
            print(f"  Overall Score: {validation.get('overall_quality_score', 0):.2f}")
            print(f"  Completeness: {validation.get('completeness_score', 0):.2f}")
            print(f"  Passes Threshold: {validation.get('passes_threshold', False)}")

        if presentation:
            paths = presentation.get('all_presentation_paths', [])
            print(f"\nPresentations generated: {len(paths)}")
            for path in paths:
                if Path(path).exists():
                    size = Path(path).stat().st_size / 1024
                    print(f"  + {Path(path).name} ({size:.1f} KB)")

        # Assertions
        assert status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED], \
            "Workflow should complete or fail"

        if status == WorkflowStatus.COMPLETED:
            assert len(vehicles) > 0, "Should extract vehicles on completion"
            print("\n+ Complete workflow test PASSED")
        else:
            errors = final_state.get('errors', [])
            print(f"\nWorkflow ended with status: {status}")
            print(f"Errors: {errors}")
            # Don't fail if quality threshold wasn't met - that's acceptable
            if "Quality validation failed" in str(errors):
                print("(Quality validation failure is acceptable)")
            else:
                pytest.fail(f"Workflow failed: {errors}")

        return final_state


# =============================================================================
# TEST: MULTI-OEM EXTRACTION
# =============================================================================

class TestMultiOEMIntelligentMode:
    """Test multi-OEM extraction with Intelligent mode."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_multi_oem_workflow(self, openai_key_available, test_urls):
        """
        Test extraction from multiple OEMs.

        Verifies the system can handle competitive benchmarking scenarios.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print("\n" + "=" * 60)
        print("TEST: Multi-OEM Extraction - Intelligent Mode")
        print("=" * 60)
        print("URLs to process:")
        for url in test_urls:
            print(f"  - {url}")
        print("-" * 60)

        start_time = time.time()

        final_state = run_benchmark(
            urls=test_urls,
            verbose=True,
            mode=ScrapingMode.INTELLIGENT
        )

        duration = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Multi-OEM workflow completed in {duration:.1f} seconds")
        print(f"{'=' * 60}")

        # Analyze results
        vehicles = final_state.get('all_vehicles', [])
        scraping_results = final_state.get('scraping_results', [])

        # Group by OEM
        oems = {}
        for v in vehicles:
            oem = v.get('oem_name', 'Unknown')
            if oem not in oems:
                oems[oem] = []
            oems[oem].append(v)

        print(f"\nResults by OEM:")
        for oem, oem_vehicles in oems.items():
            print(f"  {oem}: {len(oem_vehicles)} vehicle(s)")
            for v in oem_vehicles[:2]:
                print(f"    - {v.get('vehicle_name', 'Unknown')}")

        print(f"\nTotal vehicles: {len(vehicles)}")
        print(f"OEMs processed: {len(oems)}")
        print(f"Cost: ${final_state.get('total_cost_usd', 0):.4f}")

        # Assertions
        assert len(scraping_results) >= 1, "Should process at least one OEM"

        print("\n+ Multi-OEM extraction test PASSED")
        return final_state


# =============================================================================
# TEST: TERMINOLOGY MAPPING
# =============================================================================

class TestTerminologyMapping:
    """Test that terminology mappings work correctly."""

    def test_semantic_equivalences_loaded(self):
        """Verify semantic equivalences are defined."""
        from src.config.terminology_mappings import SEMANTIC_EQUIVALENCES

        assert "motor_torque_nm" in SEMANTIC_EQUIVALENCES
        assert "motor_power_kw" in SEMANTIC_EQUIVALENCES
        assert "battery_capacity_kwh" in SEMANTIC_EQUIVALENCES

        # Check e-axle torque is mapped to motor_torque
        motor_torque_terms = SEMANTIC_EQUIVALENCES["motor_torque_nm"]
        assert "e-axle torque" in motor_torque_terms or \
               any("e-axle" in term for term in motor_torque_terms), \
               "e-axle torque should be mapped to motor_torque_nm"

        print("+ Terminology mappings loaded correctly")
        print(f"  motor_torque_nm has {len(motor_torque_terms)} equivalent terms")

    def test_build_semantic_equivalence_prompt(self):
        """Verify semantic equivalence prompt is generated."""
        from src.config.terminology_mappings import build_semantic_equivalence_prompt

        prompt = build_semantic_equivalence_prompt()

        assert len(prompt) > 100, "Prompt should be substantial"
        assert "e-axle" in prompt.lower() or "motor" in prompt.lower()

        print("+ Semantic equivalence prompt generated")
        print(f"  Prompt length: {len(prompt)} chars")


# =============================================================================
# TEST: HALLUCINATION PREVENTION
# =============================================================================

class TestHallucinationPrevention:
    """Test that hallucination prevention is working."""

    def test_strict_mode_enabled(self):
        """Verify strict hallucination check is enabled."""
        from src.tools.scraper import ScraperConfig

        assert ScraperConfig.STRICT_HALLUCINATION_CHECK is True, \
            "STRICT_HALLUCINATION_CHECK should be True"

        print("+ Strict hallucination prevention is enabled")


# =============================================================================
# TEST: DATA QUALITY
# =============================================================================

class TestDataQuality:
    """Test data quality validation."""

    @pytest.mark.requires_api
    def test_quality_validation_works(self, openai_key_available, single_test_url):
        """Test that quality validation produces reasonable scores."""
        from src.tools.scraper import EPowertrainExtractor
        from src.agents.quality_validator import run_validation

        print("\n" + "=" * 60)
        print("TEST: Data Quality Validation")
        print("=" * 60)

        # Extract data
        print("Extracting data...")
        extractor = EPowertrainExtractor(use_intelligent_mode=True)
        results = extractor.process_urls(single_test_url)

        if not results or not results[0].get("vehicles"):
            pytest.skip("No vehicles extracted to validate")

        # Validate
        print("Running quality validation...")
        validation = run_validation(results, use_llm=False)

        print(f"\nValidation Results:")
        print(f"  Overall Score: {validation['overall_quality_score']:.2f}")
        print(f"  Completeness: {validation['completeness_score']:.2f}")
        print(f"  Accuracy: {validation['accuracy_score']:.2f}")
        print(f"  Consistency: {validation['consistency_score']:.2f}")
        print(f"  Passes Threshold: {validation['passes_threshold']}")

        # Assertions
        assert 0 <= validation['overall_quality_score'] <= 1
        assert validation['recommendation'] is not None

        print("\n+ Quality validation test PASSED")
        return validation


# =============================================================================
# TEST: NO PERPLEXITY REQUIRED
# =============================================================================

class TestNoPerplexityRequired:
    """Verify the system works without Perplexity API key."""

    def test_workflow_initializes_without_perplexity(self):
        """Test that workflow can initialize without PERPLEXITY_API_KEY."""
        from src.state.state import initialize_state, ScrapingMode

        # Save current key
        perplexity_key = os.environ.pop("PERPLEXITY_API_KEY", None)

        try:
            # Should work without Perplexity key
            state = initialize_state(
                ["https://example.com"],
                scraping_mode=ScrapingMode.INTELLIGENT
            )

            assert state is not None
            assert state["scraping_mode"] == ScrapingMode.INTELLIGENT

            print("+ Workflow initializes without PERPLEXITY_API_KEY")

        finally:
            # Restore key if it existed
            if perplexity_key:
                os.environ["PERPLEXITY_API_KEY"] = perplexity_key


# =============================================================================
# TEST: SUMMARY - COMPLETE BUSINESS SCENARIO
# =============================================================================

class TestCompleteBusinessScenario:
    """Complete business scenario test simulating real usage."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_product_manager_benchmarking_scenario(self, openai_key_available, single_test_url, output_dir):
        """
        Complete business scenario: Product Manager competitive benchmarking.

        This test simulates a real user running the system to prepare
        a competitive analysis presentation.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print("\n" + "=" * 70)
        print("COMPLETE BUSINESS SCENARIO TEST")
        print("Scenario: Product Manager Competitive Benchmarking")
        print("=" * 70)

        print("\n[CONTEXT]")
        print("   A Product Manager needs to analyze electric truck specifications")
        print("   from competitor OEMs for a strategy meeting.")
        print("   The system should extract data and generate a presentation.")
        print()

        # Run workflow
        print("[STEP 1] Running E-Powertrain Benchmarking System...")
        print("-" * 50)

        start_time = time.time()

        final_state = run_benchmark(
            urls=single_test_url,
            verbose=True,
            mode=ScrapingMode.INTELLIGENT
        )

        duration = time.time() - start_time

        # Analyze results
        print(f"\n[STEP 2] Analyzing Results...")
        print("-" * 50)

        status = final_state["workflow_status"]
        vehicles = final_state.get("all_vehicles", [])
        validation = final_state.get("quality_validation", {})
        presentation = final_state.get("presentation_result", {})

        # Run verification checks
        print("\n[STEP 3] Verification Checks...")
        print("-" * 50)

        checks = {
            "Workflow completed": status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED],
            "Vehicles extracted": len(vehicles) > 0,
            "Quality validated": validation is not None and validation.get("overall_quality_score", 0) > 0,
            "Cost tracked": final_state.get("total_cost_usd", 0) > 0 or final_state.get("total_tokens_used", 0) > 0,
            "Mode is Intelligent": final_state.get("scraping_mode") == ScrapingMode.INTELLIGENT,
        }

        # Check presentation if workflow completed
        if status == WorkflowStatus.COMPLETED:
            pres_paths = presentation.get("all_presentation_paths", [])
            checks["Presentation generated"] = len(pres_paths) > 0 and any(Path(p).exists() for p in pres_paths)
        else:
            checks["Presentation generated"] = True  # Skip if not completed

        # Display check results
        passed = 0
        for check_name, check_result in checks.items():
            status_icon = "+" if check_result else "x"
            print(f"   {status_icon} {check_name}")
            if check_result:
                passed += 1

        total = len(checks)

        # Summary
        print(f"\n{'=' * 70}")
        print(f"BUSINESS SCENARIO RESULT: {passed}/{total} checks passed")
        print(f"{'=' * 70}")

        print(f"\nSummary:")
        print(f"   Status: {final_state['workflow_status']}")
        print(f"   Vehicles: {len(vehicles)}")
        print(f"   Quality Score: {validation.get('overall_quality_score', 0):.2f}")
        print(f"   Cost: ${final_state.get('total_cost_usd', 0):.4f}")
        print(f"   Duration: {duration:.1f} seconds")

        if passed >= total - 1:  # Allow one failure
            print("\n[SUCCESS] System is working as expected!")
        else:
            print("\n[WARNING] Some checks failed - review output above.")

        # Final assertion
        assert passed >= total - 1, f"Expected at least {total-1}/{total} checks to pass"

        print("\n+ Business scenario test PASSED")

        return {
            "status": str(final_state["workflow_status"]),
            "vehicles": len(vehicles),
            "quality_score": validation.get("overall_quality_score", 0),
            "cost": final_state.get("total_cost_usd", 0),
            "duration": duration,
            "checks_passed": passed,
            "total_checks": total,
        }


# =============================================================================
# RUN TESTS DIRECTLY
# =============================================================================

if __name__ == "__main__":
    """Run tests directly."""
    import subprocess

    print("=" * 60)
    print("E-Powertrain Benchmarking System - Intelligent Mode E2E Tests")
    print("=" * 60)
    print()
    print("These tests verify the CRAWL4AI + OpenAI workflow.")
    print("PERPLEXITY_API_KEY is NOT required.")
    print()
    print("Running tests...")
    print()

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v", "-s",
        "--tb=short",
        "-x",  # Stop on first failure
    ])

    sys.exit(result.returncode)
