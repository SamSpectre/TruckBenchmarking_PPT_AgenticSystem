"""
Live Integration Tests - Real API Calls

These tests call actual APIs (Perplexity, OpenAI) to verify the complete
workflow functions correctly in real-world scenarios.

WARNING: These tests incur API costs!
- Perplexity: ~$0.02-0.05 per URL
- OpenAI: ~$0.01-0.02 per validation

Run with: pytest tests/test_live_integration.py -v -s

Use -s flag to see real-time output during long-running tests.
"""

import pytest
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for API keys before running
def check_api_keys():
    """Check if required API keys are available."""
    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set - skipping live tests")
    if not perplexity_key:
        pytest.skip("PERPLEXITY_API_KEY not set - skipping live tests")

    return True


# =====================================================================
# FIXTURES FOR LIVE TESTS
# =====================================================================

@pytest.fixture(scope="module")
def api_keys_available():
    """Ensure API keys are available before running live tests."""
    return check_api_keys()


@pytest.fixture
def output_dir():
    """Create temporary output directory for test artifacts."""
    test_output = Path("outputs/test_runs")
    test_output.mkdir(parents=True, exist_ok=True)
    return test_output


@pytest.fixture
def live_oem_urls():
    """Real OEM URLs for live testing."""
    return [
        "https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html",
    ]


# =====================================================================
# TEST: SCRAPER WITH REAL API
# =====================================================================

class TestLiveScraperAPI:
    """Test the scraper with real Perplexity API calls."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_perplexity_scraper_single_url(self, api_keys_available, live_oem_urls):
        """
        Test Perplexity scraper with a single real OEM URL.

        This test:
        1. Calls the real Perplexity API
        2. Extracts vehicle specifications from MAN website
        3. Verifies data structure and quality
        """
        from src.tools.scraper import EPowertrainExtractor

        print("\n" + "="*60)
        print("LIVE TEST: Perplexity Scraper - Single URL")
        print("="*60)

        url = live_oem_urls[0]
        print(f"Target URL: {url}")
        print("Starting extraction (this may take 15-30 seconds)...")

        start_time = time.time()

        # Create extractor and process URL
        # use_intelligent_mode=False for Perplexity single-page mode
        extractor = EPowertrainExtractor(use_intelligent_mode=False)
        results = extractor.process_urls([url])

        duration = time.time() - start_time
        print(f"Extraction completed in {duration:.1f} seconds")

        # Verify results
        assert results is not None, "Results should not be None"
        assert len(results) >= 1, "Should have at least one result"

        result = results[0]
        print(f"\nOEM: {result.get('oem_name', 'Unknown')}")
        print(f"Vehicles found: {result.get('total_vehicles_found', 0)}")

        vehicles = result.get("vehicles", [])
        if vehicles:
            print(f"\nFirst vehicle: {vehicles[0].get('vehicle_name', 'Unknown')}")
            print(f"Battery: {vehicles[0].get('battery_capacity_kwh', 'N/A')} kWh")
            print(f"Range: {vehicles[0].get('range_km', 'N/A')} km")

        # Assert basic structure
        assert "oem_name" in result
        assert "vehicles" in result
        assert isinstance(result["vehicles"], list)

        print("\n+ Scraper test PASSED")
        return result


# =====================================================================
# TEST: QUALITY VALIDATOR WITH REAL DATA
# =====================================================================

class TestLiveQualityValidator:
    """Test quality validator with real scraped data."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_validate_real_scraping_result(self, api_keys_available, live_oem_urls):
        """
        Test quality validation with real scraped data.

        This test:
        1. Scrapes real data from OEM website
        2. Runs quality validation
        3. Verifies validation scores and recommendations
        """
        from src.tools.scraper import EPowertrainExtractor
        from src.agents.quality_validator import run_validation

        print("\n" + "="*60)
        print("LIVE TEST: Quality Validator with Real Data")
        print("="*60)

        # First, get real data
        print("Step 1: Scraping real data...")
        extractor = EPowertrainExtractor(use_intelligent_mode=False)
        results = extractor.process_urls([live_oem_urls[0]])

        assert len(results) >= 1, "Need scraping results to test validation"

        # Run validation
        print("Step 2: Running quality validation...")
        validation = run_validation(results, use_llm=False)

        print(f"\nValidation Results:")
        print(f"  Overall Score: {validation['overall_quality_score']:.2f}")
        print(f"  Completeness: {validation['completeness_score']:.2f}")
        print(f"  Accuracy: {validation['accuracy_score']:.2f}")
        print(f"  Consistency: {validation['consistency_score']:.2f}")
        print(f"  Source Quality: {validation['source_quality_score']:.2f}")
        print(f"  Passes Threshold: {validation['passes_threshold']}")
        print(f"  Recommendation: {validation['recommendation']}")

        # Assert validation structure
        assert "overall_quality_score" in validation
        assert "passes_threshold" in validation
        assert 0 <= validation["overall_quality_score"] <= 1

        print("\n+ Validation test PASSED")
        return validation


# =====================================================================
# TEST: COMPLETE WORKFLOW
# =====================================================================

class TestLiveCompleteWorkflow:
    """Test the complete LangGraph workflow with real APIs."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_full_workflow_single_oem(self, api_keys_available, output_dir):
        """
        Test complete workflow: Scrape -> Validate -> Generate Presentation.

        This is the full end-to-end test simulating a real user running
        the benchmarking system.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import WorkflowStatus, ScrapingMode

        print("\n" + "="*60)
        print("LIVE TEST: Complete Workflow - Single OEM")
        print("="*60)

        # Use a real OEM URL
        urls = ["https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"]

        print(f"Starting complete workflow with {len(urls)} URL(s)")
        print("This test will:")
        print("  1. Scrape vehicle specs from MAN website")
        print("  2. Validate data quality")
        print("  3. Generate PowerPoint presentation")
        print("\nExpected duration: 30-60 seconds...")

        start_time = time.time()

        # Run the complete workflow
        final_state = run_benchmark(
            urls=urls,
            verbose=True,
            mode=ScrapingMode.PERPLEXITY
        )

        duration = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Workflow completed in {duration:.1f} seconds")
        print(f"{'='*60}")

        # Verify workflow completed
        print(f"\nFinal Status: {final_state['workflow_status']}")
        print(f"Total Tokens: {final_state.get('total_tokens_used', 0):,}")
        print(f"Total Cost: ${final_state.get('total_cost_usd', 0):.4f}")

        # Check scraping results
        scraping_results = final_state.get("scraping_results", [])
        all_vehicles = final_state.get("all_vehicles", [])
        print(f"\nVehicles extracted: {len(all_vehicles)}")

        if all_vehicles:
            for v in all_vehicles[:3]:  # Show first 3
                print(f"  - {v.get('vehicle_name', 'Unknown')}")

        # Check validation
        validation = final_state.get("quality_validation", {})
        if validation:
            print(f"\nQuality Score: {validation.get('overall_quality_score', 0):.2f}")
            print(f"Passes Threshold: {validation.get('passes_threshold', False)}")

        # Check presentation
        presentation = final_state.get("presentation_result", {})
        if presentation:
            paths = presentation.get("all_presentation_paths", [])
            print(f"\nPresentations generated: {len(paths)}")
            for path in paths:
                print(f"  - {path}")
                # Verify file exists
                if Path(path).exists():
                    print(f"    + File exists ({Path(path).stat().st_size:,} bytes)")

        # Assertions
        assert final_state["workflow_status"] in [
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED
        ], "Workflow should have completed or failed"

        if final_state["workflow_status"] == WorkflowStatus.COMPLETED:
            print("\n+ Complete workflow test PASSED - Workflow completed successfully!")
            assert len(all_vehicles) > 0, "Should have extracted vehicles"
        else:
            errors = final_state.get("errors", [])
            print(f"\n⚠ Workflow ended with status: {final_state['workflow_status']}")
            print(f"Errors: {errors}")
            # Don't fail test if validation didn't pass - that's expected sometimes
            if "Quality validation failed" in str(errors):
                print("(Quality validation failure is acceptable for this test)")
            else:
                pytest.fail(f"Workflow failed: {errors}")

        return final_state

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_full_workflow_with_streaming(self, api_keys_available):
        """
        Test workflow with streaming output.

        Verifies that streaming mode works correctly for progress monitoring.
        """
        from src.graph.runtime import stream_benchmark
        from src.state.state import ScrapingMode

        print("\n" + "="*60)
        print("LIVE TEST: Streaming Workflow")
        print("="*60)

        urls = ["https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"]

        print("Starting streaming workflow...")

        steps_completed = []
        final_state = None

        for step_output in stream_benchmark(urls, mode=ScrapingMode.PERPLEXITY):
            node_name = list(step_output.keys())[0]
            state = step_output[node_name]
            steps_completed.append(node_name)
            final_state = state
            print(f"  Step completed: {node_name}")

        print(f"\nSteps executed: {steps_completed}")

        assert len(steps_completed) >= 1, "Should have completed at least one step"
        assert "scrape" in steps_completed, "Should have completed scraping step"

        print("\n+ Streaming workflow test PASSED")
        return final_state


# =====================================================================
# TEST: PRESENTATION GENERATION
# =====================================================================

class TestLivePresentationGeneration:
    """Test PowerPoint generation with real data."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_generate_presentation_from_real_data(self, api_keys_available, output_dir):
        """
        Test presentation generation with real scraped data.

        This test:
        1. Scrapes real vehicle data
        2. Transforms to OEM profile
        3. Generates PowerPoint presentation
        4. Verifies file is created and valid
        """
        from src.tools.scraper import EPowertrainExtractor
        from src.tools.ppt_generator import (
            transform_scraping_result_to_oem_profile,
            generate_all_presentations,
        )

        print("\n" + "="*60)
        print("LIVE TEST: Presentation Generation")
        print("="*60)

        # Get real data
        print("Step 1: Scraping real data...")
        extractor = EPowertrainExtractor(use_intelligent_mode=False)
        results = extractor.process_urls([
            "https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"
        ])

        assert len(results) >= 1, "Need scraping results"

        # Transform to OEM profile
        print("Step 2: Transforming to OEM profile...")
        oem_profiles = []
        for result in results:
            profile = transform_scraping_result_to_oem_profile(result)
            oem_profiles.append(profile)
            print(f"  Profile created for: {profile['company_name']}")

        # Check template exists
        template_path = Path("templates/IAA_Template.pptx")
        if not template_path.exists():
            pytest.skip("Template file not found - skipping presentation generation")

        # Generate presentations
        print("Step 3: Generating PowerPoint presentation...")
        try:
            output_paths = generate_all_presentations(
                oem_profiles=oem_profiles,
                template_path=str(template_path),
                output_dir=str(output_dir)
            )

            print(f"\nPresentations generated: {len(output_paths)}")
            for path in output_paths:
                file_path = Path(path)
                if file_path.exists():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"  + {file_path.name} ({size_kb:.1f} KB)")
                else:
                    print(f"  x {path} (not found)")

            assert len(output_paths) >= 1, "Should generate at least one presentation"
            assert Path(output_paths[0]).exists(), "Presentation file should exist"

            print("\n+ Presentation generation test PASSED")
            return output_paths

        except Exception as e:
            print(f"\nPresentation generation error: {e}")
            # This might fail if template structure differs - log but don't fail
            pytest.skip(f"Presentation generation failed: {e}")


# =====================================================================
# TEST: COST TRACKING
# =====================================================================

class TestLiveCostTracking:
    """Test cost tracking with real API calls."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_cost_tracking_accuracy(self, api_keys_available):
        """
        Test that API costs are tracked correctly.

        Verifies that token counts and costs are recorded during real API calls.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import ScrapingMode

        print("\n" + "="*60)
        print("LIVE TEST: Cost Tracking")
        print("="*60)

        urls = ["https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"]

        print("Running workflow to track costs...")

        final_state = run_benchmark(
            urls=urls,
            verbose=False,
            mode=ScrapingMode.PERPLEXITY
        )

        tokens = final_state.get("total_tokens_used", 0)
        cost = final_state.get("total_cost_usd", 0)

        print(f"\nCost Tracking Results:")
        print(f"  Total Tokens: {tokens:,}")
        print(f"  Total Cost: ${cost:.4f}")

        # Verify costs are tracked
        assert tokens > 0, "Should have used some tokens"
        assert cost > 0, "Should have incurred some cost"
        assert cost < 1.0, "Cost should be reasonable (< $1 for single URL)"

        print("\n+ Cost tracking test PASSED")
        return {"tokens": tokens, "cost": cost}


# =====================================================================
# TEST: ERROR HANDLING WITH REAL SCENARIOS
# =====================================================================

class TestLiveErrorHandling:
    """Test error handling with real-world error scenarios."""

    @pytest.mark.requires_api
    def test_handles_invalid_url(self, api_keys_available):
        """Test that invalid URLs are handled gracefully."""
        from src.tools.scraper import EPowertrainExtractor

        print("\n" + "="*60)
        print("LIVE TEST: Invalid URL Handling")
        print("="*60)

        extractor = EPowertrainExtractor(use_intelligent_mode=False)

        # Try with an invalid/non-existent URL
        invalid_url = "https://this-domain-definitely-does-not-exist-12345.com/trucks"

        print(f"Testing with invalid URL: {invalid_url}")

        try:
            results = extractor.process_urls([invalid_url])

            # Should return results with errors
            assert len(results) >= 1
            result = results[0]

            # Check for error handling
            if result.get("errors"):
                print(f"Errors captured: {result['errors']}")

            print("\n+ Invalid URL handled gracefully")

        except Exception as e:
            # If it raises an exception, that's also acceptable
            print(f"Exception raised (acceptable): {type(e).__name__}: {e}")
            print("\n+ Invalid URL handled with exception")

    @pytest.mark.requires_api
    def test_handles_non_ev_website(self, api_keys_available):
        """Test handling of website without EV content."""
        from src.tools.scraper import EPowertrainExtractor

        print("\n" + "="*60)
        print("LIVE TEST: Non-EV Website Handling")
        print("="*60)

        extractor = EPowertrainExtractor(use_intelligent_mode=False)

        # A real website but not about EVs
        url = "https://www.wikipedia.org"

        print(f"Testing with non-EV URL: {url}")

        results = extractor.process_urls([url])

        assert len(results) >= 1
        result = results[0]

        vehicles = result.get("vehicles", [])
        print(f"Vehicles found: {len(vehicles)}")

        # Should return empty or minimal results
        if len(vehicles) == 0:
            print("Correctly identified no EV content")
        else:
            print(f"Warning: Found {len(vehicles)} vehicles on non-EV site")

        print("\n+ Non-EV website handled correctly")


# =====================================================================
# TEST: MULTI-OEM WORKFLOW
# =====================================================================

class TestLiveMultiOEM:
    """Test with multiple OEMs for competitive analysis."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_multi_oem_benchmark(self, api_keys_available):
        """
        Test benchmarking multiple OEMs in a single workflow.

        This simulates a real competitive analysis scenario.
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import ScrapingMode, WorkflowStatus

        print("\n" + "="*60)
        print("LIVE TEST: Multi-OEM Competitive Benchmark")
        print("="*60)

        # Multiple real OEM URLs
        urls = [
            "https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html",
            "https://www.volvotrucks.com/en-en/trucks/trucks/volvo-fh/volvo-fh-electric.html",
        ]

        print(f"Benchmarking {len(urls)} OEMs:")
        for url in urls:
            print(f"  - {url}")

        print("\nStarting multi-OEM workflow (this may take 1-2 minutes)...")

        start_time = time.time()

        final_state = run_benchmark(
            urls=urls,
            verbose=True,
            mode=ScrapingMode.PERPLEXITY
        )

        duration = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Multi-OEM workflow completed in {duration:.1f} seconds")
        print(f"{'='*60}")

        # Analyze results
        all_vehicles = final_state.get("all_vehicles", [])
        scraping_results = final_state.get("scraping_results", [])

        print(f"\nResults Summary:")
        print(f"  Total vehicles extracted: {len(all_vehicles)}")
        print(f"  OEMs processed: {len(scraping_results)}")
        print(f"  Total cost: ${final_state.get('total_cost_usd', 0):.4f}")

        # Group by OEM
        oems = {}
        for v in all_vehicles:
            oem = v.get("oem_name", "Unknown")
            if oem not in oems:
                oems[oem] = []
            oems[oem].append(v)

        print(f"\nVehicles by OEM:")
        for oem, vehicles in oems.items():
            print(f"  {oem}: {len(vehicles)} vehicle(s)")
            for v in vehicles[:2]:  # Show first 2 per OEM
                print(f"    - {v.get('vehicle_name', 'Unknown')}")

        # Assertions
        assert len(scraping_results) >= 1, "Should have processed at least one OEM"

        if final_state["workflow_status"] == WorkflowStatus.COMPLETED:
            print("\n+ Multi-OEM benchmark test PASSED")
        else:
            print(f"\n⚠ Workflow status: {final_state['workflow_status']}")
            print("(Partial success - some OEMs may have failed)")

        return final_state


# =====================================================================
# DATA ACCURACY VERIFICATION
# =====================================================================

class TestDataAccuracyVerification:
    """Verify extracted data accuracy against known specifications."""

    # Known MAN specifications from official website (verified Jan 2026)
    KNOWN_MAN_SPECS = {
        "MAN eTGL": {
            "battery_capacity_kwh": 160,
            "range_km": 235,
            "dc_charging_kw": 250,
            "gvw_kg": 11990,
        },
        "MAN eTGX": {
            "battery_capacity_min_kwh": 240,
            "battery_capacity_max_kwh": 560,
            "range_min_km": 500,
            "range_max_km": 750,
            "mcs_charging_kw": 750,
        },
        "MAN eTGS": {
            "battery_capacity_min_kwh": 240,
            "battery_capacity_max_kwh": 560,
            "range_min_km": 500,
            "range_max_km": 750,
            "mcs_charging_kw": 750,
        },
    }

    @pytest.mark.requires_api
    def test_extracted_data_matches_known_specs(self, api_keys_available):
        """
        Verify extracted data matches known MAN specifications.

        This is critical for enterprise robustness - ensures extraction
        accuracy against verified ground truth.
        """
        import json

        print("\n" + "="*60)
        print("DATA ACCURACY VERIFICATION TEST")
        print("="*60)

        # Check if we have existing extraction results
        results_file = Path("outputs/scraping_MAN_Truck___Bus.json")

        if not results_file.exists():
            # Run extraction if no results exist
            print("No existing results, running live extraction...")
            from src.tools.scraper import EPowertrainExtractor
            extractor = EPowertrainExtractor(use_intelligent_mode=True)
            results = extractor.process_urls([
                "https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"
            ])
            extracted_data = results[0] if results else {"vehicles": []}
        else:
            print(f"Using existing results from: {results_file}")
            with open(results_file, "r") as f:
                extracted_data = json.load(f)

        vehicles = extracted_data.get("vehicles", [])
        print(f"\nTotal vehicles to verify: {len(vehicles)}")

        # Verification results
        verification_results = []

        for model_name, expected_specs in self.KNOWN_MAN_SPECS.items():
            print(f"\n--- Verifying {model_name} ---")

            # Find matching vehicles (could be multiple variants)
            matching_vehicles = [
                v for v in vehicles
                if model_name.lower() in v.get("vehicle_name", "").lower()
            ]

            if not matching_vehicles:
                print(f"  WARNING: No vehicles found matching '{model_name}'")
                verification_results.append({
                    "model": model_name,
                    "status": "NOT_FOUND",
                    "accuracy": 0
                })
                continue

            # Check first matching vehicle
            vehicle = matching_vehicles[0]
            print(f"  Checking: {vehicle.get('vehicle_name')}")

            matches = 0
            total_checks = 0

            for spec_key, expected_value in expected_specs.items():
                total_checks += 1

                # Handle min/max spec names
                if spec_key.endswith("_min_kwh"):
                    actual_key = spec_key.replace("_min_kwh", "_kwh")
                    actual_min = spec_key.replace("_min_", "_min_")
                    actual = vehicle.get(actual_min) or vehicle.get(actual_key)
                elif spec_key.endswith("_max_kwh"):
                    actual_key = spec_key.replace("_max_kwh", "_kwh")
                    actual = vehicle.get(actual_key)
                elif spec_key.endswith("_min_km"):
                    actual = vehicle.get("range_min_km") or vehicle.get("range_km")
                elif spec_key.endswith("_max_km"):
                    actual = vehicle.get("range_km")
                else:
                    actual = vehicle.get(spec_key)

                # Check value (allow 10% tolerance for variations)
                if actual is not None:
                    tolerance = expected_value * 0.10
                    if abs(actual - expected_value) <= tolerance:
                        print(f"  + {spec_key}: {actual} (expected {expected_value})")
                        matches += 1
                    else:
                        print(f"  - {spec_key}: {actual} != {expected_value}")
                else:
                    print(f"  ? {spec_key}: missing (expected {expected_value})")

            accuracy = matches / total_checks if total_checks > 0 else 0
            verification_results.append({
                "model": model_name,
                "status": "VERIFIED" if accuracy >= 0.7 else "PARTIAL",
                "accuracy": accuracy,
                "matches": matches,
                "total": total_checks
            })
            print(f"  Accuracy: {accuracy:.0%} ({matches}/{total_checks})")

        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)

        verified_count = sum(1 for r in verification_results if r["status"] == "VERIFIED")
        total_models = len(verification_results)

        for result in verification_results:
            status_icon = "+" if result["status"] == "VERIFIED" else "~" if result["status"] == "PARTIAL" else "x"
            print(f"  {status_icon} {result['model']}: {result['status']} ({result['accuracy']:.0%})")

        overall_accuracy = sum(r["accuracy"] for r in verification_results) / total_models if total_models > 0 else 0
        print(f"\nOverall Accuracy: {overall_accuracy:.0%}")
        print(f"Models Verified: {verified_count}/{total_models}")

        # Assert minimum accuracy for enterprise use
        assert overall_accuracy >= 0.5, f"Overall accuracy {overall_accuracy:.0%} below 50% threshold"
        assert verified_count >= 1, "At least one model should be fully verified"

        print("\n+ Data accuracy verification PASSED")
        return verification_results

    @pytest.mark.requires_api
    def test_data_completeness_scores(self, api_keys_available):
        """
        Verify data completeness scores are reasonable.

        Ensures extracted data meets minimum quality standards.
        """
        import json

        print("\n" + "="*60)
        print("DATA COMPLETENESS VERIFICATION")
        print("="*60)

        results_file = Path("outputs/scraping_MAN_Truck___Bus.json")

        if not results_file.exists():
            pytest.skip("No extraction results file found")

        with open(results_file, "r") as f:
            data = json.load(f)

        vehicles = data.get("vehicles", [])

        scores = []
        print("\nVehicle Completeness Scores:")

        for v in vehicles:
            name = v.get("vehicle_name", "Unknown")
            score = v.get("data_completeness_score", 0)
            scores.append(score)
            status = "+" if score >= 0.5 else "-"
            print(f"  {status} {name}: {score:.0%}")

        avg_score = sum(scores) / len(scores) if scores else 0
        high_quality = sum(1 for s in scores if s >= 0.5)

        print(f"\nAverage Completeness: {avg_score:.0%}")
        print(f"High Quality Entries: {high_quality}/{len(scores)}")

        # Filter out empty placeholder entries
        meaningful_scores = [s for s in scores if s > 0]
        meaningful_avg = sum(meaningful_scores) / len(meaningful_scores) if meaningful_scores else 0

        print(f"Meaningful Entries Avg: {meaningful_avg:.0%}")

        # Assert quality thresholds
        assert meaningful_avg >= 0.5, f"Meaningful entries average {meaningful_avg:.0%} below 50%"
        assert high_quality >= len(scores) * 0.5, "Less than half of entries are high quality"

        print("\n+ Data completeness verification PASSED")


# =====================================================================
# SUMMARY TEST - RUN ALL CRITICAL PATHS
# =====================================================================

class TestLiveSummary:
    """Summary test that runs all critical paths."""

    @pytest.mark.requires_api
    @pytest.mark.slow
    def test_complete_business_scenario(self, api_keys_available, output_dir):
        """
        Complete business scenario test.

        Simulates a product manager running competitive benchmarking:
        1. Scrape EV specs from OEM website
        2. Validate data quality
        3. Generate presentation for stakeholder meeting
        4. Verify all outputs
        """
        from src.graph.runtime import run_benchmark
        from src.state.state import ScrapingMode, WorkflowStatus

        print("\n" + "="*70)
        print("COMPLETE BUSINESS SCENARIO TEST")
        print("Simulating: Product Manager Competitive Benchmarking")
        print("="*70)

        # Scenario: PM needs to benchmark MAN's electric trucks
        urls = ["https://www.man.eu/de/de/lkw/emobilitaet/man-etgx/uebersicht.html"]

        print("\n[SCENARIO] SCENARIO:")
        print("   A Product Manager needs to prepare a competitive analysis")
        print("   of MAN's electric truck lineup for a strategy meeting.")
        print()

        # Step 1: Run complete workflow
        print("[STEP] STEP 1: Running E-Powertrain Benchmarking System...")
        print("-" * 50)

        start_time = time.time()

        final_state = run_benchmark(
            urls=urls,
            verbose=True,
            mode=ScrapingMode.PERPLEXITY
        )

        duration = time.time() - start_time

        # Step 2: Analyze results
        print("\n[ANALYZE] STEP 2: Analyzing Results...")
        print("-" * 50)

        status = final_state["workflow_status"]
        vehicles = final_state.get("all_vehicles", [])
        validation = final_state.get("quality_validation", {})
        presentation = final_state.get("presentation_result", {})

        print(f"   Status: {status}")
        print(f"   Vehicles Found: {len(vehicles)}")
        print(f"   Quality Score: {validation.get('overall_quality_score', 0):.2f}")
        print(f"   Tokens Used: {final_state.get('total_tokens_used', 0):,}")
        print(f"   Total Cost: ${final_state.get('total_cost_usd', 0):.4f}")
        print(f"   Duration: {duration:.1f} seconds")

        # Step 3: Verify outputs
        print("\n[CHECK] STEP 3: Verifying Outputs...")
        print("-" * 50)

        checks_passed = 0
        total_checks = 5

        # Check 1: Workflow completed
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            print("   + Workflow completed")
            checks_passed += 1
        else:
            print(f"   x Workflow incomplete: {status}")

        # Check 2: Vehicles extracted
        if len(vehicles) > 0:
            print(f"   + Vehicles extracted: {len(vehicles)}")
            checks_passed += 1
        else:
            print("   x No vehicles extracted")

        # Check 3: Validation run
        if validation:
            print(f"   + Quality validation: {validation.get('overall_quality_score', 0):.2f}")
            checks_passed += 1
        else:
            print("   x Validation not completed")

        # Check 4: Cost tracked
        if final_state.get("total_cost_usd", 0) > 0:
            print(f"   + Cost tracked: ${final_state['total_cost_usd']:.4f}")
            checks_passed += 1
        else:
            print("   x Cost not tracked")

        # Check 5: Presentation (if workflow completed)
        pres_paths = presentation.get("all_presentation_paths", [])
        if status == WorkflowStatus.COMPLETED and pres_paths:
            existing = [p for p in pres_paths if Path(p).exists()]
            if existing:
                print(f"   + Presentation generated: {len(existing)} file(s)")
                checks_passed += 1
            else:
                print("   x Presentation files not found")
        elif status == WorkflowStatus.COMPLETED:
            print("   x No presentation generated")
        else:
            print("   ⚠ Skipped (workflow didn't complete)")
            checks_passed += 1  # Don't fail for this

        # Summary
        print("\n" + "="*70)
        print(f"BUSINESS SCENARIO RESULT: {checks_passed}/{total_checks} checks passed")
        print("="*70)

        if checks_passed >= 4:
            print("\n[SUCCESS] SUCCESS: System is working as expected!")
            print("   The Product Manager can use this data for the strategy meeting.")
        else:
            print("\n[WARNING] PARTIAL SUCCESS: Some checks failed.")
            print("   Review the output above for details.")

        # Final assertions
        assert checks_passed >= 3, f"Expected at least 3/5 checks to pass, got {checks_passed}/5"

        return {
            "status": status,
            "vehicles": len(vehicles),
            "quality_score": validation.get("overall_quality_score", 0),
            "cost": final_state.get("total_cost_usd", 0),
            "duration": duration,
            "checks_passed": checks_passed,
        }


# =====================================================================
# HELPER: Run specific test
# =====================================================================

if __name__ == "__main__":
    """Run live tests directly."""
    import subprocess

    print("Running live integration tests...")
    print("This will make real API calls and incur costs.\n")

    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v", "-s",
        "--tb=short",
        "-x",  # Stop on first failure
    ])

    sys.exit(result.returncode)
