"""
Quality Validator Agent for E-Powertrain Benchmarking System

HYBRID APPROACH:
1. Rule-based validation (fast, deterministic, always runs)
   - Required field checks
   - Data type validation
   - Range/bounds checking
   - Source quality scoring

2. LLM-based validation (optional, for semantic checks)
   - Value plausibility (does 5000 km range make sense?)
   - Cross-field consistency (fuel cell vehicle with no H2 tank?)
   - Data anomaly detection

Usage:
    validator = QualityValidator(use_llm=False)  # Rule-based only
    validator = QualityValidator(use_llm=True)   # Hybrid with LLM
    result = validator.validate(scraping_result)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import re


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation thresholds"""
    
    # Minimum scores to pass
    min_overall_score: float = 0.6
    min_completeness_score: float = 0.5
    min_source_quality_score: float = 0.3
    
    # Required fields for a valid vehicle
    required_vehicle_fields: List[str] = field(default_factory=lambda: [
        "vehicle_name",
        "oem_name",
        "source_url",
    ])
    
    # Important fields (affect completeness score)
    important_vehicle_fields: List[str] = field(default_factory=lambda: [
        "battery_capacity_kwh",
        "range_km",
        "motor_power_kw",
        "gvw_kg",
        "powertrain_type",
    ])
    
    # Reasonable bounds for numeric values
    value_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "battery_capacity_kwh": (10, 2000),      # 10 kWh to 2 MWh
        "range_km": (50, 1500),                   # 50 to 1500 km
        "motor_power_kw": (50, 1500),             # 50 to 1500 kW
        "gvw_kg": (2000, 100000),                 # 2 to 100 tonnes
        "dc_charging_kw": (20, 2000),             # 20 kW to 2 MW
        "motor_torque_nm": (100, 50000),          # 100 to 50000 Nm
    })


# =====================================================================
# RULE-BASED VALIDATOR
# =====================================================================

class RuleBasedValidator:
    """
    Fast, deterministic validation using predefined rules.
    No API calls, no cost, always available.
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
    
    def validate_vehicle(self, vehicle: Dict) -> Dict:
        """Validate a single vehicle's data"""
        issues = []
        warnings = []
        field_scores = {}
        
        # Check required fields
        for field_name in self.config.required_vehicle_fields:
            if not vehicle.get(field_name):
                issues.append(f"Missing required field: {field_name}")
                field_scores[field_name] = 0.0
            else:
                field_scores[field_name] = 1.0
        
        # Check important fields
        for field_name in self.config.important_vehicle_fields:
            value = vehicle.get(field_name)
            if value is None:
                warnings.append(f"Missing important field: {field_name}")
                field_scores[field_name] = 0.0
            else:
                field_scores[field_name] = 1.0
                
                # Check bounds for numeric fields
                if field_name in self.config.value_bounds:
                    min_val, max_val = self.config.value_bounds[field_name]
                    if isinstance(value, (int, float)):
                        if value < min_val or value > max_val:
                            warnings.append(
                                f"Suspicious value for {field_name}: {value} "
                                f"(expected {min_val}-{max_val})"
                            )
                            field_scores[field_name] = 0.5
        
        # Calculate completeness
        total_fields = len(self.config.required_vehicle_fields) + len(self.config.important_vehicle_fields)
        completeness = sum(field_scores.values()) / total_fields if total_fields > 0 else 0
        
        return {
            "vehicle_name": vehicle.get("vehicle_name", "Unknown"),
            "is_valid": len(issues) == 0,
            "completeness_score": completeness,
            "issues": issues,
            "warnings": warnings,
            "field_scores": field_scores,
        }
    
    def validate_source_quality(self, scraping_result: Dict) -> Dict:
        """Validate source quality based on citations"""
        official = scraping_result.get("official_citations", [])
        third_party = scraping_result.get("third_party_citations", [])
        oem_url = scraping_result.get("oem_url", "")
        
        # Extract domain
        domain = ""
        if "://" in oem_url:
            domain = oem_url.split("/")[2].replace("www.", "")
        
        # Count official citations matching domain
        matching_official = sum(1 for c in official if domain and domain in str(c))
        
        # Calculate source quality score
        total_citations = len(official) + len(third_party)
        if total_citations == 0:
            score = 0.0
            assessment = "No citations found"
        elif matching_official >= 2:
            score = 1.0
            assessment = "Multiple official sources confirmed"
        elif matching_official == 1:
            score = 0.7
            assessment = "Single official source"
        elif len(official) > 0:
            score = 0.5
            assessment = "Official citations but domain mismatch"
        else:
            score = 0.2
            assessment = "Only third-party sources"
        
        return {
            "score": score,
            "official_count": len(official),
            "third_party_count": len(third_party),
            "matching_domain_count": matching_official,
            "assessment": assessment,
        }
    
    def check_consistency(self, vehicles: List[Dict]) -> Dict:
        """Check for consistency across vehicles from same OEM"""
        issues = []
        
        if len(vehicles) < 2:
            return {"score": 1.0, "issues": []}
        
        # Check OEM name consistency
        oem_names = set(v.get("oem_name", "").lower() for v in vehicles)
        if len(oem_names) > 1:
            issues.append(f"Inconsistent OEM names: {oem_names}")
        
        # Check for duplicate vehicle names
        vehicle_names = [v.get("vehicle_name", "") for v in vehicles]
        duplicates = set(n for n in vehicle_names if vehicle_names.count(n) > 1)
        if duplicates:
            issues.append(f"Duplicate vehicle names: {duplicates}")
        
        score = 1.0 - (len(issues) * 0.2)
        return {"score": max(0, score), "issues": issues}
    
    def validate(self, scraping_result: Dict) -> Dict:
        """
        Main validation entry point for rule-based validation.
        
        Returns QualityValidationResult dict.
        """
        vehicles = scraping_result.get("vehicles", [])
        
        # Validate each vehicle
        vehicle_results = [self.validate_vehicle(v) for v in vehicles]
        
        # Calculate scores
        valid_vehicles = [r for r in vehicle_results if r["is_valid"]]
        accuracy_score = len(valid_vehicles) / len(vehicle_results) if vehicle_results else 0
        
        completeness_scores = [r["completeness_score"] for r in vehicle_results]
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        source_quality = self.validate_source_quality(scraping_result)
        consistency = self.check_consistency(vehicles)
        
        # Overall score (weighted average)
        overall_score = (
            avg_completeness * 0.35 +
            accuracy_score * 0.25 +
            consistency["score"] * 0.15 +
            source_quality["score"] * 0.25
        )
        
        # Collect all issues
        all_missing_fields = []
        all_suspicious_values = []
        low_quality_vehicles = []
        
        for result in vehicle_results:
            all_missing_fields.extend(result["issues"])
            if result["warnings"]:
                for w in result["warnings"]:
                    if "Suspicious" in w:
                        all_suspicious_values.append({
                            "vehicle": result["vehicle_name"],
                            "issue": w
                        })
            if result["completeness_score"] < 0.5:
                low_quality_vehicles.append(result["vehicle_name"])
        
        # Generate recommendation
        passes = overall_score >= self.config.min_overall_score
        if passes:
            recommendation = "Data quality acceptable. Proceed to presentation generation."
        elif overall_score >= 0.4:
            recommendation = "Data quality marginal. Consider retry with enhanced extraction."
        else:
            recommendation = "Data quality too low. Retry required."
        
        # Generate retry suggestions
        retry_suggestions = []
        if avg_completeness < 0.5:
            retry_suggestions.append("Request more specific technical specifications")
        if source_quality["score"] < 0.5:
            retry_suggestions.append("Focus extraction on official website pages")
        if consistency["issues"]:
            retry_suggestions.append("Verify OEM and vehicle naming consistency")
        
        return {
            "overall_quality_score": round(overall_score, 3),
            "passes_threshold": passes,
            "completeness_score": round(avg_completeness, 3),
            "accuracy_score": round(accuracy_score, 3),
            "consistency_score": round(consistency["score"], 3),
            "source_quality_score": round(source_quality["score"], 3),
            "missing_fields": list(set(all_missing_fields)),
            "suspicious_values": all_suspicious_values,
            "low_quality_vehicles": low_quality_vehicles,
            "recommendation": recommendation,
            "retry_suggestions": retry_suggestions,
            "validation_timestamp": datetime.now().isoformat(),
            # Detailed breakdowns
            "_vehicle_results": vehicle_results,
            "_source_quality_details": source_quality,
            "_consistency_details": consistency,
        }


# =====================================================================
# LLM-BASED VALIDATOR (Optional Enhancement)
# =====================================================================

class LLMValidator:
    """
    LLM-powered semantic validation.
    Use for deeper analysis when rule-based validation passes
    but you want additional confidence.
    
    Cost: ~$0.01-0.02 per validation (depending on vehicle count)
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(model=self.model, temperature=0)
            except ImportError:
                raise ImportError("langchain_openai required for LLM validation")
        return self._client
    
    def validate_plausibility(self, vehicles: List[Dict]) -> Dict:
        """
        Use LLM to check if vehicle specifications are plausible.
        
        Catches things like:
        - Unrealistic combinations (fuel cell vehicle with 1000 kWh battery)
        - Typos in values (3000 km range instead of 300 km)
        - Missing logical requirements (H2 tank without fuel cell)
        """
        if not vehicles:
            return {"score": 1.0, "issues": [], "llm_used": False}
        
        # Format vehicles for prompt
        vehicle_summaries = []
        for v in vehicles[:3]:  # Limit to 3 to control costs
            summary = f"""
Vehicle: {v.get('vehicle_name', 'Unknown')}
- Powertrain: {v.get('powertrain_type', 'Unknown')}
- Battery: {v.get('battery_capacity_kwh', 'N/A')} kWh
- Range: {v.get('range_km', 'N/A')} km
- Motor Power: {v.get('motor_power_kw', 'N/A')} kW
- GVW: {v.get('gvw_kg', 'N/A')} kg
- DC Charging: {v.get('dc_charging_kw', 'N/A')} kW
"""
            additional = v.get('additional_specs', {})
            if additional:
                if additional.get('fuel_cell_kw'):
                    summary += f"- Fuel Cell: {additional['fuel_cell_kw']} kW\n"
                if additional.get('h2_tank_kg'):
                    summary += f"- H2 Tank: {additional['h2_tank_kg']} kg\n"
            vehicle_summaries.append(summary)
        
        prompt = f"""You are a commercial vehicle engineer validating scraped technical specifications.

Review these vehicle specifications for plausibility:

{chr(10).join(vehicle_summaries)}

Check for:
1. Unrealistic values (e.g., 5000 km range, 50 kg battery)
2. Inconsistent combinations (e.g., fuel cell without H2 tank)
3. Likely data errors or typos
4. Missing critical specs for the powertrain type

Respond in JSON format:
{{
    "plausibility_score": 0.0-1.0,
    "issues": ["issue 1", "issue 2"],
    "confidence": "high/medium/low"
}}

Be concise. Only flag clear issues, not minor gaps."""

        try:
            response = self.client.invoke(prompt)
            import json
            result = json.loads(response.content)
            result["llm_used"] = True
            return result
        except Exception as e:
            return {
                "score": 0.5,
                "issues": [f"LLM validation failed: {str(e)}"],
                "llm_used": False,
                "error": str(e)
            }


# =====================================================================
# MAIN VALIDATOR CLASS
# =====================================================================

class QualityValidator:
    """
    Main quality validator combining rule-based and optional LLM validation.
    
    Usage:
        validator = QualityValidator(use_llm=False)
        result = validator.validate(scraping_result)
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        config: ValidationConfig = None,
        llm_model: str = "gpt-4o-mini"
    ):
        self.use_llm = use_llm
        self.rule_validator = RuleBasedValidator(config)
        self.llm_validator = LLMValidator(llm_model) if use_llm else None
    
    def validate(self, scraping_result: Dict) -> Dict:
        """
        Validate scraping result and return QualityValidationResult.
        
        Always runs rule-based validation.
        Optionally runs LLM validation if enabled.
        """
        # Always run rule-based validation
        result = self.rule_validator.validate(scraping_result)
        
        # Optionally enhance with LLM
        if self.use_llm and self.llm_validator:
            vehicles = scraping_result.get("vehicles", [])
            llm_result = self.llm_validator.validate_plausibility(vehicles)
            
            # Merge LLM results
            if llm_result.get("llm_used"):
                llm_score = llm_result.get("plausibility_score", 1.0)
                
                # Adjust overall score based on LLM findings
                original_score = result["overall_quality_score"]
                result["overall_quality_score"] = round(
                    (original_score * 0.7 + llm_score * 0.3), 3
                )
                
                # Add LLM issues
                if llm_result.get("issues"):
                    result["suspicious_values"].extend([
                        {"vehicle": "LLM Check", "issue": issue}
                        for issue in llm_result["issues"]
                    ])
                
                result["_llm_validation"] = llm_result
                
                # Update passes_threshold
                result["passes_threshold"] = (
                    result["overall_quality_score"] >= 
                    self.rule_validator.config.min_overall_score
                )
        
        return result


# =====================================================================
# LANGGRAPH NODE FUNCTION
# =====================================================================

def validation_node(state: "BenchmarkingState") -> Dict[str, Any]:
    """
    LangGraph node for quality validation.

    Takes scraping results from state, validates them, and returns updated state.

    Args:
        state: Current BenchmarkingState with scraping_results

    Returns:
        Dict with state updates (LangGraph merges this with current state)
    """
    from src.state.state import WorkflowStatus, AgentType

    # Update status
    updates = {
        "workflow_status": WorkflowStatus.VALIDATING,
        "current_agent": AgentType.VALIDATOR,
    }

    scraping_results = state.get("scraping_results", [])

    if not scraping_results:
        return {
            **updates,
            "workflow_status": WorkflowStatus.QUALITY_FAILED,
            "quality_validation": {
                "overall_quality_score": 0.0,
                "passes_threshold": False,
                "completeness_score": 0.0,
                "accuracy_score": 0.0,
                "consistency_score": 0.0,
                "source_quality_score": 0.0,
                "missing_fields": ["No scraping results to validate"],
                "suspicious_values": [],
                "low_quality_vehicles": [],
                "recommendation": "No data available for validation",
                "retry_suggestions": ["Run scraping first"],
                "validation_timestamp": datetime.now().isoformat(),
            },
            "errors": state.get("errors", []) + ["No scraping results to validate"],
        }

    # Check if LLM validation is enabled (from settings)
    try:
        from src.config.settings import settings
        use_llm = getattr(settings, 'use_llm_validation', False)
    except Exception:
        use_llm = False

    # Run validation
    validation_result = run_validation(scraping_results, use_llm=use_llm)

    # Determine next status based on validation
    if validation_result.get("passes_threshold", False):
        workflow_status = WorkflowStatus.GENERATING_PRESENTATION
    else:
        # Check if retries available
        retries_remaining = state.get("total_retries_remaining", 0)
        if retries_remaining > 0:
            workflow_status = WorkflowStatus.RETRYING
        else:
            workflow_status = WorkflowStatus.QUALITY_FAILED

    return {
        **updates,
        "workflow_status": workflow_status,
        "quality_validation": validation_result,
        "retry_count": state.get("retry_count", 0) + (1 if workflow_status == WorkflowStatus.RETRYING else 0),
        "total_retries_remaining": max(0, state.get("total_retries_remaining", 0) - (1 if workflow_status == WorkflowStatus.RETRYING else 0)),
    }


async def async_validation_node(state: "BenchmarkingState") -> Dict[str, Any]:
    """Async version of validation node."""
    return validation_node(state)


# =====================================================================
# STANDALONE VALIDATION FUNCTION
# =====================================================================

def run_validation(
    scraping_results: List[Dict],
    use_llm: bool = False
) -> Dict:
    """
    Validate all scraping results and return aggregated quality validation.

    This is the core validation function called by the validation_node.
    
    Args:
        scraping_results: List of ScrapingResult dicts
        use_llm: Whether to use LLM for enhanced validation
    
    Returns:
        QualityValidationResult dict
    """
    validator = QualityValidator(use_llm=use_llm)
    
    all_results = []
    all_vehicles = []
    
    for sr in scraping_results:
        result = validator.validate(sr)
        all_results.append(result)
        all_vehicles.extend(sr.get("vehicles", []))
    
    # Aggregate scores
    if all_results:
        avg_overall = sum(r["overall_quality_score"] for r in all_results) / len(all_results)
        avg_completeness = sum(r["completeness_score"] for r in all_results) / len(all_results)
        avg_accuracy = sum(r["accuracy_score"] for r in all_results) / len(all_results)
        avg_consistency = sum(r["consistency_score"] for r in all_results) / len(all_results)
        avg_source = sum(r["source_quality_score"] for r in all_results) / len(all_results)
        
        all_passes = all(r["passes_threshold"] for r in all_results)
        
        # Aggregate issues
        all_missing = []
        all_suspicious = []
        all_low_quality = []
        all_suggestions = []
        
        for r in all_results:
            all_missing.extend(r["missing_fields"])
            all_suspicious.extend(r["suspicious_values"])
            all_low_quality.extend(r["low_quality_vehicles"])
            all_suggestions.extend(r["retry_suggestions"])
        
        if all_passes:
            recommendation = f"All {len(all_results)} OEM(s) passed validation. Ready for presentation."
        else:
            failed = sum(1 for r in all_results if not r["passes_threshold"])
            recommendation = f"{failed}/{len(all_results)} OEM(s) failed validation. Review needed."
        
        return {
            "overall_quality_score": round(avg_overall, 3),
            "passes_threshold": all_passes,
            "completeness_score": round(avg_completeness, 3),
            "accuracy_score": round(avg_accuracy, 3),
            "consistency_score": round(avg_consistency, 3),
            "source_quality_score": round(avg_source, 3),
            "missing_fields": list(set(all_missing)),
            "suspicious_values": all_suspicious,
            "low_quality_vehicles": list(set(all_low_quality)),
            "recommendation": recommendation,
            "retry_suggestions": list(set(all_suggestions)),
            "validation_timestamp": datetime.now().isoformat(),
            "_per_oem_results": all_results,
        }
    
    return {
        "overall_quality_score": 0.0,
        "passes_threshold": False,
        "completeness_score": 0.0,
        "accuracy_score": 0.0,
        "consistency_score": 0.0,
        "source_quality_score": 0.0,
        "missing_fields": ["No data to validate"],
        "suspicious_values": [],
        "low_quality_vehicles": [],
        "recommendation": "No scraping results to validate",
        "retry_suggestions": ["Ensure scraper ran successfully"],
        "validation_timestamp": datetime.now().isoformat(),
    }


# =====================================================================
# CLI / TEST
# =====================================================================

if __name__ == "__main__":
    # Test with mock data
    mock_scraping_result = {
        "oem_name": "Test OEM",
        "oem_url": "https://www.testoem.com",
        "vehicles": [
            {
                "vehicle_name": "Test Truck A",
                "oem_name": "Test OEM",
                "source_url": "https://www.testoem.com/truck-a",
                "battery_capacity_kwh": 400,
                "range_km": 500,
                "motor_power_kw": 350,
                "gvw_kg": 40000,
                "powertrain_type": "BEV",
            },
            {
                "vehicle_name": "Test Truck B",
                "oem_name": "Test OEM",
                "source_url": "https://www.testoem.com/truck-b",
                "battery_capacity_kwh": None,  # Missing
                "range_km": 5000,  # Suspicious - too high!
                "motor_power_kw": 250,
                "powertrain_type": "BEV",
                # Missing gvw_kg
            },
        ],
        "official_citations": [
            "https://www.testoem.com/specs",
            "https://www.testoem.com/trucks"
        ],
        "third_party_citations": [],
        "source_compliance_score": 0.8,
    }
    
    print("Testing Quality Validator")
    print("=" * 60)
    
    # Test rule-based only
    print("\n1. Rule-Based Validation:")
    print("-" * 40)
    validator = QualityValidator(use_llm=False)
    result = validator.validate(mock_scraping_result)
    
    print(f"Overall Score: {result['overall_quality_score']}")
    print(f"Passes: {result['passes_threshold']}")
    print(f"Completeness: {result['completeness_score']}")
    print(f"Source Quality: {result['source_quality_score']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if result['suspicious_values']:
        print(f"\nSuspicious Values:")
        for sv in result['suspicious_values']:
            print(f"  - {sv}")
    
    if result['retry_suggestions']:
        print(f"\nRetry Suggestions:")
        for s in result['retry_suggestions']:
            print(f"  - {s}")
    
    print("\n" + "=" * 60)
    print("Validation test complete!")


# =====================================================================
# BACKWARDS COMPATIBILITY ALIAS
# =====================================================================

# Alias for backwards compatibility
validate_scraping_results = run_validation