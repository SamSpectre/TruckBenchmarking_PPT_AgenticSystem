"""
Agents Package for E-Powertrain Benchmarking System

This package contains LangGraph node functions for each agent in the workflow.

Usage:
    from src.agents import scraping_node, validation_node, presentation_node

    # In LangGraph
    graph.add_node("scrape", scraping_node)
    graph.add_node("validate", validation_node)
    graph.add_node("present", presentation_node)
"""

# Import node functions for LangGraph
from src.agents.scraping_agent import (
    scraping_node,
    async_scraping_node,
    run_scraping,
)

from src.agents.quality_validator import (
    validation_node,
    async_validation_node,
    run_validation,
    validate_scraping_results,  # Backwards compatibility alias
    QualityValidator,
    RuleBasedValidator,
    ValidationConfig,
)

from src.agents.presentation_generator import (
    presentation_node,
    async_presentation_node,
    comparison_presentation_node,
    run_presentation_generation,
)


__all__ = [
    # Scraping Agent
    "scraping_node",
    "async_scraping_node",
    "run_scraping",

    # Quality Validator
    "validation_node",
    "async_validation_node",
    "run_validation",
    "validate_scraping_results",
    "QualityValidator",
    "RuleBasedValidator",
    "ValidationConfig",

    # Presentation Generator
    "presentation_node",
    "async_presentation_node",
    "comparison_presentation_node",
    "run_presentation_generation",
]
