"""
Tools Package for E-Powertrain Benchmarking System

This package contains tool implementations used by agents.

Usage:
    from src.tools import EPowertrainExtractor, scrape_oem_urls
    from src.tools import generate_all_presentations, generate_oem_presentation
"""

# Scraper Tools
from src.tools.scraper import (
    EPowertrainExtractor,
    SpecificationParser,
    ScraperConfig,
    scrape_oem_urls,
)

# Presentation Tools
from src.tools.ppt_generator import (
    # Single OEM functions
    generate_presentation,
    generate_oem_presentation,
    transform_scraping_result_to_oem_profile,
    transform_vehicle_to_iaa_product,

    # Batch functions
    generate_all_presentations,
    generate_comparison_presentation,

    # Helpers
    SHAPE_IDS,
    PRODUCT_TABLE_ROWS,
)


__all__ = [
    # Scraper
    "EPowertrainExtractor",
    "SpecificationParser",
    "ScraperConfig",
    "scrape_oem_urls",

    # Presentation - Single
    "generate_presentation",
    "generate_oem_presentation",
    "transform_scraping_result_to_oem_profile",
    "transform_vehicle_to_iaa_product",

    # Presentation - Batch
    "generate_all_presentations",
    "generate_comparison_presentation",

    # Constants
    "SHAPE_IDS",
    "PRODUCT_TABLE_ROWS",
]
