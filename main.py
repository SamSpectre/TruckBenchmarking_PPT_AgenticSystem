#!/usr/bin/env python3
"""
E-Powertrain Benchmarking System - Main Entry Point

Run the complete LangGraph workflow to:
1. Scrape OEM websites for e-powertrain specifications
2. Validate data quality
3. Generate PowerPoint presentations

Usage:
    python main.py                          # Use default URLs from src/inputs/urls.txt
    python main.py --urls URL1 URL2 URL3    # Specify URLs directly
    python main.py --file my_urls.txt       # Use custom URL file
    python main.py --stream                 # Show step-by-step progress
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph.runtime import main

if __name__ == "__main__":
    main()
