"""
E-Powertrain Web Scraper Tool

Uses Crawl4AI for webpage fetching + LLM for structured data extraction.

THREE-PHASE INTELLIGENT SCRAPING:
  1. Link Discovery: LLM identifies EV spec page links from starting URL
  2. Targeted Crawl: Crawl up to 5 spec pages per OEM
  3. LLM Extraction: Pydantic schema extraction for high accuracy

UPDATED: Now uses intelligent navigation to find ALL EV spec pages.
"""

import requests
import json
import os
import re
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# Pydantic for schema-based extraction
from pydantic import BaseModel, Field

# Crawl4AI imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

load_dotenv()


# =====================================================================
# CONFIGURATION
# =====================================================================

class ScraperConfig:
    """Scraper configuration"""
    API_URL = "https://api.perplexity.ai/chat/completions"
    DEFAULT_MODEL = "sonar-pro"
    MAX_TOKENS = 8000
    TEMPERATURE = 0.1
    WAIT_BETWEEN_URLS = 8  # seconds
    MAX_CONTENT_LENGTH = 40000  # Max chars to send for extraction (increased for spec-heavy pages)

    # Intelligent Navigation Settings
    MAX_PAGES_PER_OEM = 12  # Maximum pages to crawl per OEM (increased for better coverage)
    LLM_EXTRACTION_MODEL = "openai/gpt-4o-mini"  # Fast & cheap LLM for extraction
    ENABLE_INTELLIGENT_NAVIGATION = True  # Use LLM-guided navigation

    # Auto-Fallback Settings
    AUTO_FALLBACK_ENABLED = True  # Automatically switch to intelligent mode if data quality is low
    MIN_COMPLETENESS_THRESHOLD = 0.4  # Minimum average completeness to accept (0.0-1.0)
    MIN_VEHICLES_WITH_DATA = 0.5  # At least 50% of vehicles must have >0 completeness
    FALLBACK_ON_ZERO_VEHICLES = True  # Always fallback if no vehicles extracted

    # Vehicle category filtering (for truck pages, exclude buses)
    VEHICLE_CATEGORY_FILTER = "truck"  # Set to None to include all, or "truck", "bus", etc.


# =====================================================================
# PYDANTIC MODELS FOR LLM EXTRACTION
# =====================================================================

class VehicleSpec(BaseModel):
    """Pydantic schema for vehicle specifications - used by LLM extraction"""
    vehicle_name: str = Field(description="Full model name (e.g., 'MAN eTGX 4x2')")
    battery_capacity_kwh: Optional[float] = Field(None, description="Battery capacity in kWh")
    battery_voltage_v: Optional[float] = Field(None, description="Battery voltage in volts")
    motor_power_kw: Optional[float] = Field(None, description="Motor power in kW")
    motor_torque_nm: Optional[float] = Field(None, description="Motor torque in Nm")
    range_km: Optional[float] = Field(None, description="Driving range in km")
    dc_charging_kw: Optional[float] = Field(None, description="DC fast charging power in kW")
    charging_time_minutes: Optional[float] = Field(None, description="Charging time in minutes")
    gvw_kg: Optional[float] = Field(None, description="Gross vehicle weight in kg")
    payload_capacity_kg: Optional[float] = Field(None, description="Payload capacity in kg")
    powertrain_type: Optional[str] = Field(None, description="BEV, FCEV, or PHEV")
    energy_consumption_kwh_100km: Optional[float] = Field(None, description="Energy consumption kWh/100km")


class VehicleSpecList(BaseModel):
    """List of vehicle specifications extracted from a page"""
    vehicles: List[VehicleSpec] = Field(default_factory=list)


class SpecPageLink(BaseModel):
    """A link that leads to an EV specification page"""
    url: str = Field(description="Full URL to the spec page")
    vehicle_name: Optional[str] = Field(None, description="Vehicle name if identifiable from link text")
    confidence: float = Field(default=0.8, description="Confidence that this link leads to a spec page")


class SpecPageLinks(BaseModel):
    """Links identified as leading to EV specification pages"""
    spec_links: List[SpecPageLink] = Field(default_factory=list)


# =====================================================================
# WEB CONTENT FETCHER (Stage 1: Crawl4AI)
# =====================================================================

class WebContentFetcher:
    """
    Fetches webpage content using Crawl4AI.
    Handles JavaScript rendering, dynamic content, and returns clean markdown.
    """

    def __init__(self, headless: bool = True, verbose: bool = False):
        self.browser_config = BrowserConfig(
            headless=headless,
            verbose=verbose
        )
        # Fix Windows console encoding for Crawl4AI's rich output
        self._fix_windows_encoding()

    def _fix_windows_encoding(self):
        """Fix Windows console encoding to handle Unicode output from Crawl4AI"""
        import sys
        try:
            # Reconfigure stdout/stderr to use UTF-8 with error replacement
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass  # Ignore if reconfigure not available

    async def fetch_async(self, url: str) -> Dict[str, Any]:
        """
        Fetch URL and return content with metadata.

        Returns:
            Dict with:
                - success: bool
                - markdown: str (clean content)
                - html: str (raw HTML)
                - title: str
                - error: str (if failed)
        """
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        word_count_threshold=10,
                        # Note: remove_overlay_elements=True strips too much on some sites
                    )
                )

            if result.success:
                # Convert markdown object to string (Crawl4AI returns StringCompatibleMarkdown)
                markdown_content = str(result.markdown) if result.markdown else ''
                return {
                    'success': True,
                    'markdown': markdown_content,
                    'html': result.html or '',
                    'title': getattr(result, 'title', ''),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'markdown': '',
                    'html': '',
                    'title': '',
                    'error': getattr(result, 'error_message', 'Unknown error')
                }
        except Exception as e:
            return {
                'success': False,
                'markdown': '',
                'html': '',
                'title': '',
                'error': str(e)
            }

    def fetch(self, url: str) -> Dict[str, Any]:
        """Synchronous wrapper for fetch_async"""
        return asyncio.run(self.fetch_async(url))


# =====================================================================
# INTELLIGENT SCRAPER (Three-Phase LLM-Guided Navigation)
# =====================================================================

class IntelligentScraper:
    """
    Three-phase intelligent scraper:
    1. Discover spec page links using LLM
    2. Crawl relevant pages (max 5)
    3. Extract structured specs using LLM + Pydantic schema

    This ensures ALL EV vehicles on a website are found and extracted.
    """

    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set for intelligent scraping")

        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self.max_pages = ScraperConfig.MAX_PAGES_PER_OEM
        self._fix_windows_encoding()

    def _fix_windows_encoding(self):
        """Fix Windows console encoding for Unicode output"""
        import sys
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    async def extract_all_vehicles(self, starting_url: str) -> Dict[str, Any]:
        """
        Main entry point - handles full extraction pipeline.

        Args:
            starting_url: OEM overview page URL

        Returns:
            Dict with vehicles, pages_crawled, spec_urls_found
        """
        print(f"\n[IntelligentScraper] Starting intelligent extraction from: {starting_url}")

        # Phase 1: Discover spec page links
        print("[Phase 1] Discovering spec page links...")
        spec_urls = await self._discover_spec_links(starting_url)
        print(f"[Phase 1] Found {len(spec_urls)} potential spec pages")

        # Always include the starting URL as a spec page
        if starting_url not in spec_urls:
            spec_urls.insert(0, starting_url)

        # Phase 2: Crawl spec pages (limit to max_pages)
        print(f"[Phase 2] Crawling up to {self.max_pages} spec pages...")
        crawl_results = await self._crawl_pages(spec_urls[:self.max_pages])
        print(f"[Phase 2] Successfully crawled {len(crawl_results)} pages")

        # Phase 3: Extract structured specs from each page
        print("[Phase 3] Extracting vehicle specifications with LLM...")
        all_vehicles = []
        extraction_details = []

        for url, content in crawl_results:
            vehicles = await self._extract_specs(url, content)
            print(f"  - {url}: {len(vehicles)} vehicles extracted")
            all_vehicles.extend(vehicles)
            extraction_details.append({
                'url': url,
                'vehicles_found': len(vehicles),
                'content_length': len(content)
            })

        # Deduplicate by vehicle name
        unique_vehicles = self._deduplicate(all_vehicles)
        print(f"[Phase 3] Total unique vehicles: {len(unique_vehicles)}")

        return {
            'vehicles': unique_vehicles,
            'pages_crawled': len(crawl_results),
            'spec_urls_found': spec_urls[:self.max_pages],
            'extraction_details': extraction_details,
        }

    async def _discover_spec_links(self, url: str) -> List[str]:
        """
        Phase 1: Use LLM to find spec page links on the starting page.

        Fetches the page, extracts all internal links, and uses LLM to
        identify which links likely lead to EV specification pages.
        """
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(word_count_threshold=10)
                )

            if not result.success:
                print(f"  [Phase 1] Failed to fetch page: {getattr(result, 'error_message', 'Unknown')}")
                return [url]  # Return starting URL as fallback

            # Extract base domain for filtering
            parsed = urlparse(url)
            base_domain = f"{parsed.scheme}://{parsed.netloc}"

            # Get all links from the page
            internal_links = []
            if hasattr(result, 'links') and result.links:
                internal_links = result.links.get('internal', [])

            # Also extract links from markdown using regex
            markdown_content = str(result.markdown) if result.markdown else ''
            link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
            for match in link_pattern.finditer(markdown_content):
                link_text, link_url = match.groups()
                if link_url.startswith('/'):
                    link_url = urljoin(base_domain, link_url)
                if link_url.startswith(base_domain):
                    internal_links.append({'href': link_url, 'text': link_text})

            # Also look for href patterns in HTML
            if hasattr(result, 'html') and result.html:
                href_pattern = re.compile(r'href=["\']([^"\']+)["\']')
                for href in href_pattern.findall(result.html):
                    if href.startswith('/'):
                        full_url = urljoin(base_domain, href)
                        if full_url.startswith(base_domain):
                            internal_links.append({'href': full_url, 'text': ''})

            if not internal_links:
                print("  [Phase 1] No internal links found, using starting URL only")
                return [url]

            # Use LLM to filter to spec pages
            spec_urls = await self._filter_spec_links_with_llm(internal_links, url, base_domain)

            return spec_urls

        except Exception as e:
            print(f"  [Phase 1] Error discovering links: {e}")
            return [url]

    async def _filter_spec_links_with_llm(
        self,
        links: List[Dict],
        source_url: str,
        base_domain: str
    ) -> List[str]:
        """Use LLM to identify which links lead to EV spec pages"""

        # Deduplicate links and format for LLM
        seen_urls = set()
        link_list = []
        for link in links:
            href = link.get('href', '') if isinstance(link, dict) else str(link)
            if href and href not in seen_urls and base_domain in href:
                seen_urls.add(href)
                text = link.get('text', '') if isinstance(link, dict) else ''
                link_list.append(f"- {href} (text: {text})")

        if not link_list:
            return [source_url]

        # Limit to prevent token overflow
        link_text = '\n'.join(link_list[:100])

        try:
            llm_strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider=ScraperConfig.LLM_EXTRACTION_MODEL,
                    api_token=self.api_key
                ),
                schema=SpecPageLinks.model_json_schema(),
                extraction_type="schema",
                instruction=f"""Analyze these links from an OEM commercial vehicle website ({source_url}).

Your task: Find links to pages with TECHNICAL SPECIFICATIONS for electric trucks.

**CRITICAL: PRIORITIZE INDIVIDUAL MODEL PAGES OVER GENERAL INFO PAGES**

PRIORITY 1 - INDIVIDUAL VEHICLE MODEL PAGES (MOST IMPORTANT):
- URLs with specific EV model names: eTGX, eTGS, eTGL, eActros, FH Electric, FM Electric, etc.
- Pattern: /all-models/*, /the-man-etgx/*, /the-man-etgs/*, /the-man-etgl/*, /models/*
- Product detail pages for EACH electric vehicle variant
- Example good URLs: ".../the-man-tgx/the-man-etgx/overview.html", ".../the-man-tgs/the-man-etgs/..."

PRIORITY 2 - Technical Data/Spec Pages:
- URLs with: /spec, /technical, /tech-data, /data-sheet, technische-daten
- Pages titled "Technical Data", "Specifications", "Key Facts"
- Data sheet or brochure pages

PRIORITY 3 - Electric Vehicle Overview Pages:
- URLs with: /electric-trucks/, /e-truck/, /ev/, /bev/
- Portfolio or lineup pages listing multiple EVs

**EXCLUDE (NEVER include these):**
- /charging-infrastructure, /charging-and-battery-management (general info, not vehicle specs)
- /battery-regulation, /regulation (legal/regulatory pages)
- /range-calculator, /ereadycheck (tools, not specs)
- News, press releases, blog posts
- /service/, /advice/, /consulting/ (support pages)
- /bus/, /lion-city (bus pages when looking for trucks)
- Careers, contact, legal, privacy pages
- Image galleries, videos, downloads
- Social media links

Available links:
{link_text}

Return UP TO 10 links that lead to ACTUAL VEHICLE SPECIFICATION PAGES with battery kWh, range km, motor kW, GVW data.
Prioritize individual model pages (eTGX, eTGS, eTGL separately) over generic overview pages.
""",
                input_format="markdown"
            )

            # Create minimal crawler run for extraction
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=source_url,
                    config=CrawlerRunConfig(
                        extraction_strategy=llm_strategy,
                        word_count_threshold=10
                    )
                )

            if result.extracted_content:
                try:
                    extracted = json.loads(result.extracted_content)
                    if isinstance(extracted, list) and extracted:
                        spec_links = extracted[0].get('spec_links', [])
                        urls = [link.get('url') for link in spec_links if link.get('url')]
                        if urls:
                            return urls
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"  [Phase 1] LLM extraction parse error: {e}")

        except Exception as e:
            print(f"  [Phase 1] LLM link filtering error: {e}")

        # Fallback: Use pattern matching
        return self._filter_links_by_pattern(list(seen_urls), source_url)

    def _filter_links_by_pattern(self, urls: List[str], source_url: str) -> List[str]:
        """Fallback: Filter links using URL patterns for technical spec pages"""

        # HIGHEST priority - Individual vehicle model pages (these have the real specs)
        model_page_patterns = [
            r'/the-man-etgx/', r'/the-man-etgs/', r'/the-man-etgl/',  # MAN specific
            r'/all-models/.*?/overview', r'/models/.*?e.*?/',  # Generic model pages
            r'eactros.*?overview', r'fh.*?electric.*?overview',  # Other OEMs
            r'/e-?actros/', r'/fh-?electric/', r'/fm-?electric/',
            r'/ecanter/', r'/e-?truck/', r'/electric-truck/',
        ]

        # High priority - Technical data pages
        tech_data_patterns = [
            r'/spec', r'/technical', r'/data-?sheet', r'/tech-?data',
            r'technische-daten', r'specifications', r'fiche-technique',
            r'key-?facts', r'highlights',
        ]

        # Medium priority - Generic electric vehicle pages
        ev_patterns = [
            r'/electric-trucks/', r'/ev/', r'/bev/', r'/fcev/',
            r'etg[xsl]', r'eactros', r'fh-?electric', r'vnr-?electric',
            r'fm-?electric', r'fmx-?electric', r'fe-?electric', r'fl-?electric',
            r'ecanter', r'eseries',
        ]

        # Low priority - overview pages
        overview_patterns = [
            r'/overview\.html$', r'/lineup', r'/portfolio',
        ]

        # EXPLICIT EXCLUSION patterns - these pages don't have real vehicle specs
        exclude_patterns = [
            r'/charging-infrastructure', r'/charging-and-battery-management',
            r'/battery-regulation', r'/regulation',
            r'/range-calculator', r'/ereadycheck', r'/configurator',
            r'/service/', r'/advice/', r'/consulting/', r'/getting-started',
            r'/fleet-management/', r'/general/',
            r'/bus/', r'/lion.*city', r'/city-bus',  # Exclude buses for truck pages
            r'/news', r'/press', r'/career', r'/job', r'/contact',
            r'/legal', r'/privacy', r'/cookie', r'/investor',
            r'/about-us', r'/history', r'/sustainability',
            r'\.pdf', r'\.jpg', r'\.png', r'/media/', r'/download',
            r'/video', r'/gallery', r'/image',
        ]

        model_pages = []
        tech_pages = []
        ev_pages = []
        overview_pages = []

        for url in urls:
            url_lower = url.lower()

            # Skip explicitly excluded pages
            if any(re.search(p, url_lower) for p in exclude_patterns):
                continue

            # Categorize by priority
            if any(re.search(p, url_lower) for p in model_page_patterns):
                model_pages.append(url)
            elif any(re.search(p, url_lower) for p in tech_data_patterns):
                tech_pages.append(url)
            elif any(re.search(p, url_lower) for p in ev_patterns):
                ev_pages.append(url)
            elif any(re.search(p, url_lower) for p in overview_patterns):
                overview_pages.append(url)

        # Combine in priority order: model pages first!
        spec_urls = model_pages + tech_pages + ev_pages + overview_pages

        # Always include source URL at the start
        if source_url not in spec_urls:
            spec_urls.insert(0, source_url)

        return spec_urls

    async def _crawl_pages(self, urls: List[str]) -> List[tuple]:
        """
        Phase 2: Crawl the discovered spec pages.

        Returns list of (url, markdown_content) tuples.
        """
        results = []
        crawled_urls = set()

        for url in urls:
            if url in crawled_urls:
                continue
            crawled_urls.add(url)

            try:
                async with AsyncWebCrawler(config=self.browser_config) as crawler:
                    result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(word_count_threshold=10)
                    )

                if result.success:
                    markdown_content = str(result.markdown) if result.markdown else ''
                    if len(markdown_content) > 100:  # Only include pages with content
                        results.append((url, markdown_content))

            except Exception as e:
                print(f"  [Phase 2] Error crawling {url}: {e}")

            if len(results) >= self.max_pages:
                break

        return results

    def _ensure_kg_weight(self, value: Optional[float]) -> Optional[float]:
        """Convert weight to kg if it looks like it's in tons"""
        if value is None:
            return None
        # Commercial trucks: 12,000-80,000 kg GVW typically
        # If value < 100, it's likely in tons
        if value < 100:
            return value * 1000
        return value

    async def _extract_specs(self, url: str, content: str) -> List[Dict]:
        """
        Phase 3: Extract structured vehicle specifications using OpenAI directly.

        Uses the already-fetched content (from Phase 2) and calls OpenAI API
        directly for high-accuracy structured extraction with JSON mode.
        Focuses on finding technical data tables and specification sections.
        """
        try:
            # Truncate content if too long
            if len(content) > ScraperConfig.MAX_CONTENT_LENGTH:
                content = content[:ScraperConfig.MAX_CONTENT_LENGTH]

            # Use OpenAI directly for extraction
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            extraction_prompt = f"""Extract electric commercial TRUCK specifications from this webpage.

WEBPAGE CONTENT:
---
{content}
---

**CRITICAL: Only extract data that is EXPLICITLY stated in the content above!**

LOOK FOR these data sources in the content:
1. Technical specification tables (rows with "Battery", "Range", "Power", "GVW")
2. Key facts or highlights sections with numbers
3. Product comparison tables with specs
4. Spec sheets or data sheets

For EACH electric TRUCK found (NOT buses), extract:
- vehicle_name: Full model name with variant (e.g., "MAN eTGX 4x2 semitrailer", "MAN eTGS 6x2")
- battery_capacity_kwh: Battery in kWh (use MAX if range given, e.g., "320-480 kWh" -> 480)
- battery_capacity_min_kwh: MIN battery if range given (e.g., "320-480 kWh" -> 320)
- motor_power_kw: Motor power in kW (convert: 1 PS = 0.735 kW, 1 hp = 0.746 kW)
- motor_torque_nm: Torque in Nm
- range_km: Driving range in km (use MAX if range, e.g., "500-750 km" -> 750)
- range_min_km: MIN range if given (e.g., "500-750 km" -> 500)
- dc_charging_kw: DC charging power in kW (CCS, MCS, Megawatt)
- charging_time_minutes: Charging time (to 80%)
- gvw_kg: Gross Vehicle Weight in KG (CONVERT: 28t/28 tons = 28000 kg, NOT 28!)
- payload_capacity_kg: Payload in kg
- powertrain_type: "BEV" | "FCEV" | "PHEV"
- available_configurations: Array like ["4x2", "6x2"]

**STRICT RULES - FOLLOW EXACTLY:**
1. ONLY extract specs that appear in the content - NEVER invent or estimate values
2. If a spec is not mentioned, use null - do NOT guess
3. Do NOT include buses (Lion's City, City Bus, etc.) - trucks only
4. WEIGHT: Always convert tons to kg (28t = 28000 kg, not 28)
5. If same model has variants (4x2 vs 6x2), extract each as separate vehicle
6. Each variant with different specs should be a separate entry
7. If content mentions "up to X kWh", extract X as max value
8. Use gvw_kg NOT gvwr_kg

**VALIDATION - Common MAN truck specs for reference (do NOT use these if not in content):**
- eTGX: 320-560 kWh battery, 500-750 km range
- eTGS: 400-480 kWh battery
- eTGL: ~160 kWh battery, ~235 km range
If extracted values are wildly different (e.g., 600 kWh for eTGX), verify in content!

Return ONLY valid JSON:
{{"vehicles": [
  {{"vehicle_name": "MAN eTGX 4x2 semitrailer", "battery_capacity_kwh": 480, "battery_capacity_min_kwh": 320, "range_km": 500, "gvw_kg": 44000, ...}},
  ...
]}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a precise technical data extraction expert for commercial vehicle specifications.

CRITICAL RULES:
1. ONLY extract data explicitly stated in the provided content
2. NEVER hallucinate, estimate, or infer values not directly stated
3. Use null for any specification not found in the content
4. Convert tons to kg (multiply by 1000)
5. For ranges (e.g., "320-480 kWh"), extract both min and max values
6. Exclude buses - extract trucks only

Your accuracy is critical for automotive benchmarking. Wrong data is worse than missing data."""},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum determinism
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            extracted = json.loads(result_text)

            vehicles_data = extracted.get('vehicles', [])
            vehicles = []
            for v in vehicles_data:
                # Get GVW with fallback for both field names
                gvw_raw = v.get('gvw_kg') or v.get('gvwr_kg')

                vehicle_dict = {
                    'vehicle_name': v.get('vehicle_name', 'Unknown'),
                    'source_url': url,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'battery_capacity_kwh': v.get('battery_capacity_kwh'),
                    'battery_capacity_min_kwh': v.get('battery_capacity_min_kwh'),
                    'battery_voltage_v': v.get('battery_voltage_v'),
                    'motor_power_kw': v.get('motor_power_kw'),
                    'motor_torque_nm': v.get('motor_torque_nm'),
                    'range_km': v.get('range_km'),
                    'range_min_km': v.get('range_min_km'),
                    'dc_charging_kw': v.get('dc_charging_kw'),
                    'charging_time_minutes': v.get('charging_time_minutes'),
                    'gvw_kg': self._ensure_kg_weight(gvw_raw),
                    'payload_capacity_kg': self._ensure_kg_weight(v.get('payload_capacity_kg')),
                    'powertrain_type': v.get('powertrain_type', 'BEV'),
                    'energy_consumption_kwh_100km': v.get('energy_consumption_kwh_100km'),
                    'available_configurations': v.get('available_configurations', []),
                }
                vehicles.append(vehicle_dict)
            return vehicles

        except Exception as e:
            print(f"  [Phase 3] Extraction error for {url}: {e}")

        return []

    def _get_source_quality_score(self, url: str) -> int:
        """
        Score the source URL quality - model-specific pages are more reliable.
        Higher score = more reliable source.
        """
        url_lower = url.lower()

        # Best: Individual model pages (e.g., /the-man-etgx/overview.html)
        model_patterns = [
            r'/the-man-etgx/', r'/the-man-etgs/', r'/the-man-etgl/',
            r'/all-models/.*?/the-man-e', r'/models/.*?electric',
            r'/e-?actros/', r'/fh-?electric/', r'/fm-?electric/',
        ]
        if any(re.search(p, url_lower) for p in model_patterns):
            return 100

        # Good: Technical data pages
        tech_patterns = [r'/spec', r'/technical', r'/data-?sheet', r'technische-daten']
        if any(re.search(p, url_lower) for p in tech_patterns):
            return 80

        # OK: General electric truck overview pages
        ev_patterns = [r'/electric-trucks/', r'/electric.*?overview']
        if any(re.search(p, url_lower) for p in ev_patterns):
            return 60

        # Poor: General info pages (charging, battery regulation, etc.)
        bad_patterns = [
            r'/charging', r'/battery-regulation', r'/regulation',
            r'/service/', r'/advice/', r'/consulting/', r'/general/',
        ]
        if any(re.search(p, url_lower) for p in bad_patterns):
            return 20

        # Default
        return 40

    def _deduplicate(self, vehicles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate vehicles by name, using smart quality-based selection.

        Priority order:
        1. Source URL quality (model pages > generic pages)
        2. Data completeness (more filled fields = better)
        3. Specific variants over generic names
        """
        seen = {}

        for v in vehicles:
            name = v.get('vehicle_name', '').lower().strip()
            if not name:
                continue

            # Skip buses if filtering for trucks
            if self._is_excluded_vehicle(v):
                continue

            # Normalize name for matching
            # e.g., "MAN eTGX" and "MAN eTGX 4x2" should be separate
            # but "MAN eTGX" from two sources should dedupe
            normalized_name = self._normalize_vehicle_name(name)

            source_url = v.get('source_url', '')
            source_quality = self._get_source_quality_score(source_url)

            # Count important filled fields (more specific than just any field)
            important_fields = [
                'battery_capacity_kwh', 'range_km', 'motor_power_kw', 'gvw_kg',
                'dc_charging_kw', 'charging_time_minutes', 'payload_capacity_kg'
            ]
            data_completeness = sum(1 for f in important_fields if v.get(f) is not None)

            # Combined quality score
            quality_score = source_quality + (data_completeness * 10)
            v['_quality_score'] = quality_score

            if normalized_name not in seen:
                seen[normalized_name] = v
            else:
                existing = seen[normalized_name]
                existing_quality = existing.get('_quality_score', 0)

                # Keep the one with better quality score
                if quality_score > existing_quality:
                    seen[normalized_name] = v
                # If same quality, prefer the one with more specific name
                elif quality_score == existing_quality:
                    if len(name) > len(existing.get('vehicle_name', '')):
                        seen[normalized_name] = v

        # Clean up internal score field
        result = []
        for v in seen.values():
            v.pop('_quality_score', None)
            result.append(v)

        return result

    def _normalize_vehicle_name(self, name: str) -> str:
        """
        Normalize vehicle name for deduplication matching.

        E.g., "MAN eTGX 4x2 semitrailer" -> "man etgx 4x2 semitrailer" (unique)
             "MAN eTGX" -> "man etgx" (generic, can be replaced)
        """
        # Keep configuration details as they represent different variants
        return name.lower().strip()

    def _is_excluded_vehicle(self, vehicle: Dict) -> bool:
        """Check if vehicle should be excluded based on category filter."""
        category_filter = ScraperConfig.VEHICLE_CATEGORY_FILTER
        if not category_filter:
            return False

        name = vehicle.get('vehicle_name', '') or ''
        name = name.lower()
        source_url = vehicle.get('source_url', '') or ''
        source_url = source_url.lower()
        configs = vehicle.get('available_configurations') or []

        # If filtering for trucks, exclude buses
        if category_filter == "truck":
            bus_indicators = [
                'lion city', 'city bus', 'citybus', 'coach',
                'neoplan', 'citaro', 'setra'
            ]
            # Check name (be careful not to exclude "eTGS" just because it has "bus" letters)
            if any(bus in name for bus in bus_indicators):
                return True
            # Check source URL
            if '/bus/' in source_url or '/lion-city' in source_url:
                return True
            # Check configurations
            if configs and any('bus' in str(c).lower() for c in configs if c):
                return True

        return False

    def extract_all_vehicles_sync(self, starting_url: str) -> Dict[str, Any]:
        """Synchronous wrapper for extract_all_vehicles"""
        return asyncio.run(self.extract_all_vehicles(starting_url))


# =====================================================================
# JSON EXTRACTION PROMPT FOR PERPLEXITY (IMPROVED)
# =====================================================================

PERPLEXITY_JSON_PROMPT = """Analyze the webpage content and extract electric vehicle specifications.

SOURCE URL: {url}
OEM: {oem_name}

WEBPAGE CONTENT:
---
{content}
---

Extract specifications for EACH electric/hybrid commercial vehicle found.

Return a JSON object with this EXACT structure:
{{
  "vehicles": [
    {{
      "vehicle_name": "Full model name (e.g., MAN eTGX)",
      "battery_capacity_kwh": <number or null - use MAX if range given>,
      "battery_capacity_min_kwh": <number or null - use MIN if range given>,
      "motor_power_kw": <number or null - IMPORTANT: always try to find this>,
      "motor_torque_nm": <number or null>,
      "range_km": <number or null - use MAX if range given>,
      "range_min_km": <number or null - use MIN if range given>,
      "dc_charging_kw": <number or null>,
      "charging_time_minutes": <number or null - time to 80% charge>,
      "gvw_kg": <number in KG - convert tons: 28 tons = 28000 kg>,
      "payload_capacity_kg": <number or null>,
      "powertrain_type": "BEV" | "FCEV" | "PHEV",
      "battery_chemistry": <string or null - e.g., "NMC", "LFP">,
      "available_configurations": ["4x2", "6x2", etc.]
    }}
  ]
}}

CRITICAL RULES:
1. ONLY extract data from the content provided - no external knowledge
2. Convert ALL weights to kg (1 ton = 1000 kg, NOT 1 ton = 1 kg)
3. For ranges like "240-560 kWh", use battery_capacity_kwh=560, battery_capacity_min_kwh=240
4. For ranges like "500-750 km", use range_km=750, range_min_km=500
5. Always look for motor power/torque - check for "kW", "PS", "hp", "Nm" values
6. Return ONLY valid JSON, no markdown or explanations
7. Use EXACT field names as shown above (gvw_kg NOT gvwr_kg, etc.)
"""


# =====================================================================
# RESPONSE PARSER
# =====================================================================

class SpecificationParser:
    """
    Parses Perplexity output into structured VehicleSpecifications.
    Supports both JSON and markdown table formats.
    """
    
    # Pattern to extract vehicle sections
    VEHICLE_PATTERN = re.compile(
        r'\*\*([^*]+)\*\*\s*\n\s*\|[^\n]+\|\s*\n\s*\|[-\s|]+\|\s*\n((?:\|[^\n]+\|\s*\n)+)',
        re.MULTILINE
    )
    
    # Common field mappings (website terms -> our schema)
    FIELD_MAPPINGS = {
        # Battery
        'battery capacity': 'battery_capacity_kwh',
        'battery': 'battery_capacity_kwh',
        'akku': 'battery_capacity_kwh',
        'batterie': 'battery_capacity_kwh',
        
        # Range
        'range': 'range_km',
        'reichweite': 'range_km',
        'driving range': 'range_km',
        
        # Motor/Power
        'motor power': 'motor_power_kw',
        'power': 'motor_power_kw',
        'leistung': 'motor_power_kw',
        'max power': 'motor_power_kw',
        'peak power': 'motor_power_kw',
        
        # Torque
        'torque': 'motor_torque_nm',
        'drehmoment': 'motor_torque_nm',
        
        # Charging
        'dc charging': 'dc_charging_kw',
        'charging power': 'dc_charging_kw',
        'ladeleistung': 'dc_charging_kw',
        'fast charging': 'dc_charging_kw',
        
        # Weight
        'gvw': 'gvw_kg',
        'gcw': 'gvw_kg',
        'gross vehicle weight': 'gvw_kg',
        'gesamtgewicht': 'gvw_kg',
        'zulässiges gesamtgewicht': 'gvw_kg',
        
        # Payload
        'payload': 'payload_capacity_kg',
        'nutzlast': 'payload_capacity_kg',
    }
    
    # Fields to store in additional_specs
    ADDITIONAL_FIELDS = {
        'wheel formula': 'wheel_formula',
        'wheelbase': 'wheelbase_mm',
        'radstand': 'wheelbase_mm',
        'fuel cell': 'fuel_cell_kw',
        'brennstoffzelle': 'fuel_cell_kw',
        'h2 tank': 'h2_tank_kg',
        'wasserstofftank': 'h2_tank_kg',
        'top speed': 'top_speed_kmh',
        'höchstgeschwindigkeit': 'top_speed_kmh',
        'sop': 'start_of_production',
        'market launch': 'start_of_production',
        'markteinführung': 'start_of_production',
    }
    
    @staticmethod
    def extract_numeric(value: str) -> Optional[float]:
        """Extract numeric value from string like '400 kWh' -> 400.0"""
        if not value:
            return None
        # Find first number (including decimals)
        match = re.search(r'[\d,]+\.?\d*', value.replace(',', ''))
        if match:
            try:
                return float(match.group().replace(',', ''))
            except ValueError:
                return None
        return None
    
    def parse_table_row(self, row: str) -> tuple:
        """Parse a markdown table row into (field, value)"""
        parts = [p.strip() for p in row.split('|') if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None
    
    def map_field(self, field_name: str) -> tuple:
        """Map field name to schema field. Returns (schema_field, is_additional)"""
        field_lower = field_name.lower().strip()
        
        # Check main fields
        for pattern, schema_field in self.FIELD_MAPPINGS.items():
            if pattern in field_lower:
                return schema_field, False
        
        # Check additional fields
        for pattern, schema_field in self.ADDITIONAL_FIELDS.items():
            if pattern in field_lower:
                return schema_field, True
        
        return None, None
    
    def parse_vehicle(self, name: str, table_content: str, oem_name: str, source_url: str) -> Dict:
        """Parse a single vehicle's specification table"""
        vehicle = {
            'vehicle_name': name.strip(),
            'oem_name': oem_name,
            'source_url': source_url,
            'extraction_timestamp': datetime.now().isoformat(),
            'additional_specs': {},
            'raw_table_data': table_content,
        }
        
        # Parse each row
        for line in table_content.strip().split('\n'):
            field, value = self.parse_table_row(line)
            if not field or not value:
                continue
            
            schema_field, is_additional = self.map_field(field)
            
            if schema_field:
                numeric_value = self.extract_numeric(value)
                
                if is_additional:
                    # Store in additional_specs
                    if numeric_value is not None:
                        vehicle['additional_specs'][schema_field] = numeric_value
                    else:
                        vehicle['additional_specs'][schema_field] = value
                else:
                    # Store as main field
                    if numeric_value is not None:
                        vehicle[schema_field] = numeric_value
                    # Keep string version for some fields
                    if 'capacity' in schema_field or 'power' in schema_field:
                        vehicle[schema_field] = numeric_value
        
        # Determine powertrain type
        if vehicle.get('battery_capacity_kwh') and not vehicle.get('additional_specs', {}).get('fuel_cell_kw'):
            vehicle['powertrain_type'] = 'BEV'
        elif vehicle.get('additional_specs', {}).get('fuel_cell_kw'):
            vehicle['powertrain_type'] = 'FCEV'
        
        # Calculate completeness score
        important_fields = ['battery_capacity_kwh', 'range_km', 'motor_power_kw', 'gvw_kg']
        filled = sum(1 for f in important_fields if vehicle.get(f))
        vehicle['data_completeness_score'] = filled / len(important_fields)
        
        return vehicle
    
    def parse_content(self, content: str, oem_name: str, source_url: str) -> List[Dict]:
        """Parse entire Perplexity response into list of vehicles"""
        vehicles = []
        
        # Find all vehicle sections
        matches = self.VEHICLE_PATTERN.findall(content)
        
        for name, table_content in matches:
            try:
                vehicle = self.parse_vehicle(name, table_content, oem_name, source_url)
                vehicles.append(vehicle)
            except Exception as e:
                print(f"Warning: Failed to parse vehicle '{name}': {e}")
        
        # If no vehicles found with pattern, try simpler extraction
        if not vehicles:
            vehicles = self._fallback_extraction(content, oem_name, source_url)
        
        return vehicles
    
    def _fallback_extraction(self, content: str, oem_name: str, source_url: str) -> List[Dict]:
        """Fallback extraction when regex pattern fails"""
        vehicles = []

        # Look for any headers followed by specifications
        lines = content.split('\n')
        current_vehicle = None

        for line in lines:
            # Check for vehicle header
            if line.startswith('**') and line.endswith('**'):
                if current_vehicle and current_vehicle.get('vehicle_name'):
                    vehicles.append(current_vehicle)
                current_vehicle = {
                    'vehicle_name': line.strip('* '),
                    'oem_name': oem_name,
                    'source_url': source_url,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'additional_specs': {},
                }
            elif current_vehicle and '|' in line and not line.startswith('|---'):
                field, value = self.parse_table_row(line)
                if field and value:
                    schema_field, is_additional = self.map_field(field)
                    if schema_field:
                        numeric_value = self.extract_numeric(value)
                        if is_additional:
                            current_vehicle['additional_specs'][schema_field] = numeric_value or value
                        elif numeric_value is not None:
                            current_vehicle[schema_field] = numeric_value

        if current_vehicle and current_vehicle.get('vehicle_name'):
            vehicles.append(current_vehicle)

        return vehicles

    # =====================================================================
    # JSON RESPONSE PARSING (NEW - IMPROVED EXTRACTION)
    # =====================================================================

    def parse_json_response(self, response: str, oem_name: str, url: str) -> List[Dict]:
        """
        Parse JSON response from Perplexity with proper unit handling.

        This is the NEW preferred parsing method that handles:
        - Structured JSON extraction
        - Unit conversion (tons -> kg)
        - Min/max range values
        - Completeness scoring
        """
        try:
            # Try to extract JSON from response
            data = self._extract_json(response)
            vehicles_data = data.get('vehicles', [])

            result = []
            for v in vehicles_data:
                vehicle = {
                    'vehicle_name': v.get('vehicle_name', 'Unknown'),
                    'oem_name': oem_name,
                    'source_url': url,
                    'extraction_timestamp': datetime.now().isoformat(),

                    # Core specs with proper handling
                    'battery_capacity_kwh': v.get('battery_capacity_kwh'),
                    'battery_capacity_min_kwh': v.get('battery_capacity_min_kwh'),
                    'motor_power_kw': v.get('motor_power_kw'),
                    'motor_torque_nm': v.get('motor_torque_nm'),
                    'range_km': v.get('range_km'),
                    'range_min_km': v.get('range_min_km'),
                    'dc_charging_kw': v.get('dc_charging_kw'),
                    'charging_time_minutes': v.get('charging_time_minutes'),
                    # Handle both gvw_kg and gvwr_kg (LLM sometimes returns either)
                    'gvw_kg': self._ensure_kg(v.get('gvw_kg') or v.get('gvwr_kg')),
                    'payload_capacity_kg': self._ensure_kg(v.get('payload_capacity_kg')),
                    'powertrain_type': v.get('powertrain_type', 'BEV'),

                    # Additional fields
                    'battery_chemistry': v.get('battery_chemistry'),
                    'available_configurations': v.get('available_configurations', []),

                    # Store raw for debugging
                    'additional_specs': {},
                }

                # Calculate completeness score
                vehicle['data_completeness_score'] = self._calc_completeness(vehicle)
                result.append(vehicle)

            return result

        except Exception as e:
            print(f"[Parser] JSON parsing failed: {e}, falling back to markdown")
            # Fallback to markdown parsing
            return self.parse_content(response, oem_name, url)

    def _ensure_kg(self, value: Optional[float]) -> Optional[float]:
        """
        Ensure weight is in kg, not tons misread as kg.

        Heuristic: If value < 100, it's likely in tons and needs conversion.
        Commercial vehicles:
        - Trucks typically 12,000-44,000 kg GVW
        - So a value of 12-44 is clearly in tons
        """
        if value is None:
            return None
        # If value looks like tons (< 100), convert to kg
        if value < 100:
            return value * 1000
        return value

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from response text, handling various formats"""
        # Clean the text - remove markdown code blocks if present
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError("No valid JSON found in response")

    def _calc_completeness(self, vehicle: Dict) -> float:
        """
        Calculate data completeness score based on filled important fields.

        Important fields weighted for e-powertrain benchmarking.
        """
        important_fields = [
            'battery_capacity_kwh',
            'range_km',
            'motor_power_kw',
            'gvw_kg',
            'dc_charging_kw',
            'motor_torque_nm',
            'payload_capacity_kg',
            'charging_time_minutes',
        ]

        filled = sum(1 for f in important_fields if vehicle.get(f) is not None)
        return round(filled / len(important_fields), 3)


# =====================================================================
# MAIN EXTRACTOR CLASS (Supports both Intelligent Navigation & Legacy)
# =====================================================================

class EPowertrainExtractor:
    """
    Main extractor class with intelligent auto-fallback:

    1. FAST MODE (single page): Quick extraction from a single URL
       - Best for pages with all specs visible
       - Lower latency

    2. DEEP MODE (multi-page): Three-phase LLM-guided extraction
       - Discovers spec page links using LLM
       - Crawls up to 5 relevant pages
       - Extracts specs using Pydantic schema + LLM

    3. AUTO-FALLBACK: Starts with fast mode, falls back to deep if:
       - No vehicles extracted
       - Average completeness below threshold
       - Too many vehicles with zero data

    This guarantees all data comes from the source URL.
    """

    def __init__(self, use_intelligent_mode: bool = None, auto_fallback: bool = None):
        # Determine initial mode
        if use_intelligent_mode is None:
            use_intelligent_mode = ScraperConfig.ENABLE_INTELLIGENT_NAVIGATION
        self.use_intelligent_mode = use_intelligent_mode

        # Auto-fallback setting
        if auto_fallback is None:
            auto_fallback = ScraperConfig.AUTO_FALLBACK_ENABLED
        self.auto_fallback = auto_fallback

        # Check for required API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

        # Initialize perplexity mode components (for fallback capability)
        self.parser = SpecificationParser()
        self.content_fetcher = WebContentFetcher()

        if self.perplexity_api_key:
            self.perplexity_api_key = self.perplexity_api_key.strip('"\'')
            self.headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = None

        # Initialize intelligent mode components (for fallback capability)
        if self.openai_api_key:
            self.intelligent_scraper = IntelligentScraper(self.openai_api_key)
        else:
            self.intelligent_scraper = None

        # Validate we have at least one mode available
        if self.use_intelligent_mode and not self.intelligent_scraper:
            raise ValueError("OPENAI_API_KEY must be set for intelligent mode")
        if not self.use_intelligent_mode and not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY must be set for perplexity mode")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        if '://' in url:
            return url.split('/')[2]
        return url

    def _query_perplexity(self, query: str, max_tokens: int = 8000) -> Dict:
        """Send query to Perplexity API"""
        payload = {
            "model": ScraperConfig.DEFAULT_MODEL,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
            "temperature": ScraperConfig.TEMPERATURE,
            "return_citations": False  # Not needed when we provide content
        }

        try:
            response = requests.post(
                ScraperConfig.API_URL,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return {
                'content': result['choices'][0]['message']['content'],
                'usage': result.get('usage', {}),
            }
        except Exception as e:
            return {'error': str(e)}

    def _check_data_quality(self, result: Dict) -> tuple:
        """
        Check if extraction result has sufficient data quality.

        Returns:
            (is_sufficient: bool, reason: str)
        """
        vehicles = result.get('vehicles', [])

        # Check 1: No vehicles at all
        if not vehicles:
            return False, "No vehicles extracted"

        # Check 2: Calculate average completeness
        completeness_scores = [v.get('data_completeness_score', 0) for v in vehicles]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)

        if avg_completeness < ScraperConfig.MIN_COMPLETENESS_THRESHOLD:
            return False, f"Low avg completeness: {avg_completeness:.2f} < {ScraperConfig.MIN_COMPLETENESS_THRESHOLD}"

        # Check 3: Vehicles with at least some data
        vehicles_with_data = sum(1 for score in completeness_scores if score > 0)
        data_ratio = vehicles_with_data / len(vehicles)

        if data_ratio < ScraperConfig.MIN_VEHICLES_WITH_DATA:
            return False, f"Too few vehicles with data: {vehicles_with_data}/{len(vehicles)} ({data_ratio:.0%})"

        return True, f"Quality OK: {avg_completeness:.2f} avg, {vehicles_with_data}/{len(vehicles)} with data"

    def _merge_results(self, perplexity_result: Dict, intelligent_result: Dict) -> Dict:
        """
        Merge results from perplexity and intelligent modes.
        Keeps vehicles with best completeness scores.
        """
        merged_vehicles = {}

        # Add perplexity vehicles
        for v in perplexity_result.get('vehicles', []):
            name = v.get('vehicle_name', '').lower().strip()
            if name:
                merged_vehicles[name] = v

        # Add/update with intelligent vehicles (keep best completeness)
        for v in intelligent_result.get('vehicles', []):
            name = v.get('vehicle_name', '').lower().strip()
            if name:
                existing = merged_vehicles.get(name)
                if not existing or v.get('data_completeness_score', 0) > existing.get('data_completeness_score', 0):
                    merged_vehicles[name] = v

        vehicles = list(merged_vehicles.values())

        # Create merged result
        return {
            'oem_name': intelligent_result.get('oem_name') or perplexity_result.get('oem_name'),
            'oem_url': intelligent_result.get('oem_url') or perplexity_result.get('oem_url'),
            'vehicles': vehicles,
            'total_vehicles_found': len(vehicles),
            'extraction_timestamp': datetime.now().isoformat(),
            'official_citations': list(set(
                perplexity_result.get('official_citations', []) +
                intelligent_result.get('spec_urls_found', [])
            )),
            'third_party_citations': [],
            'source_compliance_score': 1.0,
            'raw_content': f"Merged: Fast + Deep mode ({intelligent_result.get('pages_crawled', 0)} pages)",
            'pages_crawled': intelligent_result.get('pages_crawled', 1),
            'spec_urls_found': intelligent_result.get('spec_urls_found', []),
            'extraction_details': intelligent_result.get('extraction_details', []),
            'fetched_content_length': perplexity_result.get('fetched_content_length', 0),
            'tokens_used': perplexity_result.get('tokens_used', 0),
            'model_used': 'perplexity+intelligent',
            'extraction_duration_seconds': (
                perplexity_result.get('extraction_duration_seconds', 0) +
                intelligent_result.get('extraction_duration_seconds', 0)
            ),
            'errors': [],
            'warnings': ['Auto-fallback to intelligent mode was triggered'],
        }

    def extract_oem_data(self, url: str) -> Dict:
        """
        Extract OEM data from URL with intelligent auto-fallback.

        Strategy:
        1. If intelligent mode requested: use intelligent mode directly
        2. If perplexity mode requested with auto-fallback:
           a. Try perplexity first (fast)
           b. Check data quality
           c. If insufficient, fallback to intelligent mode
           d. Merge results from both modes

        Returns ScrapingResult compatible with LangGraph workflow.
        """
        start_time = time.time()
        domain = self._extract_domain(url)
        oem_name = self._extract_oem_name(domain, '')

        # Direct intelligent mode (no fallback needed)
        if self.use_intelligent_mode:
            return self._extract_intelligent(url, oem_name, start_time)

        # Perplexity mode with optional auto-fallback
        perplexity_result = self._extract_legacy(url, oem_name, domain, start_time)

        # Check if auto-fallback is enabled and intelligent mode is available
        if not self.auto_fallback or not self.intelligent_scraper:
            return perplexity_result

        # Check data quality
        is_sufficient, quality_reason = self._check_data_quality(perplexity_result)

        if is_sufficient:
            print(f"[Auto-Fallback] {quality_reason} - No fallback needed")
            return perplexity_result

        # Trigger fallback to intelligent mode
        print(f"\n[Auto-Fallback] {quality_reason}")
        print(f"[Auto-Fallback] Switching to MULTI-PAGE MODE for better extraction...")

        intelligent_start = time.time()
        intelligent_result = self._extract_intelligent(url, oem_name, intelligent_start)

        # Merge results from both modes
        merged_result = self._merge_results(perplexity_result, intelligent_result)

        # Log improvement
        perplexity_count = len(perplexity_result.get('vehicles', []))
        intelligent_count = len(intelligent_result.get('vehicles', []))
        merged_count = len(merged_result.get('vehicles', []))

        perplexity_avg = sum(v.get('data_completeness_score', 0) for v in perplexity_result.get('vehicles', [])) / max(perplexity_count, 1)
        merged_avg = sum(v.get('data_completeness_score', 0) for v in merged_result.get('vehicles', [])) / max(merged_count, 1)

        print(f"[Auto-Fallback] Results:")
        print(f"  Fast mode: {perplexity_count} vehicles, {perplexity_avg:.0%} avg completeness")
        print(f"  Deep mode: {intelligent_count} vehicles")
        print(f"  Merged: {merged_count} vehicles, {merged_avg:.0%} avg completeness")

        return merged_result

    def _extract_intelligent(self, url: str, oem_name: str, start_time: float) -> Dict:
        """
        MULTI-PAGE MODE: Three-phase LLM-guided extraction.

        Phase 1: Discover spec page links
        Phase 2: Crawl up to 5 spec pages
        Phase 3: Extract structured specs with LLM + Pydantic
        """
        print(f"\n{'='*60}")
        print(f"MULTI-PAGE MODE - {oem_name}")
        print(f"{'='*60}")

        try:
            result = self.intelligent_scraper.extract_all_vehicles_sync(url)

            # Convert to ScrapingResult format
            vehicles = result.get('vehicles', [])

            # Add OEM name to each vehicle
            for v in vehicles:
                v['oem_name'] = oem_name
                # Calculate completeness score
                important_fields = ['battery_capacity_kwh', 'range_km', 'motor_power_kw', 'gvw_kg']
                filled = sum(1 for f in important_fields if v.get(f))
                v['data_completeness_score'] = filled / len(important_fields)

            duration = time.time() - start_time

            return {
                'oem_name': oem_name,
                'oem_url': url,
                'vehicles': vehicles,
                'total_vehicles_found': len(vehicles),
                'extraction_timestamp': datetime.now().isoformat(),
                'official_citations': result.get('spec_urls_found', [url]),
                'third_party_citations': [],
                'source_compliance_score': 1.0,  # All from OEM domain
                'raw_content': f"Intelligent extraction from {result.get('pages_crawled', 1)} pages",
                'pages_crawled': result.get('pages_crawled', 1),
                'spec_urls_found': result.get('spec_urls_found', [url]),
                'extraction_details': result.get('extraction_details', []),
                'fetched_content_length': 0,  # Not applicable
                'tokens_used': 0,  # Tracked separately
                'model_used': ScraperConfig.LLM_EXTRACTION_MODEL,
                'extraction_duration_seconds': round(duration, 2),
                'errors': [],
                'warnings': [],
            }

        except Exception as e:
            print(f"[Intelligent Mode] ERROR: {e}")
            return self._create_error_result(url, self._extract_domain(url), str(e), start_time)

    def _extract_legacy(self, url: str, oem_name: str, domain: str, start_time: float) -> Dict:
        """
        SINGLE PAGE MODE: Two-stage Crawl4AI + LLM extraction.

        Stage 1: Fetch webpage content with Crawl4AI
        Stage 2: Extract structured specs with LLM
        """
        print(f"\n{'='*60}")
        print(f"SINGLE PAGE MODE - {oem_name}")
        print(f"{'='*60}")

        # ===== STAGE 1: Fetch webpage content with Crawl4AI =====
        print(f"[Stage 1] Fetching content from: {url}")
        fetch_result = self.content_fetcher.fetch(url)

        if not fetch_result['success']:
            print(f"[Stage 1] FAILED: {fetch_result['error']}")
            return self._create_error_result(
                url, domain,
                f"Failed to fetch webpage: {fetch_result['error']}",
                start_time
            )

        raw_markdown = fetch_result['markdown']
        print(f"[Stage 1] SUCCESS: Fetched {len(raw_markdown)} chars of content")

        if len(raw_markdown) < 100:
            print(f"[Stage 1] WARNING: Very little content fetched")
            return self._create_error_result(
                url, domain,
                "Webpage content too short - may be blocked or empty",
                start_time
            )

        # Truncate content if too long
        content_for_extraction = raw_markdown[:ScraperConfig.MAX_CONTENT_LENGTH]
        if len(raw_markdown) > ScraperConfig.MAX_CONTENT_LENGTH:
            print(f"[Stage 1] Truncated content from {len(raw_markdown)} to {ScraperConfig.MAX_CONTENT_LENGTH} chars")

        # ===== STAGE 2: Extract structured data with LLM (JSON format) =====
        print(f"[Stage 2] Extracting specifications...")

        # Use the improved JSON prompt for better extraction
        query = PERPLEXITY_JSON_PROMPT.format(
            url=url,
            oem_name=oem_name,
            content=content_for_extraction
        )

        result = self._query_perplexity(query)

        if 'error' in result:
            print(f"[Stage 2] FAILED: {result['error']}")
            return self._create_error_result(url, domain, result['error'], start_time)

        print(f"[Stage 2] SUCCESS: Extraction complete")

        # Parse JSON response (with fallback to markdown)
        vehicles = self.parser.parse_json_response(result['content'], oem_name, url)
        print(f"[Stage 2] Extracted {len(vehicles)} vehicles")

        duration = time.time() - start_time
        usage = result.get('usage', {})

        return {
            'oem_name': oem_name,
            'oem_url': url,
            'vehicles': vehicles,
            'total_vehicles_found': len(vehicles),
            'extraction_timestamp': datetime.now().isoformat(),
            'official_citations': [url],  # Data is from this URL
            'third_party_citations': [],  # No third-party sources used
            'source_compliance_score': 1.0,  # 100% from source URL
            'raw_content': result['content'],
            'fetched_content_length': len(raw_markdown),
            'tokens_used': usage.get('total_tokens', 0),
            'model_used': ScraperConfig.DEFAULT_MODEL,
            'extraction_duration_seconds': round(duration, 2),
            'errors': [],
            'warnings': [],
        }
    
    def _extract_oem_name(self, domain: str, content: str) -> str:
        """Extract OEM name from domain or content"""
        # Known OEM domains
        oem_domains = {
            'tesla.com': 'Tesla',
            'man.eu': 'MAN Truck & Bus',
            'daimler-truck.com': 'Daimler Truck',
            'mercedes-benz-trucks.com': 'Mercedes-Benz Trucks',
            'volvotrucks.com': 'Volvo Trucks',
            'scania.com': 'Scania',
            'daf.com': 'DAF Trucks',
            'iveco.com': 'IVECO',
            'renault-trucks.com': 'Renault Trucks',
            'byd.com': 'BYD',
            'nikola.com': 'Nikola',
        }
        
        for dom, name in oem_domains.items():
            if dom in domain:
                return name
        
        # Clean domain as fallback
        name = domain.replace('www.', '').split('.')[0]
        return name.upper() if len(name) <= 4 else name.title()
    
    def _create_error_result(self, url: str, domain: str, error: str, start_time: float) -> Dict:
        """Create error result structure"""
        return {
            'oem_name': domain.replace('www.', '').split('.')[0].title(),
            'oem_url': url,
            'vehicles': [],
            'total_vehicles_found': 0,
            'extraction_timestamp': datetime.now().isoformat(),
            'official_citations': [],
            'third_party_citations': [],
            'source_compliance_score': 0.0,
            'raw_content': '',
            'tokens_used': 0,
            'model_used': ScraperConfig.DEFAULT_MODEL,
            'extraction_duration_seconds': round(time.time() - start_time, 2),
            'errors': [error],
            'warnings': [],
        }
    
    def process_urls(self, urls: List[str], output_dir: str = "outputs") -> List[Dict]:
        """
        Process multiple URLs and return list of ScrapingResults.
        Also saves to JSON files.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        print("=" * 60)
        print("E-POWERTRAIN SPECIFICATION EXTRACTOR")
        print("=" * 60)
        print(f"URLs to process: {len(urls)}")
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*20} OEM {i}/{len(urls)} {'='*20}")
            
            result = self.extract_oem_data(url)
            all_results.append(result)
            
            # Print summary
            print(f"OEM: {result['oem_name']}")
            print(f"Vehicles found: {result['total_vehicles_found']}")
            print(f"Official citations: {len(result['official_citations'])}")
            print(f"Source score: {result['source_compliance_score']}")
            
            if result['vehicles']:
                for v in result['vehicles']:
                    print(f"  - {v['vehicle_name']}: {v.get('battery_capacity_kwh', 'N/A')} kWh, {v.get('range_km', 'N/A')} km")
            
            if result['errors']:
                print(f"Errors: {result['errors']}")
            
            # Save individual result
            safe_name = re.sub(r'[^\w\-]', '_', result['oem_name'])
            output_path = Path(output_dir) / f"scraping_{safe_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved to: {output_path}")
            
            # Wait between URLs
            if i < len(urls):
                print(f"\nWaiting {ScraperConfig.WAIT_BETWEEN_URLS}s before next URL...")
                time.sleep(ScraperConfig.WAIT_BETWEEN_URLS)
        
        # Save combined results
        combined_path = Path(output_dir) / "all_scraping_results.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nAll results saved to: {combined_path}")
        
        # Summary
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}")
        successful = sum(1 for r in all_results if not r['errors'])
        total_vehicles = sum(r['total_vehicles_found'] for r in all_results)
        print(f"Successful: {successful}/{len(urls)}")
        print(f"Total vehicles: {total_vehicles}")
        
        return all_results
    
    def process_urls_file(self, urls_file: str = 'src/inputs/urls.txt') -> List[Dict]:
        """Process URLs from file"""
        try:
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f 
                       if line.strip() and line.strip().startswith('http')]
        except FileNotFoundError:
            print(f"ERROR: {urls_file} not found")
            return []
        
        if not urls:
            print("ERROR: No valid URLs found")
            return []
        
        return self.process_urls(urls)


# =====================================================================
# AGENT FUNCTION - Called by LangGraph
# =====================================================================

def scrape_oem_urls(urls: List[str]) -> List[Dict]:
    """
    Scrape OEM URLs and return list of ScrapingResults.
    
    This is the function called by the scraping_agent node.
    
    Args:
        urls: List of OEM website URLs
    
    Returns:
        List of ScrapingResult dicts
    """
    extractor = EPowertrainExtractor()
    return extractor.process_urls(urls)


# =====================================================================
# CLI
# =====================================================================

def main():
    print("E-POWERTRAIN SPECIFICATION EXTRACTOR")
    print("Structured output for LangGraph workflow")
    print("-" * 60)
    
    try:
        extractor = EPowertrainExtractor()
        results = extractor.process_urls_file()
        
        print("\nExtraction completed!")
        print("Results saved to outputs/ directory")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()