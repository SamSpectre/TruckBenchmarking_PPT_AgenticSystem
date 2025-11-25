"""
E-Powertrain Web Scraper Tool

Uses Perplexity API to extract technical specifications from OEM websites.
Outputs structured data matching the ScrapingResult schema.

UPDATED: Now produces ScrapingResult compatible output for LangGraph workflow.
"""

import requests
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

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


# =====================================================================
# RESPONSE PARSER
# =====================================================================

class SpecificationParser:
    """
    Parses Perplexity markdown output into structured VehicleSpecifications.
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
# MAIN EXTRACTOR CLASS
# =====================================================================

class EPowertrainExtractor:
    """
    Main extractor class that queries Perplexity and returns structured data.
    """
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY must be set in .env file")
        
        self.api_key = self.api_key.strip('"\'')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.parser = SpecificationParser()
    
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
            "return_citations": True
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
                'citations': result.get('citations', []),
                'usage': result.get('usage', {}),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_oem_data(self, url: str) -> Dict:
        """
        Extract OEM data from URL and return ScrapingResult.
        
        This is the main method that returns data compatible with the
        LangGraph workflow state.
        """
        start_time = time.time()
        domain = self._extract_domain(url)
        
        # Build extraction query
        query = f"""
I need the technical specification tables for ALL electric and hybrid commercial vehicles from {url}.

Please go directly to {url} and extract the technical specifications for each electric/hybrid truck or commercial vehicle they offer. Present the data in the exact same table format as it appears on their official website.

For each vehicle model found, show:

**[Vehicle Model Name]**
| Specification | Value |
|---------------|-------|
[Use the exact specification names and values as shown on the official website]

Requirements:
- Extract data ONLY from {domain} (the official manufacturer website)
- Use the exact parameter names as they appear on the official website
- Include ALL technical specifications available for each vehicle
- Present each vehicle's specs in separate tables
- Focus on e-powertrain data: battery, motor, range, charging, power, GVW, etc.
- Include wheel formula, wheelbase if available

DO NOT use third-party websites - only extract from the official {domain} website.
"""
        
        print(f"Extracting from: {url}")
        result = self._query_perplexity(query)
        
        if 'error' in result:
            return self._create_error_result(url, domain, result['error'], start_time)
        
        # Parse citations
        citations = result.get('citations', [])
        official_citations = []
        third_party_citations = []
        
        domain_variations = [domain, domain.replace('www.', ''), f"www.{domain.replace('www.', '')}"]
        
        for citation in citations:
            citation_url = citation.get('url', '') if isinstance(citation, dict) else str(citation)
            if any(d in citation_url for d in domain_variations):
                official_citations.append(citation_url)
            else:
                third_party_citations.append(citation_url)
        
        # Parse vehicles
        oem_name = self._extract_oem_name(domain, result['content'])
        vehicles = self.parser.parse_content(result['content'], oem_name, url)
        
        # Calculate source compliance score
        total_citations = len(official_citations) + len(third_party_citations)
        if total_citations == 0:
            source_score = 0.0
        else:
            source_score = len(official_citations) / total_citations
        
        duration = time.time() - start_time
        usage = result.get('usage', {})
        
        return {
            'oem_name': oem_name,
            'oem_url': url,
            'vehicles': vehicles,
            'total_vehicles_found': len(vehicles),
            'extraction_timestamp': datetime.now().isoformat(),
            'official_citations': official_citations,
            'third_party_citations': third_party_citations,
            'source_compliance_score': round(source_score, 2),
            'raw_content': result['content'],
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