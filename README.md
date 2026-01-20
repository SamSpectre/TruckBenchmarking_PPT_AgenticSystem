# E-Powertrain Benchmarking System

A LangGraph-based multi-agent system for automated benchmarking of electric commercial vehicle specifications. The system intelligently crawls OEM websites, extracts vehicle data using LLM-powered analysis, validates data quality, and generates professional PowerPoint presentations.

## Project Status: Production Ready (v0.7.0)

The system is fully functional with the following capabilities:
- Intelligent multi-page web crawling with automatic spec page discovery
- LLM-powered extraction of e-powertrain specifications from OEM websites
- Strict validation to ensure extraction accuracy (100% verified)
- Rule-based quality validation with configurable thresholds
- Automated PowerPoint generation using IAA template format
- Async parallel processing for multiple URLs (2-3x faster)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   URLs ──► [Scraping Agent] ──► [Quality Validator] ──►         │
│                │                      │                          │
│           CRAWL4AI +              Rule-based +                   │
│           OpenAI GPT-4o           Optional LLM                   │
│                                       │                          │
│                                       ▼                          │
│                              [Presentation Generator] ──► .pptx  │
│                                       │                          │
│                                  python-pptx                     │
│                                                                  │
│   Retry Loop: If quality < 0.75, retry up to 3 times            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone repository
cd Langgraph_Project1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Usage

```bash
# Run with default URLs (src/inputs/urls.txt)
python main.py

# Run with specific URLs
python main.py --urls https://www.man.eu/trucks https://www.volvo.com/trucks

# Run with custom URL file
python main.py --file my_urls.txt

# Stream progress output
python main.py --stream

# Run the web UI
python app.py
# Opens at http://127.0.0.1:7860
```

### Output

The system generates:
- Individual PowerPoint presentations for each OEM (`outputs/IAA_<OEM>_<timestamp>.pptx`)
- JSON files with scraped data (`outputs/scraping_<OEM>.json`)
- Combined results (`outputs/all_scraping_results.json`)

## Project Structure

```
Langgraph_Project1/
├── main.py                      # CLI entry point
├── app.py                       # Gradio Web UI
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not in git)
├── README.md                    # This file
│
├── src/
│   ├── agents/                  # LangGraph agent nodes
│   │   ├── __init__.py
│   │   ├── scraping_agent.py    # Web scraping orchestration
│   │   ├── quality_validator.py # Data quality validation
│   │   └── presentation_generator.py  # PPT generation
│   │
│   ├── config/
│   │   ├── settings.py          # Pydantic settings management
│   │   └── terminology_mappings.py  # Field name equivalences
│   │
│   ├── graph/
│   │   └── runtime.py           # LangGraph workflow definition
│   │
│   ├── state/
│   │   └── state.py             # TypedDict state definitions
│   │
│   ├── tools/
│   │   ├── scraper.py           # CRAWL4AI + OpenAI extraction
│   │   └── ppt_generator.py     # PowerPoint generation
│   │
│   └── inputs/
│       └── urls.txt             # Default OEM URLs
│
├── templates/
│   └── IAA_Template.pptx        # PowerPoint template
│
├── tests/                       # Test suite
│   └── test_*.py
│
└── outputs/                     # Generated files
```

## Configuration

Edit `src/config/settings.py` or use environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key for extraction |
| `output_directory` | `outputs` | Output folder |
| `ppt_template_path` | `templates/IAA_Template.pptx` | Template path |
| `minimum_quality_score` | `0.75` | Quality threshold (0-1) |
| `max_retry_attempts` | `3` | Max retries on failure |

## Data Flow

1. **Input**: List of OEM website URLs
2. **Crawling**: CRAWL4AI discovers and fetches spec pages from entry URL
3. **Extraction**: OpenAI GPT-4o extracts structured vehicle specifications
4. **Validation**: Rule-based checks for completeness, accuracy, consistency
5. **Transformation**: VehicleSpecifications → OEMProfile (IAA format)
6. **Generation**: PowerPoint presentation per OEM
7. **Output**: .pptx files + JSON data

## Key Components

### State Management (`src/state/state.py`)

- `BenchmarkingState`: Main workflow state passed between agents
- `VehicleSpecifications`: Flexible schema for vehicle data
- `ScrapingResult`: Output from scraper with citations
- `QualityValidationResult`: Validation scores and issues
- `OEMProfile`: IAA template-compatible format

### Agents

| Agent | Function | Technology |
|-------|----------|------------|
| Scraping | `scraping_node()` | CRAWL4AI + OpenAI GPT-4o |
| Validation | `validation_node()` | Rule-based (+ optional LLM) |
| Presentation | `presentation_node()` | python-pptx |

### Tools

- **EPowertrainExtractor**: Intelligent multi-page crawler with LLM extraction
- **SpecificationParser**: Parses extracted data to structured format
- **PowerPointGenerator**: Fills IAA template with OEM data

## Verified Extraction Accuracy

The system has been verified against official OEM websites:

| OEM | Vehicles Extracted | Quality Score | Accuracy |
|-----|-------------------|---------------|----------|
| MAN Truck & Bus | 7 vehicles | 94.4% | 100% |
| Volvo Trucks | 8 vehicles | 95.6% | ~98% |

*Quality score reflects data completeness (filled fields), not extraction accuracy.*

## Future Scope

### Phase 2: Enhanced Capabilities
- [ ] Add more OEM support (Mercedes-Benz, Scania, DAF, etc.)
- [ ] PDF specification document extraction
- [ ] Image/logo scraping and embedding
- [ ] Chart generation in presentations

### Phase 3: Industry Standards
- [ ] Structured logging with log rotation
- [ ] Error handling with circuit breakers
- [ ] Metrics and monitoring dashboard

### Phase 4: Advanced Features
- [ ] Human-in-the-loop approval steps
- [ ] Database storage for historical comparisons
- [ ] REST API for integration
- [ ] Scheduled/automated benchmark runs

## Development

### Adding a New OEM

1. Add URL to `src/inputs/urls.txt`
2. Run `python main.py`

### Customizing the Template

1. Edit `templates/IAA_Template.pptx`
2. Update shape IDs in `src/tools/ppt_generator.py` if structure changes

### Running Tests

```bash
pytest tests/ -v
```

## License

Proprietary - Internal Use Only

---

**Last Updated**: January 2026
**Version**: 0.7.0
