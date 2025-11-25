# E-Powertrain Benchmarking System

A LangGraph-based multi-agent system for automated benchmarking of electric commercial vehicle specifications. The system scrapes OEM websites, validates data quality, and generates professional PowerPoint presentations.

## Project Status: MVP Complete

The core workflow is functional with the following capabilities:
- Web scraping of e-powertrain specifications from OEM websites
- Rule-based quality validation with configurable thresholds
- Automated PowerPoint generation using IAA template format
- Cost tracking across all API calls

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   URLs ──► [Scraping Agent] ──► [Quality Validator] ──►         │
│                │                      │                          │
│           Perplexity API         Rule-based +                    │
│           (sonar-pro)            Optional LLM                    │
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
- Perplexity API key
- OpenAI API key (optional, for LLM validation)

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
# Edit .env with your API keys
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

# Show workflow graph
python main.py --graph
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
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not in git)
├── CLAUDE.md                    # Context for Claude Code sessions
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
│   │   └── settings.py          # Pydantic settings management
│   │
│   ├── graph/
│   │   └── runtime.py           # LangGraph workflow definition
│   │
│   ├── state/
│   │   └── state.py             # TypedDict state definitions
│   │
│   ├── tools/
│   │   ├── scraper.py           # Perplexity API integration
│   │   └── ppt_generator.py     # PowerPoint generation
│   │
│   └── inputs/
│       └── urls.txt             # Default OEM URLs
│
├── templates/
│   └── IAA_Template.pptx        # PowerPoint template
│
└── outputs/                     # Generated files
```

## Configuration

Edit `src/config/settings.py` or use environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `PERPLEXITY_API_KEY` | Required | Perplexity API key |
| `output_directory` | `outputs` | Output folder |
| `ppt_template_path` | `templates/IAA_Template.pptx` | Template path |
| `minimum_quality_score` | `0.75` | Quality threshold (0-1) |
| `max_retry_attempts` | `3` | Max retries on failure |

## Data Flow

1. **Input**: List of OEM website URLs
2. **Scraping**: Perplexity API extracts vehicle specifications
3. **Validation**: Rule-based checks for completeness, accuracy, consistency
4. **Transformation**: VehicleSpecifications → OEMProfile (IAA format)
5. **Generation**: PowerPoint presentation per OEM
6. **Output**: .pptx files + JSON data

## Key Components

### State Management (`src/state/state.py`)

- `BenchmarkingState`: Main workflow state passed between agents
- `VehicleSpecifications`: Flexible schema for vehicle data
- `ScrapingResult`: Output from scraper with citations
- `QualityValidationResult`: Validation scores and issues
- `OEMProfile`: IAA template-compatible format

### Agents

| Agent | Function | Model |
|-------|----------|-------|
| Scraping | `scraping_node()` | Perplexity sonar-pro |
| Validation | `validation_node()` | Rule-based (+ optional GPT-4o) |
| Presentation | `presentation_node()` | python-pptx |

### Tools

- **EPowertrainExtractor**: Queries Perplexity API with structured prompts
- **SpecificationParser**: Parses markdown tables to structured data
- **PowerPointGenerator**: Fills IAA template with OEM data

## Future Scope

### Phase 2: Enhanced Capabilities
- [ ] Add Anthropic Claude as alternative LLM provider
- [ ] Implement true async scraping with aiohttp for parallel URL fetching
- [ ] Add chart generation (matplotlib/plotly) to presentations
- [ ] PDF specification document extraction
- [ ] Image/logo scraping and embedding

### Phase 3: Industry Standards
- [ ] Structured logging with log rotation
- [ ] Unit and integration test suite (pytest)
- [ ] Error handling with circuit breakers and exponential backoff
- [ ] Metrics and monitoring dashboard

### Phase 4: Advanced Features
- [ ] Human-in-the-loop approval steps
- [ ] Database storage for historical comparisons
- [ ] REST API for integration
- [ ] Web UI dashboard
- [ ] Scheduled/automated benchmark runs
- [ ] Multi-language support

### Phase 5: Plug-and-Play Architecture
- [ ] Provider abstraction layer (swap LLM providers via config)
- [ ] Plugin system for custom scrapers
- [ ] Configurable agent pipelines
- [ ] Template marketplace for different presentation styles

## Cost Tracking

The system tracks API costs automatically:

| Model | Input (per 1K) | Output (per 1K) | Notes |
|-------|----------------|-----------------|-------|
| sonar-pro | $0.003 | $0.015 | + $0.005/request |
| gpt-4o | $0.0025 | $0.01 | Optional validation |
| gpt-5-mini | $0.00025 | $0.002 | Future use |

Typical run cost: ~$0.03 per OEM

## Development

### Adding a New OEM

1. Add URL to `src/inputs/urls.txt`
2. Run `python main.py`

### Customizing the Template

1. Edit `templates/IAA_Template.pptx`
2. Update shape IDs in `src/tools/ppt_generator.py` if structure changes

### Adding a New LLM Provider

1. Create provider in `src/providers/` (future)
2. Implement `BaseLLMProvider` interface
3. Register in `ProviderFactory`

## Contributing

This project is under active development. See CLAUDE.md for current context and development notes.

## License

Proprietary - Internal Use Only

---

**Last Updated**: November 2024
**Version**: 0.1.0 (MVP)
