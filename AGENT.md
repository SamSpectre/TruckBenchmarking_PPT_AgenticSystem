# Claude Code Context File

> **Purpose**: This file provides context for Claude Code sessions to understand the project state, history, and continue development seamlessly.
> **Update this file**: After each development session to maintain continuity.
> **Last Updated**: December 4, 2024 (Session 6)

---

## Project Overview

**Name**: E-Powertrain Benchmarking System
**Type**: LangGraph Multi-Agent System
**Status**: MVP Complete + Async Parallel Processing (v0.6.0)
**Last Session**: December 4, 2024 (Session 6 - Async Parallel URL Processing)

### What This Project Does

Automates the benchmarking of electric commercial vehicles (trucks, buses) by:
1. Scraping specifications from OEM websites (two modes available)
2. Validating data quality with rule-based checks
3. Generating PowerPoint presentations with **plug-and-play template support**

### Key Capabilities (v0.6.0)
- **Web Scraping**: Dual-mode (Perplexity + Intelligent) with auto-fallback
- **Async Parallel Processing**: Process multiple URLs simultaneously (2-3x faster for 3+ URLs)
- **Quality Validation**: Rule-based + optional LLM validation
- **PPT Generation**: Fixed IAA template OR any custom template via plug-and-play system
- **Multi-Slide Support**: Automatic overflow slides when products exceed items_per_slide
- **Two UIs**: Main benchmarking app + Template management app

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                          │
├─────────────────────────────────────────────────────────────────┤
│  app.py (Main Benchmarking UI)    template_app.py (Template UI) │
│  http://127.0.0.1:7860            http://127.0.0.1:7870         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE WORKFLOW (LangGraph)                   │
│                     src/graph/runtime.py                         │
├─────────────────────────────────────────────────────────────────┤
│  scraping_node ──► validation_node ──► presentation_node        │
│       │                   │                    │                 │
│  EPowertrain         RuleBasedValidator   generate_all_         │
│  Extractor           + LLMValidator       presentations         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POWERPOINT GENERATION                         │
├─────────────────────────────────────────────────────────────────┤
│  ORIGINAL (Hardcoded)              PLUG-AND-PLAY (New)          │
│  src/tools/ppt_generator.py        src/tools/dynamic_ppt_       │
│  - IAA template only               generator.py                  │
│  - Hardcoded shape IDs             - Any template via JSON      │
│  - Fixed field mappings            - Dynamic mappings            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TEMPLATE SYSTEM (Session 4)                     │
├─────────────────────────────────────────────────────────────────┤
│  template_analyzer.py    →  Discovers template structure         │
│  template_analysis_agent.py  →  LLM/rule-based mapping gen      │
│  template_registry.py    →  Manages saved mappings               │
│  src/config/template_schemas/*.json  →  Mapping configs          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files & Their Purpose

### Core System (Sessions 1-3)
| File | Purpose | Status |
|------|---------|--------|
| `app.py` | Main Gradio Web UI for benchmarking | Complete |
| `main.py` | CLI entry point | Complete |
| `src/graph/runtime.py` | LangGraph StateGraph workflow | Complete |
| `src/state/state.py` | TypedDict state definitions | Complete |
| `src/config/settings.py` | Pydantic settings | Complete |
| `src/agents/scraping_agent.py` | Scraping node | Complete |
| `src/agents/quality_validator.py` | Validation node | Complete |
| `src/agents/presentation_generator.py` | Presentation node | Complete |
| `src/tools/scraper.py` | Dual-mode scraper | Complete |
| `src/tools/ppt_generator.py` | IAA template generator (hardcoded) | Complete |

### Plug-and-Play Template System (Session 4 - NEW)
| File | Purpose | Status |
|------|---------|--------|
| `template_app.py` | **NEW** Gradio UI for template management | Complete |
| `src/tools/template_analyzer.py` | **NEW** Discovers any template structure | Complete |
| `src/tools/dynamic_ppt_generator.py` | **NEW** JSON-config based generator | Complete |
| `src/agents/template_analysis_agent.py` | **NEW** LLM/rule-based mapping | Complete |
| `src/config/template_registry.py` | **NEW** Manages saved mappings | Complete |
| `src/config/template_schemas/iaa_template.json` | **NEW** IAA mapping config | Complete |

---

## How to Run

### Option 1: Main Benchmarking App (Original)
```bash
cd "/mnt/c/Users/SEHGALS/Langgraph Projects/Langgraph_Project1"
python app.py
# Opens at http://127.0.0.1:7860
```

### Option 2: Template Management App (NEW)
```bash
cd "/mnt/c/Users/SEHGALS/Langgraph Projects/Langgraph_Project1"
python template_app.py
# Opens at http://127.0.0.1:7870
```

### Option 3: CLI
```bash
source ./venv/Scripts/activate
python main.py --mode perplexity --stream
```

---

## Plug-and-Play Template System (Session 4)

### Architecture

```
User uploads .pptx template
         │
         ▼
┌─────────────────────────────┐
│  1. Template Analyzer       │  (python-pptx)
│  - Discover all shapes      │
│  - Extract table structures │
│  - Find text placeholders   │
└─────────────────────────────┘
         │
         ▼ JSON structure
┌─────────────────────────────┐
│  2. Template Mapping Agent  │  (LLM or rule-based)
│  - Analyze shape names/text │
│  - Map to vehicle data      │
│  - Generate field mapping   │
└─────────────────────────────┘
         │
         ▼ Mapping config JSON
┌─────────────────────────────┐
│  3. Dynamic PPT Generator   │
│  - Load mapping config      │
│  - Apply data to template   │
│  - Generate presentation    │
└─────────────────────────────┘
```

### How to Use Custom Templates

**Via UI (template_app.py):**
1. Upload any `.pptx` template
2. Click "Analyze Template"
3. Click "Generate Mapping" (uses LLM if API key available)
4. Edit the JSON mapping if needed
5. Save with a name
6. Test by generating a sample presentation

**Via Code:**
```python
from src.tools.template_analyzer import TemplateAnalyzer
from src.agents.template_analysis_agent import TemplateAnalysisAgent
from src.tools.dynamic_ppt_generator import DynamicOEMPresentationGenerator

# 1. Analyze template
analyzer = TemplateAnalyzer()
analysis = analyzer.analyze_for_mapping("my_template.pptx")

# 2. Generate mapping (LLM or rule-based)
agent = TemplateAnalysisAgent()
mapping = agent.generate_mapping(analysis)  # or generate_mapping_manual()

# 3. Save mapping
from src.config.template_registry import TemplateRegistry
registry = TemplateRegistry()
registry.save_mapping("my_template", mapping)

# 4. Generate presentations
generator = DynamicOEMPresentationGenerator("src/config/template_schemas/my_template.json")
result = generator.generate_from_scraping_result(scraping_result)
```

### Mapping Config Format
```json
{
    "template_name": "custom_template",
    "template_hash": "abc123",
    "shape_mappings": [
        {
            "shape_id": 2,
            "shape_name": "Title",
            "type": "text",
            "data_field": "oem_info.oem_name",
            "format": "{value}",
            "default": "Unknown OEM"
        },
        {
            "shape_id": 8,
            "shape_name": "Specs Table",
            "type": "table",
            "row_mappings": [
                {"row_index": 0, "data_field": "products[].name"},
                {"row_index": 1, "data_field": "products[].battery", "format": "{value} kWh"}
            ]
        },
        {
            "shape_id": 25,
            "type": "bullet_list",
            "data_field": "computed_fields.expected_highlights",
            "max_items": 4
        }
    ]
}
```

---

## LangGraph 1.0 Compatibility (Session 4 Analysis)

### Current Status
| Component | Status | Notes |
|-----------|--------|-------|
| langgraph version | >=1.0.3 | Latest is 1.0.4 |
| StateGraph API | Compatible | No changes needed |
| MemorySaver checkpointer | Compatible | Correct pattern used |
| Message reducer | Defined but unused | Can implement later |
| Human-in-the-loop | NOT implemented | Uses new `interrupt()` function |
| Auto-summarization | NOT implemented | Uses middleware approach |

### Future: Human-in-the-Loop Pattern
```python
from langgraph.types import Command, interrupt

def validation_node(state):
    if quality_score < threshold:
        feedback = interrupt("Review required: quality score below threshold")
        # Resume with: Command(resume=user_feedback)
        return {"human_feedback": feedback}
```

### Future: Auto-Summarization
```python
from langchain.agents.middleware import SummarizationMiddleware

middleware = SummarizationMiddleware(
    model="gpt-4o-mini",
    max_tokens_before_summary=4000,
    messages_to_keep=20
)
```

---

## Data Schemas

### Vehicle Data (for template mapping)
```python
{
    "oem_info": {
        "oem_name": str,
        "country": str,
        "website": str,
        "category": str
    },
    "products": [
        {
            "name": str,
            "wheel_formula": str,
            "gvw_gcw": str,  # formatted: "40,000 kg GVW"
            "range": str,     # formatted: "500 km"
            "battery": str,   # formatted: "600 kWh"
            "charging": str,  # formatted: "400 kW DC"
            "performance": str,
            "powertrain": str,
            "sop": str,
            "markets": str,
            "application": str
        }
    ],
    "computed_fields": {
        "expected_highlights": List[str],
        "assessment": List[str],
        "technologies": List[str]
    }
}
```

### VehicleSpecifications (raw scraped data)
```python
vehicle_name, oem_name, source_url, category, powertrain_type
battery_capacity_kwh, battery_capacity_min_kwh, battery_voltage_v
motor_power_kw, motor_torque_nm
range_km, range_min_km
dc_charging_kw, gvw_kg, payload_capacity_kg
additional_specs: Dict[str, Any]
data_completeness_score: float  # 0.0 to 1.0
```

---

## API Keys Required

```env
# .env file in project root
OPENAI_API_KEY=sk-proj-...      # For intelligent mode, validation, template mapping
PERPLEXITY_API_KEY=pplx-...     # For perplexity scraping mode
```

---

## Development History

### Session 1: November 25, 2024 - MVP Complete
- LangGraph workflow with 3 nodes (scrape, validate, present)
- Perplexity API integration
- IAA PowerPoint template generator
- End-to-end working

### Session 2: December 2, 2024 - Improved Scraper
- JSON-based extraction prompt (better than markdown)
- Unit conversion heuristics (`_ensure_kg`)
- Min/max range handling
- Completeness improved: 0.5 → 0.875

### Session 3: December 2, 2024 - Gradio Frontend
- `app.py` - Main web UI
- Auto-fallback between scraping modes
- Provider name hiding
- GVW field fix

### Session 4: December 3, 2024 - Plug-and-Play Templates
**Goal**: Allow users to upload ANY PowerPoint template and have AI generate mappings.

**Key Decisions Made**:
1. **Non-destructive add-on**: All new files, existing code untouched
2. **JSON mapping configs**: Safer than generated Python code
3. **Separate UI**: `template_app.py` runs on port 7870

**Files Created**:
| File | Lines | Purpose |
|------|-------|---------|
| `src/tools/template_analyzer.py` | ~300 | Discovers template shapes/tables |
| `src/tools/dynamic_ppt_generator.py` | ~400 | JSON-config based generator |
| `src/agents/template_analysis_agent.py` | ~350 | LLM + rule-based mapping |
| `src/config/template_registry.py` | ~250 | CRUD for mappings |
| `src/config/template_schemas/iaa_template.json` | ~100 | IAA reference mapping |
| `template_app.py` | ~300 | Gradio UI for templates |

**Test Results**:
- Template analysis: 22 shapes discovered
- Mapping generation: 6 shape mappings created
- PPT generation: Success, 6 shapes populated

### Session 5: December 3, 2024 - Multi-Slide Template Support
**Goal**: Enable templates with multiple slides, supporting overflow/pagination for arrays.

**User Requirements**:
- One OEM per presentation file (current behavior preserved)
- Full slide layout duplication for overflow slides

**Key Changes**:
1. **JSON Schema v2.0.0**: Added `slide_index`, `supports_multi_slide`, `pagination_rules`
2. **DynamicPPTGenerator refactor**: Multi-slide generation with full slide duplication
3. **Template analysis agent**: Generates slide-aware mappings
4. **template_app.py**: Shows multi-slide info in UI

**Files Modified**:
| File | Changes |
|------|---------|
| `src/tools/dynamic_ppt_generator.py` | Major refactor: `_duplicate_slide()`, `_calculate_overflow_slides()`, `_group_mappings_by_slide()` |
| `src/config/template_schemas/iaa_template.json` | Added `version: 2.0.0`, `supports_multi_slide`, `pagination_rules`, `slide_index` per shape |
| `src/agents/template_analysis_agent.py` | Updated prompt and rule-based generation for multi-slide |
| `template_app.py` | Shows multi-slide info, 5-product test data |

**Test Results**:
- 5 products → 3 slides created (2+2+1 pagination)
- 18 shapes populated across slides
- Full slide duplication working
- Backwards compatible with v1.0.0 mappings

**New JSON Schema Features**:
```json
{
  "version": "2.0.0",
  "supports_multi_slide": true,
  "pagination_rules": {
    "products": {
      "items_per_slide": 2,
      "overflow_behavior": "duplicate_slide",
      "source_slide_index": 0
    }
  },
  "shape_mappings": [
    {
      "shape_id": 8,
      "slide_index": 0,
      "type": "table",
      "repeatable_for": "products",
      "max_products": 2
    }
  ]
}
```

### Session 6: December 4, 2024 - Async Parallel URL Processing
**Goal**: Process multiple URLs simultaneously to significantly reduce extraction time.

**User Requirements**:
- Auto-detect mode: parallel for 2+ URLs, sequential for single URL
- Semaphore-based rate limiting (simple, not overcomplicated)
- Simple progress summary ("Processing 5 URLs in parallel...")
- Must NOT affect extraction quality

**Key Changes**:
1. **New dependencies**: Added `aiohttp>=3.9.0` for async HTTP, `nest-asyncio>=1.6.0` for nested event loop compatibility
2. **AsyncRateLimiter class**: Semaphore-based rate limiting for API calls
3. **Async extraction methods**: `_extract_oem_data_async()`, `_extract_legacy_async()`, `_extract_intelligent_async()`
4. **Parallel URL processing**: `_process_urls_parallel()` with `asyncio.gather()`
5. **Auto-detect in `process_urls()`**: Switches between parallel/sequential based on URL count
6. **Parallel page crawling**: Updated `_crawl_pages()` to crawl pages concurrently
7. **Event loop fix**: Applied `nest_asyncio.apply()` at module load to fix LiteLLM/Gradio compatibility

**Files Modified**:
| File | Changes |
|------|---------|
| `requirements.txt` | Added `aiohttp>=3.9.0`, `nest-asyncio>=1.6.0` |
| `src/tools/scraper.py` | Major additions: `AsyncRateLimiter`, async extraction methods, `_process_urls_parallel()`, updated `process_urls()`, parallel `_crawl_pages()`, `nest_asyncio.apply()` |

**New Configuration Parameters** (in `ScraperConfig`):
```python
PARALLEL_URL_THRESHOLD = 2      # Use parallel for 2+ URLs
MAX_CONCURRENT_URLS = 3         # Max URLs processed simultaneously
MAX_CONCURRENT_CRAWLS = 5       # Max concurrent page fetches
MAX_CONCURRENT_API_CALLS = 3    # Max concurrent API calls
ASYNC_TIMEOUT_SECONDS = 300     # 5 min timeout per URL
```

**Expected Performance Improvement**:
| URLs | Before | After |
|------|--------|-------|
| 1 | ~30-60s | Same (sequential) |
| 3 | ~2-3 min | ~40-80s (2-3x faster) |
| 5 | ~4-5 min | ~1-2 min (2-3x faster) |

**Backwards Compatibility**:
- Public API unchanged: `extractor.process_urls(urls)` works identically
- `scraping_agent.py` needs NO changes
- Single URL uses original sequential code path

---

## Learnings & Best Practices

### 1. Prompt Engineering
- **JSON > Markdown**: LLMs extract structured data more reliably with JSON schema
- **Explicit units**: "28 tons = 28000 kg, NOT 28 kg"
- **Handle ranges**: Separate fields for min/max

### 2. Template System Design (Session 4)
- **Why JSON over Python code generation**:
  - Safer: No code execution risk
  - Editable: Users can tweak mappings
  - Debuggable: Easy to inspect
  - Portable: Can export/import

- **Shape discovery with python-pptx**:
  ```python
  for shape in slide.shapes:
      print(f"ID: {shape.shape_id}, Name: {shape.name}, Type: {shape.shape_type}")
      if shape.has_table:
          # Analyze table structure
      if hasattr(shape, 'text_frame'):
          # Extract text content
  ```

### 3. LangGraph Patterns
- Nodes return dicts that merge with state
- Use enums for status/mode
- Conditional routing via functions

### 4. Gradio on Windows
- Use `server_name="127.0.0.1"` NOT `0.0.0.0`
- Implement auto port detection
- `app.launch()` is blocking - expected

### 5. Non-Destructive Development
- When adding major features, create NEW files
- Don't modify working code unless necessary
- Keep backwards compatibility

---

## Known Issues & Limitations

### Current Limitations
1. ~~**Single-slide templates only**~~: Multi-slide support implemented in Session 5!
2. **No image placeholders**: Can't map images to template
3. **No chart generation**: Tables only
4. ~~**Sequential URL processing**~~: Async parallel processing implemented in Session 6!

### Workarounds
- For images: Manually add after generation
- For charts: Use Excel/external tools

---

## Future Roadmap

### Completed
- [x] Web UI for non-technical users (Gradio)
- [x] Plug-and-play template system
- [x] LangGraph 1.0 compatibility analysis
- [x] Multi-slide template support (Session 5)
- [x] Async parallel URL processing (Session 6)

### Phase 2: Enterprise Features (Next)
- [ ] Human-in-the-loop (`interrupt()` function)
- [ ] Auto-summarization middleware
- [ ] SQLite/PostgreSQL persistence
- [ ] Structured logging (loguru)

### Phase 3: Template Enhancements
- [x] Multi-slide template support (completed Session 5)
- [ ] Image placeholder mapping
- [ ] Chart generation from data
- [ ] Template validation (required fields check)

### Phase 4: Production Hardening
- [ ] pytest test suite
- [ ] Circuit breakers / exponential backoff
- [ ] REST API endpoint
- [ ] Real-time progress streaming

### Phase 5: Advanced
- [ ] Provider abstraction (Anthropic, Gemini, local)
- [ ] PDF extraction support
- [ ] Template marketplace

---

## Quick Reference Commands

```bash
# Run Main Benchmarking App
python app.py
# http://127.0.0.1:7860

# Run Template Management App
python template_app.py
# http://127.0.0.1:7870

# CLI with perplexity mode
python main.py --mode perplexity --stream

# Test template analyzer
python src/tools/template_analyzer.py templates/IAA_Template.pptx

# Test dynamic generator
python src/tools/dynamic_ppt_generator.py

# List available templates
python src/config/template_registry.py

# Test end-to-end plug-and-play
python -c "
from src.tools.dynamic_ppt_generator import DynamicOEMPresentationGenerator
gen = DynamicOEMPresentationGenerator('src/config/template_schemas/iaa_template.json')
print(gen.mapping['template_name'])
"
```

---

## Important Notes for Claude

### Environment
1. **Always use venv**: `./venv/Scripts/python.exe` on Windows WSL
2. **Path format**: `/mnt/c/Users/SEHGALS/Langgraph Projects/Langgraph_Project1`
3. **Python imports**: Always use `sys.path.insert(0, '.')` for standalone scripts

### Key Behaviors
4. **Template file**: `templates/IAA_Template.pptx` exists and works
5. **Output folder**: `outputs/` in project root
6. **Default scraping mode**: Perplexity (better extraction)
7. **Check completeness**: Look at `data_completeness_score`

### Gotchas
8. **Gradio on Windows**: Use `server_name="127.0.0.1"`
9. **LLM field names**: Use fallback pattern: `data.get('gvw_kg') or data.get('gvwr_kg')`
10. **No provider names**: User requested hiding LLM provider names in UI
11. **Nested event loops**: LiteLLM/Crawl4AI have internal async queues - using `asyncio.run()` from Gradio causes "Queue bound to different event loop" errors. Fixed with `nest_asyncio.apply()` at module load.

### Two Systems
12. **Original PPT generator**: `src/tools/ppt_generator.py` (hardcoded IAA)
13. **New dynamic generator**: `src/tools/dynamic_ppt_generator.py` (any template)
14. **Both work independently**: Original system untouched

### Template System
15. **Mapping configs**: `src/config/template_schemas/*.json`
16. **Registry manages**: List, load, save, validate mappings
17. **Two UI apps**: `app.py` (main) and `template_app.py` (templates)

---

## File Structure

```
Langgraph_Project1/
├── app.py                          # Main Gradio UI (benchmarking)
├── template_app.py                 # NEW: Template Management UI
├── main.py                         # CLI entry point
├── CLAUDE.md                       # This file - project context
├── .env                            # API keys (not in git)
├── requirements.txt                # Python dependencies
│
├── templates/
│   └── IAA_Template.pptx           # Original PowerPoint template
│
├── outputs/                        # Generated files
│   ├── *.pptx                      # Presentations
│   ├── *.json                      # Scraping results
│   └── *_analysis.json             # Template analysis files
│
├── src/
│   ├── graph/
│   │   └── runtime.py              # LangGraph workflow
│   │
│   ├── state/
│   │   └── state.py                # State definitions
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py             # Pydantic settings
│   │   ├── template_registry.py    # NEW: Template CRUD
│   │   └── template_schemas/       # NEW: Mapping JSON files
│   │       ├── iaa_template.json
│   │       └── iaa_template_mapping.json
│   │
│   ├── agents/
│   │   ├── scraping_agent.py
│   │   ├── quality_validator.py
│   │   ├── presentation_generator.py
│   │   └── template_analysis_agent.py  # NEW: LLM mapping agent
│   │
│   ├── tools/
│   │   ├── scraper.py              # EPowertrainExtractor
│   │   ├── ppt_generator.py        # Original IAA generator
│   │   ├── template_analyzer.py    # NEW: Template discovery
│   │   └── dynamic_ppt_generator.py # NEW: JSON-config generator
│   │
│   └── inputs/
│       └── urls.txt                # Default URLs
│
└── venv/                           # Python virtual environment
```

---

## Session Handoff Checklist

When starting a new session, Claude should:

1. **Read this CLAUDE.md** for full context
2. **Check git status** for any uncommitted changes
3. **Verify environment**: `./venv/Scripts/python.exe --version`
4. **Test imports**: `python -c "from src.graph.runtime import run_benchmark; print('OK')"`

### If Working on Templates
5. **Check template registry**: `python src/config/template_registry.py`
6. **Test analyzer**: `python src/tools/template_analyzer.py templates/IAA_Template.pptx`

### If Working on Scraping/Main App
7. **Test main app**: `python app.py` (check http://127.0.0.1:7860)
8. **Check outputs folder**: `ls outputs/`

---

**Next Session Suggestions**:
1. Test async parallel URL processing with 3+ URLs to verify performance gains
2. Implement human-in-the-loop with `interrupt()` function
3. Add auto-summarization middleware
4. Integrate template selection into main app.py
5. pytest test suite for template system
6. Image placeholder mapping support
