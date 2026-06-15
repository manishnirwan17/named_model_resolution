# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Data-Aware Model Router + Config-Builder

A **heuristics-based, data-aware config-builder and model router** that ingests gold-layer datamarts, classifies their schemas, routes each dataset to the correct ML model (BOCPD / MMM / PSI / ARIMA / …), profiles sample data for statistical transformation recommendations, and drives per-model pipelines (detect use-cases → prep → transform → tune).

**Design principle:** Deterministic, rule-based, reproducible. No ML training in the build path. Semantic embeddings are an optional future supplement only.

---

## Development Setup

This project uses **`uv`** (Python 3.13) with a **workspace** of two installable packages.

```bash
uv sync                                  # install all workspace deps
uv run python main.py --help             # see CLI options
uv run python main.py \
    --catalog-type file \
    --catalog-path ./sample_data/ \
    --output-dir ./output/               # run the router on local files

uv run pytest                            # run all tests
uv run pytest tests/test_catalog_parser.py  # single test file
```

Quick verification snippets:
```bash
# Verify spec parsing → should print 10 datamart names
uv run python -c "
from named_model_resolution.catalog_parser import parse_catalog
c = parse_catalog('pharma_knowledge_base/gold_layer_datamarts.csv')
print(list(c.datamarts.keys()))
"

# Verify column matching with abbreviations + guardrail
uv run python -c "
from named_model_resolution.column_matcher import ColumnMatcher
m = ColumnMatcher('pharma_knowledge_base/configs')
w = []
specs = m.match_all({'WK_END':'date','TRX':'float','MYSTERY':'float','PROD_CODE':'object'}, warnings=w)
for s in specs: print(s.name, '->', s.semantic_subtype, f'({s.match_source}, conf={s.confidence})')
print('WARNINGS:', w)
"
```

---

## Repository Layout

```
pharma_knowledge_base/
  gold_layer_datamarts.csv        ← spec: 10 gold-layer datamarts (multi-section flat file)
  configs/
    candidates.yaml               ← geography/date/measure/segment candidate lists (expand here)
    abbreviations.yaml            ← WK_END→week_end_date, TRX→trx, etc. (append here)
    model_routing.yaml            ← per-model routing rules (add new models here)
    transform_rules.yaml          ← statistical transform heuristics

named_model_resolution/           ← config-builder library (uv workspace member)
  src/named_model_resolution/
    models.py                     ← all dataclasses (ColumnSpec, RouterResult, etc.)
    catalog_parser.py             ← parse gold_layer_datamarts.csv → DatamartCatalog
    column_matcher.py             ← Layer 2: fuzzy + abbreviation-aware matching
    classifier.py                 ← Layer 1: table → fact | dimension
    config_assembler.py           ← Layer 3: fact + dims + routing rules → ModelConfig list

orchestrator/                     ← orchestration package (uv workspace member)
  src/orchestrator/
    connectors/
      base.py                     ← CatalogConnector Protocol (platform-agnostic)
      file_connector.py           ← CSV/Parquet from a directory
      sql_connector.py            ← SQLAlchemy (Databricks SQL, Postgres, etc.)
    profiler.py                   ← sample N rows → ColumnProfile (skewness, grain, transforms)
    router.py                     ← crawl → classify → route → profile → RouterResult list
    pipelines/
      base.py                     ← ModelPipeline ABC (detect → prep → transform → tune)
      bocpd_pipeline.py
      mmm_pipeline.py
      psi_pipeline.py
      arima_pipeline.py
      __init__.py                 ← PIPELINE_REGISTRY dict

main.py                           ← CLI entry point
```

---

## Architecture

### Layer 1 → Classify each table as fact vs dimension (`classifier.py`)

- **Fact** → ≥1 date column AND ≥1 measure/unclassified_metric column
- **Dimension** → ≥1 key column AND majority of columns are dimension_attribute/flag, no date
- Tie-break: Jaccard overlap ≥ 0.5 against a DatamartCatalog entry → inherit naming-convention type

### Layer 2 → Column semantic subtype matching (`column_matcher.py`)

Four-step pipeline per column:
1. Normalize (lowercase, special chars → `_`)
2. Abbreviation expand (`abbreviations.yaml`) — `WK_END` → `week_end_date`, `NBRX` → `nbrx`
3. Candidate list match (`candidates.yaml`) — exact/substring → `date | geography | measure | segment | key | flag`
4. Heuristic fallbacks:
   - Token overlap with candidate lists → partial match (conf 0.6)
   - dtype = date/timestamp → `date` (conf 0.9)
   - dtype = numeric, not a key → **`unclassified_metric`** (guardrail gate — logged warning, passed to models)
   - dtype = string, low cardinality → `dimension_attribute`
   - else → `unknown` (logged warning, not used for routing)

### Layer 3 → Assemble per-model configs (`config_assembler.py`)

Scores all models in `model_routing.yaml` against each fact table's column subtypes:
```
score = required_count × 1.0 + optional_matched × 0.3 + preferred_measures_matched × 0.5
```
Returns a ranked list of `ModelConfig`s (all candidates, not just top-1). Callers decide how many to use.

---

## Model Routing + Pipeline Expansion

To add a new model (e.g., Prophet, NeuralProphet, Causal Impact):
1. Add a routing rule block to `pharma_knowledge_base/configs/model_routing.yaml`
2. Create `orchestrator/src/orchestrator/pipelines/<model>_pipeline.py` implementing `ModelPipeline`
3. Register it in `orchestrator/src/orchestrator/pipelines/__init__.py` → `PIPELINE_REGISTRY`

No changes to the router, classifier, or column matcher are needed.

---

## Model Routing Reference

| Model | Required subtypes | Primary use case |
|-------|------------------|-----------------|
| **BOCPD** | date + measure | Weekly trend-shift / changepoint detection |
| **MMM** | date + measure | Marketing mix attribution |
| **PSI** | date + measure + segment | Population / segment drift detection |
| **ARIMA** | date + measure | Seasonal trend forecasting (fallback) |

---

## Key Conventions

- **Never bury config in code.** Candidate lists, abbreviations, routing rules, and transform thresholds all live in `pharma_knowledge_base/configs/` YAML files. Code reads them at runtime.
- **`unclassified_metric` columns are never dropped.** Any numeric column that can't be matched gets flagged and passed to models that accept generic measures. Warnings are surfaced in `RouterResult.warnings`.
- **Catalog spec parsing is multi-section aware.** `gold_layer_datamarts.csv` has 3 logical table blocks stacked vertically with section-header rows. `catalog_parser.py` handles this; `read_csv()` alone cannot.
- **Connector is platform-agnostic.** Add `SQLCatalogConnector(engine)` for any SQLAlchemy-compatible source (including Databricks via `databricks-sql-connector`). A Databricks-specific connector can be added later without touching the router.
- **Statistical transforms are data-driven.** The `Profiler` samples rows at runtime and applies `transform_rules.yaml` thresholds (skewness > 1.5 → log-transform, etc.). Suggestions live in `ColumnProfile.suggested_transforms`.
