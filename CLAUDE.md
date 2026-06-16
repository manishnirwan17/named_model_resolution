# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Data-Aware Model Router + Insights Pipeline

A **heuristics-based, data-aware config-builder and model router** that ingests gold-layer datamarts, classifies their schemas, routes each dataset to the correct ML model (BOCPD / MMM / PSI / ARIMA), runs a quality gate, executes model pipelines, and emits a structured JSON payload (`InsightsPayload`) for LLM summarisation.

**Primary deployment target: Databricks.** Local execution is supported for development only.

**Design principle:** Deterministic, rule-based, reproducible. No ML training in the build path. Semantic embeddings are an optional future supplement only.

---

## Development Setup

Single `pyproject.toml` at repo root. All packages — `named_model_resolution/`, `orchestrator/`, `insights_runner/`, and `pipeline/` — are Python packages directly at repo root. No `src/` layer, no submodules.

```bash
pip install -e ".[pipeline]"                  # full install: router + quality gate + BOCPD/MMM
pip install -e .                              # router + quality gate only (no PyMC/arviz)
python main.py --help
pytest
pytest tests/test_catalog_parser.py          # single test file
```

Quick verification:
```bash
python -c "
from named_model_resolution.catalog_parser import parse_catalog
c = parse_catalog('pharma_knowledge_base/gold_layer_datamarts.csv')
print(list(c.datamarts.keys()))
"

python -c "
from named_model_resolution.column_matcher import ColumnMatcher
m = ColumnMatcher('pharma_knowledge_base/configs')
w = []
specs = m.match_all({'WK_END':'date','TRX':'float','F2F':'int','MYSTERY':'float'}, warnings=w)
for s in specs: print(s.name, '->', s.semantic_subtype, f'({s.match_source})')
print('WARNINGS:', w)
"
```

---

## Databricks Setup

### Installation (single package, one command)

Everything — router, quality gate, BOCPD pipeline, MMM pipeline — is in one repo with one `pyproject.toml`. The `[pipeline]` optional extra pulls in the heavier model deps.

**`%pip` pattern (preferred, auto-restarts kernel):**
```python
# Cell 1 — must be first cell in the notebook
repo = "/Workspace/Users/your-user/named_model_resolution"
%pip install -e {repo}[pipeline] --quiet
```

**`subprocess` pattern (no restart — jobs, shared setup cells):**
```python
import subprocess, sys, importlib

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
_repo = "/Workspace" + "/".join(_nb_path.split("/")[:-1])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{_repo}[pipeline]", "--quiet"])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
importlib.invalidate_caches()
```

### Connector for Unity Catalog

Use `SQLCatalogConnector` with `databricks-sql-connector` (install separately):
```python
from sqlalchemy import create_engine
from orchestrator.connectors import SQLCatalogConnector

engine = create_engine(
    "databricks+connector://token@<workspace-host>/<http-path>",
    connect_args={"catalog": "nexora_poc_catalog", "schema": "gold"},
)
connector = SQLCatalogConnector(engine, schema="gold")
```

### Cluster requirements

- **Router + quality gate only:** any cluster (no special compute needed)
- **BOCPD runner:** single-node, standard memory, Python-only (no Spark needed)
- **MMM runner (PyMC/NUTS):** single-node, high-memory or GPU. Multi-node Spark clusters will NOT distribute NUTS sampling.

### Key env vars read by `insights_generation/pipeline/config.py`

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABRICKS_RUNTIME_VERSION` | (auto-set) | Triggers UC-aware I/O mode |
| `UC_CATALOG` | `nexora_poc_catalog` | Unity Catalog catalog name |
| `UC_SCHEMA` | `gold` | Schema for input + output tables |
| `UC_VOLUME` | `model_artifacts` | Volume path for .nc trace files |
| `GOLD_LABELLED_TABLE` | `{catalog}.gold.engagement_mmm_labelled` | Override input table |
| `MODEL_OUTPUT_SCHEMA` | same as `UC_SCHEMA` | Override output schema |

These are only used by the standalone `insights_generation` pipeline scripts (not by `insights_runner`, which constructs its own `DatasetConfig` from `RouterResult`).

### `dataset_config.json` on Databricks

`pipeline/config.py` loads `dataset_config.json` on first import. If absent, it auto-detects schema from the gold table (samples 10% via Spark). After the `insights_generation/pipeline/` → `pipeline/` move, this file is now resolved relative to **repo root** (`Path(__file__).parent.parent` = repo root when `pipeline/` is at root level). On Databricks:
- Pre-commit a tuned `dataset_config.json` to the repo root, or
- Let it auto-detect on first notebook run (takes 30-60s for large tables)

**`insights_runner` does NOT use the global `DATASET` from `pipeline/config.py`** — it builds its own `DatasetConfig` from `RouterResult` column subtypes. The `dataset_config.json` is only needed when running the standalone pipeline scripts (`pipeline/data_prep.py`, `pipeline/bocpd.py`, etc.) directly.

---

## Repository Layout

```
pharma_knowledge_base/
  gold_layer_datamarts.csv        spec: 10 gold-layer datamarts (multi-section flat file)
  configs/
    candidates.yaml               geography/date/measure/segment/channel candidate lists
    abbreviations.yaml            WK_END->week_end_date, TRX->trx, etc.
    model_routing.yaml            per-model routing rules (add new models here)
    transform_rules.yaml          statistical transform heuristics

named_model_resolution/           Python package (directly at repo root)
  models.py                       all dataclasses (ColumnSpec, RouterResult, etc.)
  catalog_parser.py               parse gold_layer_datamarts.csv -> DatamartCatalog
  column_matcher.py               Layer 2: fuzzy + abbreviation-aware matching
  classifier.py                   Layer 1: table -> fact | dimension
  config_assembler.py             Layer 3: fact + dims + routing rules -> ModelConfig list

orchestrator/                     Python package (directly at repo root)
  connectors/
    base.py                       CatalogConnector Protocol (platform-agnostic)
    file_connector.py             CSV/Parquet from a directory
    sql_connector.py              SQLAlchemy (Databricks SQL, Postgres, etc.)
  profiler.py                     sample N rows -> ColumnProfile (skewness, grain, transforms)
  router.py                       crawl -> classify -> route -> profile -> RouterResult list
  pipelines/
    base.py                       ModelPipeline ABC
    bocpd_pipeline.py / mmm_pipeline.py / psi_pipeline.py / arima_pipeline.py
    __init__.py                   PIPELINE_REGISTRY dict

insights_runner/                  Python package: quality gate + model bridge layer
  pipeline.py                     entry point: run(connector, router_result, ...) -> InsightsPayload
  quality_gate/
    thresholds.yaml               all quality thresholds per model (config-not-code)
    models.py                     QualityCheckResult, ModelQualityReport, QualityReport
    checks.py                     pure check functions (fill_rate, collinearity, ACF, ...)
    assessor.py                   QualityAssessor + CHECK_REGISTRY (data-driven dispatch)
  runners/
    __init__.py                   RUNNER_REGISTRY dict
    base.py                       ModelRunner ABC
    bocpd_runner.py               bridges RouterResult -> insights_generation BOCPD pipeline
    mmm_runner.py                 bridges RouterResult -> insights_generation MMM pipeline
    psi_runner.py / arima_runner.py   stubs (quality gate runs; pipeline not yet implemented)
  output/
    models.py                     InsightsPayload dataclass + .to_json()
    builder.py                    build() assembles all parts into InsightsPayload

pipeline/                         Python package at repo root (installed via [pipeline] optional extra)
  bocpd.py                        run_bocpd(), extract_candidates()
  mmm_data_prep.py                transform_channels(mkt, dataset_config=None), build_model_matrix()
  mmm_fit.py                      build_pymc_model(X, y, ..., dataset_config=None)
  dataset_config.py               DatasetConfig + ChannelSpec dataclasses
  config.py                       global DATASET, UC-aware I/O, ON_DATABRICKS; dataset_config.json -> repo root
  data_prep.py                    aggregate_to_market(), add_features(), split()
  integration.py                  BOCPD + MMM signal integration / anomaly classification
  validation.py                   validation report generation
  insights.py                     LangGraph insights agent (future use)

main.py                           CLI entry point
```

---

## Architecture

### Layer 1 — Classify each table as fact vs dimension (`classifier.py`)

- **Fact** — >=1 date column AND >=1 measure/unclassified_metric column
- **Dimension** — >=1 key column AND majority of columns are dimension_attribute/flag, no date
- Tie-break: Jaccard overlap >= 0.5 against a DatamartCatalog entry -> inherit naming-convention type

### Layer 2 — Column semantic subtype matching (`column_matcher.py`)

Four-step pipeline per column:
1. Normalize (lowercase, special chars -> `_`)
2. Abbreviation expand (`abbreviations.yaml`) — `WK_END` -> `week_end_date`
3. Candidate list match (`candidates.yaml`) — token-boundary match (not greedy substring) -> subtype
4. Heuristic fallbacks:
   - Numeric indicator guard: `_idx`, `n_` prefixes suppress date matching
   - dtype = date/timestamp -> `date` (conf 0.9)
   - dtype = numeric, not a key -> `unclassified_metric` (guardrail — warning surfaced, passed to models)
   - dtype = string, low cardinality -> `dimension_attribute`
   - else -> `unknown`

Channel detection: `channel` subtype is checked before `measure` and `date`. MMM routing requires `channel` in required subtypes — pure sales tables do not route to MMM.

### Layer 3 — Assemble per-model configs (`config_assembler.py`)

Scores all models in `model_routing.yaml`:
```
score = required_count x 1.0 + optional_matched x 0.3 + preferred_measures_matched x 0.5
```
Star-schema join detection (1-hop): if a fact table has columns matching a dimension table's key, the dim's column subtypes are inherited before scoring. Snowflake multi-hop excluded.

### Quality gate + insights runner

```
RouterResult -> connector.sample_rows(n=5000) -> QualityAssessor.assess() per model
  FAIL -> skip model, record reason
  WARN -> run model with caveat
  PASS -> run model
       -> RUNNER_REGISTRY[model_name].run(df, router_result, model_config)
       -> OutputBuilder.build() -> InsightsPayload.to_json()
```

Quality checks are model-aware (different check sets per model, thresholds in `thresholds.yaml`).
Runners construct `DatasetConfig` from `RouterResult` column subtypes — no `dataset_config.json` needed.

---

## Model Routing + Extensibility

To add a new model to the **router** (3 steps, no core code changes):
1. Add a routing rule block to `pharma_knowledge_base/configs/model_routing.yaml`
2. Create `orchestrator/pipelines/<model>_pipeline.py` implementing `ModelPipeline`
3. Register in `orchestrator/pipelines/__init__.py` -> `PIPELINE_REGISTRY`

To add a new model to the **insights runner** (3 steps, no core code changes):
1. Add a threshold block to `insights_runner/quality_gate/thresholds.yaml`
2. Create `insights_runner/runners/<model>_runner.py` implementing `ModelRunner`
3. Register in `insights_runner/runners/__init__.py` -> `RUNNER_REGISTRY`

---

## Model Routing Reference

| Model | Required subtypes | Key gate | Primary use case |
|-------|------------------|----------|-----------------|
| **BOCPD** | date + measure | - | Weekly trend-shift / changepoint detection |
| **MMM** | date + measure + channel | channel required | Marketing mix attribution |
| **PSI** | date + measure + segment | - | Population / segment drift detection |
| **ARIMA** | date + measure | - | Seasonal trend forecasting (fallback) |

---

## Key Conventions

- **Config-not-code.** Candidate lists, abbreviations, routing rules, quality thresholds, and transform rules all live in YAML files. Rarely need to edit Python to handle new datasets.
- **`unclassified_metric` never dropped.** Flagged and passed to models that accept generic measures. Warnings always surfaced in `RouterResult.warnings`.
- **Token-boundary matching, not greedy substring.** `"week" in "n_weeks"` was a false match. `column_matcher.py` uses token-set subset check. Numeric indicator guard (`_idx`, `n_`) further prevents false date classification.
- **MMM requires channel.** Adding `channel` to MMM's `required` list ensures pure sales/adherence tables don't route to MMM.
- **`insights_generation` functions accept explicit `dataset_config`.** `transform_channels()`, `build_model_matrix()`, and `build_pymc_model()` all accept `dataset_config=None` (falls back to global `DATASET`). The runners always pass an explicit config built from `RouterResult` — no global mutation.
- **String detail messages use ASCII only.** The quality gate check detail strings avoid Unicode (no `>=` as a unicode char) so they serialize cleanly on all platforms including Windows cp1252 and Databricks driver output.
- **Signal deduplication.** `Router.run(deduplicate=True)` groups fact tables by (routing_signature, top_model) and marks less-rich duplicates `is_duplicate_signal=True`. Skip these in the insights loop.
