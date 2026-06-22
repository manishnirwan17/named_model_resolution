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

### Key env vars read by `pipeline/config.py`

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABRICKS_RUNTIME_VERSION` | (auto-set) | Triggers UC-aware I/O mode |
| `UC_CATALOG` | `nexora_poc_catalog` | Unity Catalog catalog name |
| `UC_SCHEMA` | `gold` | Schema for input + output tables |
| `UC_VOLUME` | `model_artifacts` | Volume path for .nc trace files |
| `GOLD_LABELLED_TABLE` | `{catalog}.gold.engagement_mmm_labelled` | Override input table |
| `MODEL_OUTPUT_SCHEMA` | same as `UC_SCHEMA` | Override output schema |

These are only used by the standalone `pipeline/` scripts (not by `insights_runner`, which constructs its own `DatasetConfig` from `RouterResult`).

### `dataset_config.json` on Databricks

`pipeline/config.py` loads `dataset_config.json` from **repo root** on first import (`Path(__file__).parent.parent` = repo root). If absent:
- On Databricks: auto-detects schema from the gold table (samples 10% via Spark, takes 30-60s)
- Locally: falls back silently to an empty placeholder — runners always supply `dataset_config` explicitly, so this is harmless

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
                                  slices df to candidate window per model before quality gate + runner
  quality_gate/
    thresholds.yaml               all quality thresholds + candidate_window_years per model
    models.py                     QualityCheckResult, ModelQualityReport, QualityReport
    checks.py                     pure check functions (fill_rate, collinearity, ACF, ...)
    assessor.py                   QualityAssessor + CHECK_REGISTRY (data-driven dispatch)
  runners/
    __init__.py                   RUNNER_REGISTRY dict
    base.py                       ModelRunner ABC
    bocpd_runner.py               bridges RouterResult -> pipeline/ BOCPD; emits cp_context_windows
    mmm_runner.py                 bridges RouterResult -> pipeline/ MMM pipeline
    psi_runner.py / arima_runner.py   stubs (quality gate runs; pipeline not yet implemented)
  output/
    models.py                     InsightsPayload dataclass + .to_json()
    builder.py                    build() assembles all parts into InsightsPayload

pipeline/                         Python package at repo root (installed via [pipeline] optional extra)
  bocpd.py                        run_bocpd(), extract_candidates()
  mmm_data_prep.py                transform_channels(mkt, dataset_config=None), build_model_matrix()
  mmm_fit.py                      build_pymc_model(X, y, ..., dataset_config=None)
  dataset_config.py               DatasetConfig + ChannelSpec dataclasses
  config.py                       global DATASET (placeholder locally), UC-aware I/O, ON_DATABRICKS
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
2. Abbreviation expand (`abbreviations.yaml`) — `WK_END` -> `week_end_date`, `G1_MEETING_TYPE` -> `meeting_type`
3. Candidate list match (`candidates.yaml`) — two short-circuit guards run first, then token-boundary match:
   - **Date-suffix override** — last token in `{date, dt, datetime, timestamp, ts}` → always `date` (catches `CALL_OR_RTE_SENT_DATE`)
   - **Rolling-metric guard** — name matches `_last_\d+[dD]$` (e.g. `f2f_last_90d`) → skip all candidate matching; falls to dtype heuristic (numeric → `unclassified_metric`)
   - Then exact match → token-boundary subset match → subtype assigned
4. Heuristic fallbacks:
   - Numeric indicator guard: `_idx`, `n_` prefixes suppress date matching
   - dtype = date/timestamp -> `date` (conf 0.9)
   - dtype = numeric, not a key -> `unclassified_metric` (guardrail — warning surfaced, passed to models)
   - dtype = string, low cardinality -> `dimension_attribute`
   - else -> `unknown`

Channel detection: `channel` subtype is checked before `measure` and `date`. MMM routing requires `channel` in required subtypes — pure sales tables do not route to MMM.

**MMM string-channel guard.** `mmm_runner._build_dataset_config` skips any channel column whose dtype is VARCHAR/string — only numeric columns can receive adstock/decay transforms.

### Layer 3 — Assemble per-model configs (`config_assembler.py`)

Scores all models in `model_routing.yaml`:
```
score = required_count x 1.0 + optional_matched x 0.3 + preferred_measures_matched x 0.5
```
Star-schema join detection (1-hop): if a fact table has columns matching a dimension table's key, the dim's column subtypes are inherited before scoring. Snowflake multi-hop excluded.

### Quality gate + insights runner

```
RouterResult -> connector.sample_rows(n=5000) -> per model:
  1. Slice df to [max_date - candidate_window_years, max_date]   (thresholds.yaml per model)
     - date column selected by select_date_column() scoring (not first-seen)
     - parse_dates_flexible() handles ISO, YYYY-MM, "2025 Q3" quarter strings
  2. QualityAssessor.assess(df_windowed)
       FAIL -> skip model, record reason
       WARN/PASS -> run model
  3. RUNNER_REGISTRY[model_name].run(df_windowed, router_result, model_config)
     - normalize_to_series() detects segment/geography extra-grain and does 2-step aggregation
       (step 1: collapse within-segment; step 2: national roll-up + nunique count column)
  4. OutputBuilder.build() -> InsightsPayload.to_json()
```

Quality checks and runners both operate on the same windowed slice. `candidate_window` info (years, cutoff_date, max_date, n_rows) is embedded in each model's signal block.

**Registered quality checks:**

| Check | Models | Notes |
|-------|--------|-------|
| `fill_rate` | all | Best-viable logic: at least ONE date and ONE measure must pass (not all). Pool includes `measure + unclassified_metric + channel`. |
| `zero_variance` | all | CV < 0.01 → WARN |
| `date_continuity` | BOCPD, PSI, ARIMA | Per-grain `min_periods` dict (daily/weekly/monthly/quarterly). Profiler grain preferred; computed from gap median as fallback. |
| `channel_collinearity` | MMM | Pairwise Pearson; high collinearity → WARN |
| `segment_balance` | PSI | Iterates **all** segment columns; returns PASS on first viable one |
| `autocorrelation` | ARIMA | Lag-1 ACF check |
| `min_row_count` | all | Hard floor from `global.min_row_count` |
| `distribution_shape` | BOCPD, MMM, ARIMA | FAIL if all measure columns are constant (unique_count≤2) or near-zero CV (<0.02); WARN if best measure has very few unique values |

**Grain-aware minimums** (in `thresholds.yaml` under `min_periods`):

| Grain | BOCPD | ARIMA |
|-------|-------|-------|
| daily | 30 | 120 |
| weekly | 8 | 16 |
| monthly | 12 | 24 |
| quarterly | 4 | 8 |

---

## BOCPD Output Structure

`BOCPDRunner` emits three signal keys:

| Key | Description |
|-----|-------------|
| `cp_candidates` | All detected changepoints with metadata (date, cp_prob, exp_run_length, week_idx) |
| `cp_context_windows` | Per-changepoint slice: ±`_CP_WINDOW_WEEKS` (default 8) rows around each CP, with log and raw measure values |
| `cp_probs_series` | Full week-by-week cp_prob + exp_run_length for the entire candidate window (exhaustive record) |

Column keys in `cp_context_windows` are named after the actual measure column: `log_{measure_col}`, `{measure_col}_at_cp`, etc. — never hardcoded to "sales" or "TRX".

To adjust the context window half-width: edit `_CP_WINDOW_WEEKS` at the top of `insights_runner/runners/bocpd_runner.py`.

---

## Model Routing + Extensibility

To add a new model to the **router** (3 steps, no core code changes):
1. Add a routing rule block to `pharma_knowledge_base/configs/model_routing.yaml`
2. Create `orchestrator/pipelines/<model>_pipeline.py` implementing `ModelPipeline`
3. Register in `orchestrator/pipelines/__init__.py` -> `PIPELINE_REGISTRY`

To add a new model to the **insights runner** (3 steps, no core code changes):
1. Add a threshold block to `insights_runner/quality_gate/thresholds.yaml` (include `candidate_window_years`)
2. Create `insights_runner/runners/<model>_runner.py` implementing `ModelRunner`
3. Register in `insights_runner/runners/__init__.py` -> `RUNNER_REGISTRY`

---

## Model Routing Reference

| Model | Required subtypes | Candidate window | Primary use case |
|-------|------------------|-----------------|-----------------|
| **BOCPD** | date + measure | 2 years | Weekly trend-shift / changepoint detection |
| **MMM** | date + measure + channel | 1 year | Marketing mix attribution |
| **PSI** | date + measure + segment | 1 year | Population / segment drift detection |
| **ARIMA** | date + measure | 3 years | Seasonal trend forecasting (fallback) |

---

## Key Conventions

- **Config-not-code.** Candidate lists, abbreviations, routing rules, quality thresholds, transform rules, and candidate window sizes all live in YAML files. Rarely need to edit Python to handle new datasets.
- **`unclassified_metric` never dropped.** Flagged and passed to models that accept generic measures. Warnings always surfaced in `RouterResult.warnings`.
- **Token-boundary matching, not greedy substring.** `column_matcher.py` uses token-set subset check. Two pre-matching guards protect against false positives: date-suffix override (last token `_date` etc.) and rolling-metric guard (`_last_90d` patterns).
- **MMM requires channel.** Adding `channel` to MMM's `required` list ensures pure sales/adherence tables don't route to MMM. Non-numeric channel columns (VARCHAR) are additionally filtered in `_build_dataset_config`.
- **`select_date_column` is the single date picker.** Used by all runners, quality checks, and the pipeline window slicer — eliminates `[0]`/`next()` patterns. Scores date candidates by match_source, confidence, null_pct, date_grain, and unique_count.
- **Segment-aware aggregation.** `normalize_to_series` detects extra grain dimensions (`segment`, `key`, `geography` subtypes) via `column_specs` and performs a 2-step rollup: within-segment collapse then national aggregation. A `n_{segment}` unique-count column is added for PSI analysis.
- **Distribution quality scoring.** `_score_spec` in `_measure_selector.py` penalises constant (−3.0) and binary (−3.0) columns and near-zero CV (−1.0). Healthy right-skewed columns (sk 0.3–15, typical for engagement/sales counts) get +0.4; extreme skew (sk>15, claims) +0.2; near-normal (|sk|≤1.5, prices/rates) +0.2.
- **`pipeline/` functions accept explicit `dataset_config`.** `transform_channels()`, `build_model_matrix()`, and `build_pymc_model()` all accept `dataset_config=None` (falls back to global `DATASET`). The runners always pass an explicit config built from `RouterResult` — no global mutation.
- **BOCPD context window keys follow the measure column.** `cp_context_windows` rows use `log_{measure_col}` and `{measure_col}` as keys — generic over any continuous metric, not just TRX/sales.
- **String detail messages use ASCII only.** The quality gate check detail strings avoid Unicode so they serialize cleanly on Windows cp1252 and Databricks driver output.
- **Signal deduplication.** `Router.run(deduplicate=True)` groups fact tables by (routing_signature, top_model) and marks less-rich duplicates `is_duplicate_signal=True`. Skip these in the insights loop.
- **`pipeline/config.py` local fallback.** Locally, if `dataset_config.json` and the gold parquet are both absent, `_load_dataset()` returns an empty `DatasetConfig(target_col="trx")` placeholder rather than crashing. This is safe because `insights_runner` never uses the global `DATASET`.
- **Profiler grain is reliable post-fix.** `_infer_date_grain` deduplicates dates before computing gaps — prevents HCP-level repeated-date data from returning "daily" when the underlying grain is monthly.
