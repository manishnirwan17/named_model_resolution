# Named Model Resolution

A **data-aware model router + insights pipeline** for pharma analytics gold-layer datasets.

Given a catalog (Databricks Unity Catalog, a SQL database, or a folder of files), it:

1. Discovers all tables/datasets
2. Classifies each column by semantic subtype (`date`, `measure`, `geography`, `segment`, `channel`, …)
3. Handles abbreviated column names (`WK_END`, `TRX`, `NBRX`) via an expansion dictionary
4. Routes each fact table to the right ML model(s) — BOCPD, MMM, PSI, ARIMA, …
5. Resolves star-schema joins (fact → dimension, 1-hop only) to inherit column subtypes
6. Profiles sample data (skewness, nulls, date grain) and recommends statistical transforms
7. Runs a **quality gate** per model (fill rate, variance, collinearity, date continuity, …)
8. Executes model pipelines (BOCPD changepoint detection, MMM attribution) and emits a structured JSON payload ready for LLM summarisation

---

## Installation

### On Databricks (primary target)

Everything lives in one repo — `pipeline/` (BOCPD + MMM implementations) is a first-class package alongside `named_model_resolution/`, `orchestrator/`, and `insights_runner/`. A single `%pip` cell installs it all.

**Option A — `%pip` (recommended, auto-restarts kernel)**

```python
# Cell 1 — must be the very first cell; Databricks restarts the kernel after %pip
repo = "/Workspace/Users/your-user/named_model_resolution"
%pip install -e {repo}[pipeline] --quiet
```

The `[pipeline]` extra pulls in the heavier model deps (PyMC, bayesian-changepoint-detection, arviz). Omit it if you only need the router and quality gate without running BOCPD/MMM.

**Option B — `subprocess` (no kernel restart, for jobs or shared setup cells)**

```python
import subprocess, sys, importlib

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
_repo = "/Workspace" + "/".join(_nb_path.split("/")[:-1])

subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{_repo}[pipeline]", "--quiet"])
if _repo not in sys.path:
    sys.path.insert(0, _repo)
importlib.invalidate_caches()
print("Install complete — imports ready")
```

> **Python version:** Requires >= 3.10. DBR 14.x = Python 3.10, DBR 15.x = Python 3.12.
> **PyMC sampling:** Use a single-node cluster (GPU or high-memory). Multi-node Spark clusters will not distribute NUTS sampling.

### Local development

```bash
pip install -e ".[pipeline]"   # full install including BOCPD/MMM deps
pip install -e .               # router + quality gate only (no PyMC)
```

---

## Running on Databricks — Full Notebook Pattern

### Cell 1 — Install (see above)

### Cell 2 — Config paths

```python
from pathlib import Path

REPO_ROOT   = Path("/Workspace/Users/your-user/named_model_resolution")
SPEC_PATH   = REPO_ROOT / "pharma_knowledge_base/gold_layer_datamarts.csv"
CONFIGS_DIR = REPO_ROOT / "pharma_knowledge_base/configs"

# Unity Catalog location — match your workspace
UC_CATALOG = "nexora_poc_catalog"
UC_SCHEMA  = "gold"
```

### Cell 3 — Connector

**Option A — Databricks Unity Catalog via SQL connector**

```python
from sqlalchemy import create_engine
from orchestrator.connectors import SQLCatalogConnector

# Install: pip install databricks-sql-connector sqlalchemy-databricks
engine = create_engine(
    "databricks+connector://token@<workspace-host>/<http-path>",
    connect_args={"catalog": UC_CATALOG, "schema": UC_SCHEMA},
)
connector = SQLCatalogConnector(engine, schema=UC_SCHEMA)
```

**Option B — Parquet files in a UC Volume or DBFS path**

```python
from orchestrator.connectors import FileCatalogConnector

connector = FileCatalogConnector("/Volumes/nexora_poc_catalog/gold/raw_data/")
```

### Cell 4 — Run the router

```python
from named_model_resolution.catalog_parser import parse_catalog
from orchestrator.router import Router

catalog = parse_catalog(SPEC_PATH)
router  = Router(connector, SPEC_PATH, CONFIGS_DIR)

# Route all tables; pass datasets=[...] to limit scope
results = router.run(deduplicate=True)

for r in results:
    if r.is_duplicate_signal:
        print(f"  [dup of {r.signal_group_primary}] {r.dataset_name}")
        continue
    print(f"\n{r.dataset_name}  ({r.classification.table_type})")
    for mc in r.model_configs[:3]:
        print(f"  [{mc.confidence:.3f}] {mc.model_name}")
    if r.warnings:
        for w in r.warnings:
            print(f"  WARN: {w}")
```

### Cell 5 — Run insights pipeline on a fact table

```python
from insights_runner.pipeline import run as run_insights

for r in results:
    if r.classification.table_type != "fact" or r.is_duplicate_signal:
        continue

    payload = run_insights(
        connector=connector,
        router_result=r,
        catalog=catalog,
        configs_dir=CONFIGS_DIR,
        # models_to_run=["BOCPD"],  # restrict to skip MMM/PyMC when not needed
    )

    print(payload.to_json())   # <- feed this string directly to an LLM
    break
```

**What `payload.to_json()` contains:**

```json
{
  "dataset_name": "engagement_mmm_labelled",
  "metadata": {
    "table_type": "fact",
    "routing": {"top_model": "MMM", "confidence": 0.717},
    "star_schema": {"dimension_tables": ["gold_hcp_details"], "join_keys": {"hcp_id": "HCP_ID"}}
  },
  "quality_gate": {
    "overall_decision": "WARN",
    "per_model": {
      "MMM":   {"decision": "WARN", "checks": {"channel_collinearity": {"status": "WARN", "metric": 0.87}}},
      "BOCPD": {"decision": "PASS", "checks": {"fill_rate": {"status": "PASS", "metric": 1.0}}}
    }
  },
  "model_signals": {
    "BOCPD": {"ran": true, "signals": {"n_changepoints": 3, "cp_candidates": [...]}},
    "MMM":   {"ran": true, "signals": {"model_fit": {"in_sample_mape": 0.062, "rhat_max": 1.003}}}
  },
  "warnings": ["..."],
  "knowledge_base_context": {"use_cases": ["Decompose TRx drivers across channels"]}
}
```

---

## Running Locally (CLI)

```bash
python main.py --catalog-type file --catalog-path ./sample_data/ --output-dir ./output/

# Specific tables only
python main.py --catalog-type file --catalog-path ./sample_data/ --datasets Gold_Rx_Claims

# With pipeline execution
python main.py --catalog-type file --catalog-path ./sample_data/ --run-pipelines

# Deduplicate tables with identical routing signals
python main.py --catalog-type file --catalog-path ./sample_data/ --deduplicate
```

---

## Project Structure

```
named_model_resolution/     core classification + routing package
orchestrator/               connector + profiler + router + pipeline stubs
insights_runner/            quality gate + model bridge → InsightsPayload JSON
pipeline/                   BOCPD + MMM pipeline implementations (installed via [pipeline] extra)
pharma_knowledge_base/
  gold_layer_datamarts.csv  10-datamart spec (multi-section flat file)
  configs/
    candidates.yaml         column name candidate lists (add new names here)
    abbreviations.yaml      abbreviation map (add new abbrevs here)
    model_routing.yaml      per-model routing rules (add new models here)
    transform_rules.yaml    statistical transform thresholds
pyproject.toml              single install config
main.py                     CLI entry point
```

---

## How to Extend

### Fix an unrecognised column

If a column shows as `unclassified_metric` or `unknown` in warnings, add it to
`pharma_knowledge_base/configs/candidates.yaml`:

```yaml
measure_candidates:
  - my_new_kpi
channel_candidates:
  - my_new_channel
```

Or if it's an abbreviation, add to `abbreviations.yaml`:

```yaml
mnk: my_new_kpi
```

### Add a new routed model (3 steps)

1. **`model_routing.yaml`** — add a routing rule block:
```yaml
CausalImpact:
  required: [date, measure]
  optional: [geography]
  accepts: [unclassified_metric]
  preferred_measures: [trx]
  description: "Causal inference on intervention events"
  use_case_hints:
    - "Measure impact of a launch or label change on TRx"
```

2. **`orchestrator/pipelines/causal_impact_pipeline.py`** — implement `ModelPipeline`

3. **`orchestrator/pipelines/__init__.py`** — register:
```python
from .causal_impact_pipeline import CausalImpactPipeline
PIPELINE_REGISTRY["CausalImpact"] = CausalImpactPipeline
```

### Add a new model to the insights runner (3 steps)

1. **`insights_runner/quality_gate/thresholds.yaml`** — add threshold block
2. **`insights_runner/runners/causal_impact_runner.py`** — implement `ModelRunner`
3. **`insights_runner/runners/__init__.py`** — add to `RUNNER_REGISTRY`

No other files change in either case.

---

## Key Concepts

**Quality gate** — before any expensive model run, `insights_runner` checks fill rate, zero variance, channel collinearity, date continuity, segment balance, and autocorrelation. FAIL → model skipped with reason; WARN → model runs with caveat flagged in output; PASS → model runs normally.

**Guardrail gate** — any numeric column that cannot be matched to a known subtype becomes `unclassified_metric`, is never dropped, and is passed to any model that accepts generic measures. Warnings are always surfaced.

**Star schema (1-hop)** — if a fact table has a foreign key to a dimension table in the catalog, the dimension's column subtypes are inherited for routing. Snowflake-style multi-hop joins are intentionally excluded (too expensive).

**Routing confidence** — models are scored and returned as a ranked list. A table scores multiple candidates; inspect `result.model_configs` to see all options and their confidence scores.

**Configs, not code** — candidate lists, abbreviation maps, routing rules, quality thresholds, and transform rules all live in YAML files. Rarely need to edit Python to handle new datasets or models.
