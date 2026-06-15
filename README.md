# Named Model Resolution

A **data-aware model router** for pharma analytics gold-layer datasets. Give it a catalog (folder of files, a database, or a Databricks schema), and it:

1. Discovers all tables/datasets
2. Classifies each column by semantic subtype (`date`, `measure`, `geography`, `segment`, …)
3. Handles abbreviated column names (`WK_END`, `TRX`, `NBRX`) via an expansion dictionary
4. Routes each fact table to the right ML model(s) — BOCPD, MMM, PSI, ARIMA, …
5. Profiles sample data (skewness, nulls, date grain) and recommends statistical transforms
6. Runs per-model pipelines: detect use-cases → prep → transform → tune

---

## Installation

### Option A — Local development (uv)

```bash
git clone <repo>
cd named_model_resolution
uv sync          # installs both packages + all deps into .venv
```

### Option B — Databricks notebook (recommended pattern)

The standard `%pip` magic restarts the kernel automatically and is the simplest approach.
Both packages must be in the **same `%pip` cell** (each `%pip` restarts the kernel, so splitting them loses the first install):

```python
# Cell 1 — must be the very first cell; Databricks restarts the kernel after this
repo = "/Workspace/Users/your-user/named_model_resolution"
%pip install -e {repo}/named_model_resolution -e {repo}/orchestrator databricks-sqlalchemy
```

All subsequent cells import normally after the restart.

---

### Option C — Databricks notebook (no kernel restart, `subprocess` pattern)

If you need to install without restarting (e.g. inside a job or a shared setup cell),
use `subprocess` **and** manually add the `src/` directories to `sys.path` afterward.
`subprocess pip install` writes the `.pth` files but Python's `sys.path` is already
frozen at kernel startup — it won't re-read them mid-session.

```python
import subprocess, sys, importlib

# Resolve repo root from the notebook's own path (works for any user/clone location)
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_nb_path = _ctx.notebookPath().get()
_repo_root = "/Workspace" + "/".join(_nb_path.split("/")[:-1])

print(f"Repo root: {_repo_root}")

# 1. Install deps that aren't in sys.path yet (sqlalchemy, scipy, etc.)
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "databricks-sqlalchemy", "scipy", "--quiet"])

# 2. Add the project directories directly — bypasses the .pth file mechanism entirely
for pkg in ("named_model_resolution", "orchestrator"):
    pkg_path = f"{_repo_root}/{pkg}"
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

# 3. Tell Python to re-scan for newly visible modules
importlib.invalidate_caches()

print("sys.path updated — imports are ready")
```

Then in the next cell, imports work immediately without a restart:

```python
from named_model_resolution.catalog_parser import parse_catalog
from orchestrator.connectors import FileCatalogConnector, SQLCatalogConnector
from orchestrator.router import Router
```

> **Python version:** Requires ≥ 3.10. Databricks Runtime 14.x = Python 3.10, 15.x = Python 3.12.

---

## Running via CLI (local)

```bash
# Route datasets from a local folder of CSV/Parquet files
uv run python main.py \
    --catalog-type file \
    --catalog-path ./sample_data/ \
    --output-dir ./output/

# Process specific tables only
uv run python main.py \
    --catalog-type file \
    --catalog-path ./sample_data/ \
    --datasets Gold_Rx_Claims Gold_Patient_Adherence

# Route AND run the top-ranked pipeline for each dataset
uv run python main.py \
    --catalog-type file \
    --catalog-path ./sample_data/ \
    --run-pipelines

# Write per-dataset routing configs as JSON to ./output/
uv run python main.py \
    --catalog-type file \
    --catalog-path ./sample_data/ \
    --output-dir ./output/
```

---

## Running in a Databricks Notebook

After installation (see above), create a `main.ipynb` with the following pattern:

### Cell 1 — Install (first time only)
```python
repo = "/Workspace/Users/your-user/named_model_resolution"
%pip install -e {repo}/named_model_resolution
%pip install -e {repo}/orchestrator
```

### Cell 2 — Imports and paths
```python
from pathlib import Path

REPO_ROOT = Path("/Workspace/Users/your-user/named_model_resolution")
SPEC_PATH   = REPO_ROOT / "pharma_knowledge_base/gold_layer_datamarts.csv"
CONFIGS_DIR = REPO_ROOT / "pharma_knowledge_base/configs"
```

### Cell 3 — Choose a connector

**Option A: Local files (CSV/Parquet in a folder)**
```python
from orchestrator.connectors import FileCatalogConnector

connector = FileCatalogConnector(REPO_ROOT / "sample_data")
```

**Option B: Databricks Unity Catalog via SQL**
```python
from sqlalchemy import create_engine
from orchestrator.connectors import SQLCatalogConnector

# Uses the Databricks SQL connector — install separately:
#   pip install databricks-sql-connector sqlalchemy-databricks
engine = create_engine(
    "databricks+connector://token@<host>/<http_path>",
    connect_args={"catalog": "hive_metastore", "schema": "gold"},
)
connector = SQLCatalogConnector(engine, schema="gold")
```

**Option C: Any SQLAlchemy-compatible database**
```python
from sqlalchemy import create_engine
from orchestrator.connectors import SQLCatalogConnector

engine = create_engine("postgresql://user:pass@host/db")
connector = SQLCatalogConnector(engine, schema="public")
```

### Cell 4 — Run the router
```python
from orchestrator.router import Router

router = Router(connector, SPEC_PATH, CONFIGS_DIR)
results = router.run()   # pass datasets=["TableName", ...] to limit scope
```

### Cell 5 — Inspect results
```python
for r in results:
    print(f"\n{'─'*60}")
    print(f"Dataset    : {r.dataset_name}")
    print(f"Table type : {r.classification.table_type}")

    if r.classification.matched_catalog_entry:
        print(f"Catalog match: {r.classification.matched_catalog_entry} "
              f"(score={r.classification.catalog_match_score:.2f})")

    print("Column subtypes:")
    for col in r.classification.columns:
        exp = f" → {col.expanded_name}" if col.expanded_name else ""
        print(f"  {col.name:25s} {col.semantic_subtype:20s} conf={col.confidence:.1f}{exp}")

    if r.model_configs:
        print("\nModel routing (ranked):")
        for mc in r.model_configs:
            print(f"  [{mc.confidence:.3f}] {mc.model_name}")
            for uc in mc.use_cases[:2]:
                print(f"         • {uc}")

    if r.column_profiles:
        flagged = [p for p in r.column_profiles if p.suggested_transforms]
        if flagged:
            print("\nTransform suggestions:")
            for p in flagged:
                print(f"  {p.name}: {'; '.join(p.suggested_transforms)}")

    if r.warnings:
        print("\nWarnings:")
        for w in r.warnings:
            print(f"  ⚠ {w}")
```

### Cell 6 — Run a specific model pipeline
```python
from orchestrator.pipelines import PIPELINE_REGISTRY

for r in results:
    if not r.model_configs:
        continue
    top_model = r.model_configs[0].model_name          # highest-confidence model
    pipeline_cls = PIPELINE_REGISTRY.get(top_model)
    if pipeline_cls:
        output = pipeline_cls().run(connector, r)
        print(output)
```

---

## Expected Output

```
────────────────────────────────────────────────────────────
Dataset    : Gold_Patient_Adherence
Table type : fact
Catalog match: Gold_Patient_Adherence (score=0.64)

Column subtypes:
  Patient_sk                key                  conf=1.0
  period                    date                 conf=1.0
  adherence_ratio           measure              conf=1.0
  is_adherent               flag                 conf=1.0
  MYSTERY_NUM_COL           unclassified_metric  conf=0.5

Model routing (ranked):
  [0.660] PSI
         • Patient segment distribution drift over time
  [0.583] ARIMA
         • Forecast weekly TRx/NRx over next N periods
  [0.528] BOCPD
         • Weekly persistency trend monitoring across territory

Transform suggestions:
  adherence_ratio: [ratio_bound_check] values > 1.0 found in ratio column — check for scale mismatch

Warnings:
  ⚠ Column 'MYSTERY_NUM_COL' (dtype=float64) could not be matched to a known subtype.
    Classified as 'unclassified_metric' — will be offered to models that accept generic measures.
    Expand candidates.yaml or abbreviations.yaml if this is a known measure.
```

---

## Project Structure

```
named_model_resolution/               ← config-builder library
  src/named_model_resolution/
    models.py                         all shared dataclasses
    catalog_parser.py                 parses gold_layer_datamarts.csv
    column_matcher.py                 abbreviation expand + subtype matching + guardrail
    classifier.py                     fact vs dimension classification
    config_assembler.py               ranked model config assembly

orchestrator/                         orchestration package
  src/orchestrator/
    connectors/
      base.py                         CatalogConnector Protocol (interface)
      file_connector.py               reads CSV/Parquet from a folder
      sql_connector.py                reads via SQLAlchemy (Databricks, Postgres, etc.)
    profiler.py                       samples data → skewness/grain/transform suggestions
    router.py                         main entry point — runs the full pipeline
    pipelines/
      base.py                         ModelPipeline abstract base class
      bocpd_pipeline.py               BOCPD: changepoint detection
      mmm_pipeline.py                 MMM: marketing mix
      psi_pipeline.py                 PSI: population drift
      arima_pipeline.py               ARIMA/SARIMA/Prophet: seasonal forecasting
      __init__.py                     PIPELINE_REGISTRY dict

pharma_knowledge_base/
  gold_layer_datamarts.csv            spec: 10 gold-layer datamarts (do not edit)
  configs/
    candidates.yaml                   ← ADD NEW COLUMN NAMES HERE
    abbreviations.yaml                ← ADD NEW ABBREVIATIONS HERE
    model_routing.yaml                ← ADD NEW MODELS HERE
    transform_rules.yaml              statistical transform thresholds

main.py                               CLI entry point
```

---

## How to Extend

### Add a new column name / fix an unrecognised column

If a column shows up as `unclassified_metric` or `unknown` in warnings, add it to the appropriate list in `pharma_knowledge_base/configs/candidates.yaml`:

```yaml
measure_candidates:
  - persistency
  - my_new_kpi      # ← add here
```

Or if it's an abbreviation, add to `abbreviations.yaml`:

```yaml
my_abbr: my_new_kpi
```

### Add a new model

1. Add a routing rule block to `pharma_knowledge_base/configs/model_routing.yaml`:
```yaml
CausalImpact:
  required: [date, measure]
  optional: [geography]
  accepts: [unclassified_metric]
  preferred_measures: [trx, market_share]
  description: "Causal inference on intervention events"
  use_case_hints:
    - "Measure impact of a launch or label change on TRx"
```

2. Create `orchestrator/src/orchestrator/pipelines/causal_impact_pipeline.py` implementing `ModelPipeline` (copy any existing pipeline as a template — all four stages must be implemented).

3. Register it in `orchestrator/src/orchestrator/pipelines/__init__.py`:
```python
from .causal_impact_pipeline import CausalImpactPipeline
PIPELINE_REGISTRY["CausalImpact"] = CausalImpactPipeline
```

No changes to the router, classifier, or column matcher are needed.

### Add a new connector (e.g., BigQuery, Snowflake)

Create a new file in `orchestrator/src/orchestrator/connectors/` that implements the three methods from `CatalogConnector` in `base.py`:

```python
class BigQueryConnector:
    def list_datasets(self) -> list[str]: ...
    def get_schema(self, dataset: str) -> dict[str, str]: ...
    def sample_rows(self, dataset: str, n: int = 1000) -> pd.DataFrame: ...
```

Pass it to `Router(connector=..., ...)` — nothing else changes.

---

## Key Concepts

**Guardrail gate** — any numeric column that can't be matched to a known subtype is not silently dropped. It becomes `unclassified_metric` and is passed through to any model that lists `unclassified_metric` in its `accepts` rule. Warnings are always surfaced so you know what wasn't matched.

**Routing confidence** — models are scored and returned as a ranked list, not a single hard assignment. A table with `[week_end_date, persistency, territory]` will score both BOCPD and ARIMA as candidates. Inspect `result.model_configs` to see all options.

**Configs, not code** — candidate lists, abbreviation maps, routing rules, and transform thresholds are all in the YAML files under `pharma_knowledge_base/configs/`. You should rarely need to edit Python files to handle new datasets.
