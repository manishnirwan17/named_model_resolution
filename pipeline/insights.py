"""
07_insights.py — Convert BOCPD + MMM signals into natural language insights.

A LangGraph ReAct agent backed by ChatDatabricks reads the pipeline artefacts
(integration report, validation report, MMM contributions, dataset metadata) and
produces a structured JSON insight report.

Deployment path (Databricks):
  1. Run `log_insights_agent()` to register the agent in MLflow.
  2. Deploy the logged model to a Databricks Model Serving endpoint.
  3. The endpoint accepts {"messages": [{"role": "user", "content": "..."}]}.

Local development:
  Set DATABRICKS_HOST + DATABRICKS_TOKEN in your environment and run:
    python -c "from pipeline.insights import generate_insights; generate_insights()"
  or override INSIGHTS_MODEL_ENDPOINT to point at a compatible serving endpoint.

Required packages:
  uv add databricks-langchain langgraph mlflow

Reads:  integration_report.csv  (MODEL_OUT / "integration_report.csv")
        VALIDATION_RPT           (UC table on Databricks, CSV locally)
        CONTRIBUTIONS            (UC table on Databricks, parquet locally)
        mmm_meta.json            (MODEL_OUT / "mmm_meta.json")
Writes: insights_report.json    (MODEL_OUT / "insights_report.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from pipeline.config import (
    CONTRIBUTIONS,
    DATASET,
    MODEL_OUT,
    ON_DATABRICKS,
    PARAMS,
    VALIDATION_RPT,
    read_csv,
    read_parquet,
)

try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except Exception:
    pass

# ── Resolved artefact paths ───────────────────────────────────────────────────
_INTEG_REPORT = Path(MODEL_OUT) / "integration_report.csv"
_MMM_META     = Path(MODEL_OUT) / "mmm_meta.json"
_INSIGHTS_OUT = Path(MODEL_OUT) / "insights_report.json"

# ── DataFrame cache (avoids re-reading large tables on every tool call) ───────
_DF_CACHE: dict[str, pd.DataFrame] = {}


def _load_contributions() -> pd.DataFrame:
    if "contributions" not in _DF_CACHE:
        df = read_parquet(CONTRIBUTIONS)
        df["week"] = pd.to_datetime(df["week"])
        _DF_CACHE["contributions"] = df
    return _DF_CACHE["contributions"]


def _load_validation() -> pd.DataFrame:
    if "validation" not in _DF_CACHE:
        _DF_CACHE["validation"] = read_csv(VALIDATION_RPT)
    return _DF_CACHE["validation"]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_changepoint_report() -> str:
    """Return all detected changepoints from the integration report.

    Each record includes the changepoint week, posterior probability, root-cause
    classification, relative contribution shifts (field channels and total), and
    the post-changepoint MMM residual z-score.

    Call this first to understand what structural breaks the model detected.
    """
    if not _INTEG_REPORT.exists():
        return "integration_report.csv not found — run the integration step first."
    df = pd.read_csv(str(_INTEG_REPORT))
    if df.empty:
        return "No changepoints detected in the latest pipeline run."

    lines = [f"Detected {len(df)} changepoint(s):\n"]
    for _, row in df.iterrows():
        lines.append(
            f"  • {row['cp_week']}  prob={row['cp_prob']:.3f}"
            f"  class={row['classification']}\n"
            f"    total_contrib_shift={row.get('total_contrib_shift_rel', 'n/a')}"
            f"  field_contrib_shift={row.get('field_contrib_shift_rel', 'n/a')}"
            f"  residual_z_post={row.get('residual_z_post', 'n/a')}\n"
        )
    return "".join(lines)


@tool
def get_channel_contribution_shifts(cp_week: str) -> str:
    """Return the pre/post channel contribution breakdown around a single changepoint.

    For the given changepoint week, computes the mean per-channel contribution in
    the PRE_WINDOW weeks before vs POST_WINDOW weeks after the break so you can
    identify which channels drove the structural shift.

    Args:
        cp_week: Changepoint date as YYYY-MM-DD (use dates from get_changepoint_report).
    """
    try:
        cp_ts = pd.Timestamp(cp_week)
    except Exception:
        return f"Invalid date format: {cp_week!r}. Use YYYY-MM-DD."

    pre_w = pd.Timedelta(weeks=PARAMS["PRE_WINDOW"])
    pst_w = pd.Timedelta(weeks=PARAMS["POST_WINDOW"])

    contrib = _load_contributions()
    ch_cols = [c for c in contrib.columns if c.startswith("contrib_")]

    pre_mask = (contrib["week"] >= cp_ts - pre_w) & (contrib["week"] < cp_ts)
    pst_mask = (contrib["week"] >= cp_ts) & (contrib["week"] < cp_ts + pst_w)

    if not pre_mask.any():
        return f"No contribution data found before {cp_week}."

    rows = [
        f"Channel contributions around {cp_week} "
        f"(pre={PARAMS['PRE_WINDOW']}w / post={PARAMS['POST_WINDOW']}w):\n",
        f"  {'Channel':<38} {'Pre':>10} {'Post':>10} {'Δ rel':>8}\n",
        "  " + "-" * 70 + "\n",
    ]
    for col in ch_cols:
        pre_v = float(contrib.loc[pre_mask, col].mean())
        pst_v = float(contrib.loc[pst_mask, col].mean()) if pst_mask.any() else float("nan")
        rel   = (pst_v - pre_v) / (abs(pre_v) + 1e-9) if not np.isnan(pst_v) else float("nan")
        rows.append(
            f"  {col:<38} {pre_v:>10.4f} {pst_v:>10.4f} {rel:>8.3f}\n"
        )
    return "".join(rows)


@tool
def get_validation_results() -> str:
    """Return the model validation results (PASS / FAIL / SKIP per check).

    Covers BOCPD changepoint detection accuracy, MMM coefficient recovery, and
    in-sample + hold-out MAPE thresholds. Use this to gauge model quality before
    drawing business conclusions.
    """
    try:
        df = _load_validation()
    except Exception as exc:
        return f"Could not read validation report: {exc}"

    passed  = int((df["status"] == "PASS").sum())
    failed  = int((df["status"] == "FAIL").sum())
    skipped = int((df["status"] == "SKIP").sum())
    total   = passed + failed

    icon_map = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}
    lines = [f"Validation: {passed}/{total} PASS  |  {failed} FAIL  |  {skipped} SKIP\n\n"]
    for _, row in df.iterrows():
        icon = icon_map.get(row["status"], row["status"])
        lines.append(f"  {icon}  {row['check']:<45}  {row.get('detail', '')}\n")
    return "".join(lines)


@tool
def get_dataset_metadata() -> str:
    """Return metadata about this dataset: target variable, channels, and known events.

    Use this first to understand what the channel names mean (field vs broadcast),
    what the primary target metric is, and whether any organic changepoints are
    pre-configured (and therefore expected in the results).
    """
    lines = [
        f"Target column:      {DATASET.target_col}\n",
        f"Log-target column:  {DATASET.target_log_col}\n",
        f"Week column:        {DATASET.week_col}\n",
        f"Lifecycle column:   {DATASET.lifecycle_col or 'not configured'}\n",
        f"\nField channels ({len(DATASET.field_channels)}):\n",
    ]
    for ch in DATASET.field_channels:
        lines.append(
            f"  • {ch:<32}  decay={DATASET.decay.get(ch, 0.30):.2f}"
            f"  hill_alpha={DATASET.hill_alpha(ch):.2f}\n"
        )
    lines.append(f"\nBroadcast channels ({len(DATASET.broadcast_channels)}):\n")
    for ch in DATASET.broadcast_channels:
        lines.append(
            f"  • {ch:<32}  decay={DATASET.decay.get(ch, 0.60):.2f}"
            f"  hill_alpha={DATASET.hill_alpha(ch):.2f}\n"
        )
    if DATASET.competitor_channels:
        lines.append(f"\nCompetitor channels ({len(DATASET.competitor_channels)}):\n")
        for ch in DATASET.competitor_channels:
            lines.append(f"  • {ch}\n")
    known_cps = DATASET.organic_cp_timestamps
    if known_cps:
        lines.append("\nKnown organic changepoints (from dataset_config.json):\n")
        for name, ts in known_cps.items():
            lines.append(f"  • {name}: {ts.date()}\n")
    else:
        lines.append("\nKnown organic changepoints: none configured.\n")
    return "".join(lines)


@tool
def get_recent_channel_trends(n_weeks: int = 12) -> str:
    """Return the mean weekly contribution of each channel over the most recent N weeks.

    Use this to identify which channels are currently the largest growth drivers and
    how much of the target metric is baseline vs channel-attributed.

    Args:
        n_weeks: Number of most recent weeks to average (default 12).
    """
    contrib = _load_contributions()
    recent  = contrib.sort_values("week").tail(n_weeks)
    ch_cols = [c for c in contrib.columns if c.startswith("contrib_")]

    means      = {col: float(recent[col].mean()) for col in ch_cols}
    baseline   = float(recent["baseline"].mean()) if "baseline" in recent.columns else 0.0
    total_attr = sum(means.values())

    start_w = recent["week"].min().date()
    end_w   = recent["week"].max().date()

    lines = [
        f"Channel trends  [{start_w} → {end_w}]  (most recent {len(recent)} weeks):\n",
        f"  Baseline (intercept):   {baseline:>10.4f}\n\n",
        f"  {'Channel':<38} {'Mean contrib':>14} {'Share':>8}\n",
        "  " + "-" * 64 + "\n",
    ]
    for col, val in sorted(means.items(), key=lambda x: -abs(x[1])):
        share = val / (abs(total_attr) + 1e-9) * 100
        lines.append(f"  {col:<38} {val:>14.4f} {share:>7.1f}%\n")
    lines.append(f"\n  Total attributed (excl. baseline): {total_attr:.4f}\n")
    return "".join(lines)


@tool
def get_model_metadata() -> str:
    """Return MMM model metadata: feature columns and channel adstock/saturation parameters.

    Use this to understand how the model was built (adstock decay, Hill curve shape,
    half-saturation points) so you can correctly interpret which channels have more
    carry-over effect vs which respond instantly to spend.
    """
    if not _MMM_META.exists():
        return "mmm_meta.json not found — run MMM data prep and fit first."
    with open(str(_MMM_META)) as f:
        meta = json.load(f)

    scaler   = meta.get("scaler", {})
    ch_meta  = meta.get("channel_meta", {})

    lines = ["MMM model metadata:\n\n"]
    lines.append(f"  Features: {scaler.get('feature_cols', [])}\n\n")
    lines.append("  Channel transforms (adstock + Hill saturation):\n")
    lines.append(f"  {'Channel':<30}  {'Type':>4}  {'Decay':>6}  {'HillAlpha':>10}  {'HillK':>10}\n")
    lines.append("  " + "-" * 68 + "\n")
    for ch, info in ch_meta.items():
        lines.append(
            f"  {ch:<30}  {info['type']:>4}  {info['decay']:>6.2f}"
            f"  {info['hill_alpha']:>10.2f}  {info['hill_K']:>10.4f}\n"
        )
    return "".join(lines)


# ── Tools list (exported for logging / serving) ───────────────────────────────

INSIGHTS_TOOLS = [
    get_changepoint_report,
    get_channel_contribution_shifts,
    get_validation_results,
    get_dataset_metadata,
    get_recent_channel_trends,
    get_model_metadata,
]


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert marketing analytics interpreter for a pharmaceutical / life-sciences company.
You have access to the outputs of a BOCPD (Bayesian Online Changepoint Detection) + Bayesian
Marketing Mix Modeling (MMM) pipeline and must convert the quantitative signals into clear,
actionable insights for a commercial leadership audience.

Workflow — follow this order:
1. Call get_dataset_metadata to understand the channels, target metric, and any known events.
2. Call get_validation_results to assess model quality. If checks FAIL, note this as a caveat.
3. Call get_changepoint_report to see all detected structural breaks.
4. For every changepoint with prob > 0.50, call get_channel_contribution_shifts to identify
   the channels that drove the break.
5. Call get_recent_channel_trends to capture the current channel performance picture.
6. Call get_model_metadata if you need to interpret coefficient magnitudes or carry-over.

Reasoning rules:
- Distinguish confirmed business events (known organic CPs from metadata, high-prob detections
  with clear channel shifts) from data artifacts (high residual_z, classification=artifact_*).
- Quote concrete numbers: dates, percentages, channel names — avoid vague language.
- If the validation report has FAIL entries, flag that conclusions should be treated with caution.

Return your final response as a single JSON object with EXACTLY these keys — no markdown fences,
no preamble, only the JSON:
{
  "executive_summary": "<2–3 sentence high-level takeaway for senior leadership>",
  "changepoints": [
    {
      "week": "YYYY-MM-DD",
      "classification": "<from integration report>",
      "channels_driving": ["<channel>", ...],
      "field_contrib_shift_pct": <field_contrib_shift_rel * 100>,
      "narrative": "<1–2 sentence plain-English explanation of what happened and why>"
    }
  ],
  "top_channel_drivers": [
    {"channel": "<name>", "avg_weekly_contribution": <float>, "share_pct": <float>}
  ],
  "model_quality": {
    "in_sample_mape": "<e.g. 6.2% or N/A>",
    "holdout_mape": "<e.g. 9.1% or N/A>",
    "overall": "<PASS | FAIL | MIXED | UNKNOWN>",
    "caveats": "<list any FAIL checks, or empty string>"
  },
  "recommendations": [
    "<concrete, specific action or investigation item>"
  ]
}
"""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_insights_agent() -> Any:
    """Build a LangGraph ReAct agent using ChatDatabricks.

    Requires:
      - DATABRICKS_HOST + DATABRICKS_TOKEN (or running on a Databricks cluster)
      - INSIGHTS_MODEL_ENDPOINT env var (default: databricks-claude-sonnet-4-5)

    Install: uv add databricks-langchain langgraph
    """
    try:
        from databricks_langchain import ChatDatabricks
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise RuntimeError(
            "Install required packages: uv add databricks-langchain langgraph"
        ) from exc

    endpoint = os.getenv("INSIGHTS_MODEL_ENDPOINT", "databricks-claude-sonnet-4-5")
    llm = ChatDatabricks(endpoint=endpoint)
    return create_react_agent(llm, tools=INSIGHTS_TOOLS, prompt=_SYSTEM_PROMPT)


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_user_prompt(extra_context: str | None = None) -> str:
    lines = [
        f"Generate a complete insights report for the latest pipeline run.\n"
        f"Target metric: {DATASET.target_col}  |  "
        f"Channels: {len(DATASET.field_channels)} field + "
        f"{len(DATASET.broadcast_channels)} broadcast"
        + (f" + {len(DATASET.competitor_channels)} competitor"
           if DATASET.competitor_channels else "")
        + "."
    ]
    if extra_context:
        lines.append(f"\nAdditional context from the analyst:\n{extra_context}")
    return "\n".join(lines)


# ── Output helpers ────────────────────────────────────────────────────────────

def _extract_agent_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else last.get("content", "")
        else:
            content = result.get("output", "")
    else:
        content = result

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(p for p in parts if p).strip()

    return str(content).strip()


def _parse_insights_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "executive_summary": raw_text,
            "changepoints": [],
            "top_channel_drivers": [],
            "model_quality": {"overall": "UNKNOWN"},
            "recommendations": [],
            "raw_response": raw_text,
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_insights(extra_context: str | None = None) -> dict[str, Any]:
    """Run the insights agent end-to-end and write insights_report.json.

    Args:
        extra_context: Optional free-text analyst context appended to the default
                       prompt (e.g. "Focus on the Q3 2023 launch window.").

    Returns:
        Parsed insights dict with keys: executive_summary, changepoints,
        top_channel_drivers, model_quality, recommendations.
    """
    print("=" * 60)
    print("  07  INSIGHTS GENERATION")
    print("=" * 60)

    agent = build_insights_agent()
    prompt = _build_user_prompt(extra_context)
    endpoint = os.getenv("INSIGHTS_MODEL_ENDPOINT", "databricks-claude-sonnet-4-5")

    print(f"  Model endpoint: {endpoint}")
    print("  Invoking agent ...\n")

    result  = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    raw     = _extract_agent_text(result)
    insights = _parse_insights_json(raw)

    _INSIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(_INSIGHTS_OUT), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, default=str)

    print(f"  Insights report written -> {_INSIGHTS_OUT}")
    if "executive_summary" in insights:
        print(f"\n  Summary: {insights['executive_summary']}")
    if insights.get("changepoints"):
        print(f"  Changepoints analysed: {len(insights['changepoints'])}")
    mq = insights.get("model_quality", {})
    print(f"  Model quality: {mq.get('overall', 'UNKNOWN')}")
    print("=" * 60)

    return insights


# ── MLflow deployment ─────────────────────────────────────────────────────────

def log_insights_agent(
    experiment_name: str = "/Shared/insights_generation",
    run_name: str = "insights_agent",
) -> str:
    """Log the insights agent to MLflow for deployment on Model Serving.

    Registers a LangGraph agent with the six retrieval tools. After logging,
    deploy the returned run's model artifact to a Databricks Model Serving endpoint.

    Args:
        experiment_name: MLflow experiment path (default /Shared/insights_generation).
        run_name: Display name for the MLflow run.

    Returns:
        MLflow run ID of the logged model.

    Example (Databricks notebook)::

        from pipeline.insights import log_insights_agent
        run_id = log_insights_agent()
        # Then create a Model Serving endpoint pointing at runs:/<run_id>/insights_agent
    """
    try:
        import mlflow
        import mlflow.langchain
    except ImportError as exc:
        raise RuntimeError("Install mlflow: uv add mlflow") from exc

    agent = build_insights_agent()
    input_example = {
        "messages": [
            {"role": "user", "content": _build_user_prompt()}
        ]
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.langchain.log_model(
            lc_model=agent,
            artifact_path="insights_agent",
            input_example=input_example,
        )
        run_id = run.info.run_id
        print(f"  Logged insights agent  run_id={run_id}")
        print(f"  Artifact: runs:/{run_id}/insights_agent")

    return run_id


if __name__ == "__main__":
    generate_insights()
