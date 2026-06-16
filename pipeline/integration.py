"""
05_integration.py — Connect BOCPD change-points to MMM attribution shifts.

For each detected CP, compares channel contribution BEFORE vs AFTER the break
to produce a root-cause classification per change-point.

Root-cause classification logic:
  HIGH_RESIDUAL + no channel shift  → artifact_trx_spike candidate
  field-channel contribution jumps  → new_channel_spike candidate
  all contributions rise uniformly  → legit_spike candidate
  lifecycle / competitor shifts     → organic / competitive event

"Field channels" for classification are read from DATASET.field_channels, so the
logic automatically adapts when the channel schema changes.

Reads:  CP_CANDIDATES  (from 02_bocpd.py)
        CONTRIBUTIONS  (from 04_mmm_fit.py)
Writes: outputs/model_outputs/integration_report.csv  (or UC Volume on Databricks)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import (
    CP_CANDIDATES, CONTRIBUTIONS, MODEL_OUT,
    read_parquet, read_csv, PARAMS, DATASET,
)
try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except Exception:
    pass

INTEG_OUT = Path(MODEL_OUT) / "integration_report.csv"


def pre_post_mean(series: pd.Series, weeks: pd.Series,
                  cp_week: pd.Timestamp,
                  pre: int, post: int) -> tuple[float, float]:
    """Mean of series in the [cp_week - pre, cp_week) and [cp_week, cp_week + post) windows."""
    pre_mask  = (weeks >= cp_week - pd.Timedelta(weeks=pre)) & (weeks < cp_week)
    post_mask = (weeks >= cp_week) & (weeks < cp_week + pd.Timedelta(weeks=post))
    pre_val   = float(series[pre_mask].mean())  if pre_mask.any()  else np.nan
    post_val  = float(series[post_mask].mean()) if post_mask.any() else np.nan
    return pre_val, post_val


def _is_field_contrib(col: str) -> bool:
    """Return True if a contrib_* column corresponds to a field channel."""
    base = col.removeprefix("contrib_").removesuffix("_sat")
    return base in DATASET.field_channels


def classify_cp(row: dict) -> str:
    """
    Rule-based root-cause classification for a single change-point.
    Thresholds come from PARAMS so they can be tuned without touching code.
    """
    residual_thresh = PARAMS["RESIDUAL_ZSCORE_THRESH"]
    contrib_thresh  = PARAMS["CONTRIB_SHIFT_THRESH"]

    if row["residual_z_post"] > residual_thresh:
        return "artifact_trx_spike_candidate"

    field_shift = row.get("field_contrib_shift_rel", 0)
    if field_shift > contrib_thresh:
        total_shift = row.get("total_contrib_shift_rel", 0)
        if field_shift / (total_shift + 1e-9) > 0.7:
            return "new_channel_spike_candidate"
        return "legit_spike_candidate"

    if row.get("lifecycle_shift", 0) > 0.5:
        return "organic_lifecycle_event"

    if row.get("competitor_shift_rel", 0) > 0.3:
        return "competitive_spend_event"

    return "unclassified"


def integration() -> None:
    print("=" * 60)
    print("  05  INTEGRATION")
    print("=" * 60)

    pre  = PARAMS["PRE_WINDOW"]
    post = PARAMS["POST_WINDOW"]

    cps      = read_csv(CP_CANDIDATES)
    contribs = read_parquet(CONTRIBUTIONS)
    contribs["week"] = pd.to_datetime(contribs["week"])
    if "week" in cps.columns:
        cps["week"] = pd.to_datetime(cps["week"])

    if len(cps) == 0:
        print("  No CP candidates found. Run 02_bocpd.py first.")
        return
    print(f"  {len(cps)} CP candidates from BOCPD")

    # Identify contribution columns from the CONTRIBUTIONS parquet
    ch_contrib_cols = [c for c in contribs.columns
                       if c.startswith("contrib_") and "_sat" in c]
    comp_col = [c for c in contribs.columns
                if c.startswith("contrib_") and
                any(cc in c for cc in DATASET.competitor_channels)]

    # Residual z-score on the full series (for artifact detection)
    res_mean = contribs["residual"].mean()
    res_std  = contribs["residual"].std()
    contribs["residual_z"] = (contribs["residual"] - res_mean) / (res_std + 1e-9)

    records = []
    for _, cp in cps.iterrows():
        cp_week = pd.Timestamp(cp["week"])

        log_col = DATASET.target_log_col
        rec = {
            "cp_week":       cp_week.date(),
            "cp_prob":       round(float(cp["cp_prob"]), 4),
            log_col:         round(float(cp.get(log_col, cp.get("log_sales", 0))), 4),
        }

        total_pre, total_post = 0.0, 0.0
        field_pre, field_post = 0.0, 0.0

        for col in ch_contrib_cols:
            pre_v, post_v = pre_post_mean(contribs[col], contribs["week"],
                                           cp_week, pre, post)
            rec[f"{col}_pre"]  = round(pre_v,  4) if not np.isnan(pre_v)  else None
            rec[f"{col}_post"] = round(post_v, 4) if not np.isnan(post_v) else None
            total_pre  += pre_v  if not np.isnan(pre_v)  else 0
            total_post += post_v if not np.isnan(post_v) else 0
            if _is_field_contrib(col):
                field_pre  += pre_v  if not np.isnan(pre_v)  else 0
                field_post += post_v if not np.isnan(post_v) else 0

        rec["total_contrib_shift_rel"] = round(
            (total_post - total_pre) / (abs(total_pre) + 1e-9), 4)
        rec["field_contrib_shift_rel"] = round(
            (field_post - field_pre) / (abs(field_pre) + 1e-9), 4)

        # Lifecycle shift (optional)
        if "contrib_lc_num" in contribs.columns:
            lc_pre, lc_post = pre_post_mean(contribs["contrib_lc_num"],
                                             contribs["week"], cp_week, pre, post)
            rec["lifecycle_shift"] = round((lc_post or 0) - (lc_pre or 0), 4)

        # Competitor shift (optional)
        if comp_col:
            cp_pre, cp_post = pre_post_mean(contribs[comp_col[0]],
                                             contribs["week"], cp_week, pre, post)
            rec["competitor_shift_rel"] = round(
                ((cp_post or 0) - (cp_pre or 0)) / (abs(cp_pre or 0) + 1e-9), 4)

        # Residual z-score post CP
        post_mask = ((contribs["week"] >= cp_week) &
                     (contribs["week"] < cp_week + pd.Timedelta(weeks=post)))
        rec["residual_z_post"] = round(
            float(contribs.loc[post_mask, "residual_z"].mean())
            if post_mask.any() else 0, 4)

        rec["classification"] = classify_cp(rec)
        records.append(rec)

    report = pd.DataFrame(records)

    print(f"\n  Classification summary:")
    print(report.groupby("classification")["cp_week"].count().to_string())
    print(f"\n  Full report preview:")
    display_cols = ["cp_week", "cp_prob", "classification",
                    "total_contrib_shift_rel", "field_contrib_shift_rel",
                    "residual_z_post"]
    print(report[[c for c in display_cols if c in report.columns]].to_string(index=False))

    report.to_csv(INTEG_OUT, index=False)
    print(f"\n  Integration report written -> {INTEG_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    integration()
