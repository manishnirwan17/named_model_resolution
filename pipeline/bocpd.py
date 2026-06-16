"""
02_bocpd.py — Bayesian Online Change-Point Detection on the weekly sales series.

Engine: `bayesian_changepoint_detection` (Adams & MacKay 2007, Normal-Gamma /
StudentT predictive). Install:  pip install bayesian_changepoint_detection

Reads:  MARKET_SERIES   (output of 01_data_prep.py)
Writes: CP_PROBS        (week, log_sales, cp_prob, exp_run_length per row)
        CP_CANDIDATES   (flagged change-point dates with context)

Run locally:  uv run python models/02_bocpd.py
On Databricks: %run ./02_bocpd  (single-node; no Spark needed for BOCPD)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import bayesian_changepoint_detection.online_changepoint_detection as oncd

from pipeline.config import (
    MARKET_SERIES, CP_PROBS, CP_CANDIDATES,
    read_parquet, write_parquet, write_csv, PARAMS, DATASET,
)


# ── BOCPD via the package, returning the full run-length matrix ───────────────
def run_bocpd(log_sales: np.ndarray,
              mu_0: float, kappa_0: float, alpha_0: float, beta_0: float,
              hazard_lam: int = 52,
              short_window: int = 3) -> dict:
    """
    Run BOCPD over the full series using bayesian_changepoint_detection.

    Returns a dict with:
      "R"             : (T+1, T+1) run-length posterior matrix
      "cp_prob"       : length-T array — P(run length <= short_window) per step.
                        Rises toward 1 right after a change-point (run length is
                        short because a new run just started).
      "exp_run_length": length-T array — E[run length] per step. Collapses toward
                        0 at a change-point. The most robust single CP signal.

    Note on the prior: pass beta_0 estimated from the FIRST-DIFFERENCE variance of
    the training series, not the level variance. Level variance is inflated by trend
    and makes a long run over-confident, muting real shifts.
    """
    data = np.asarray(log_sales, dtype=float)
    T    = len(data)

    obs = oncd.StudentT(alpha=alpha_0, beta=beta_0, kappa=kappa_0, mu=mu_0)
    R, _maxes = oncd.online_changepoint_detection(
        data, partial(oncd.constant_hazard, hazard_lam), obs)

    # Column t+1 of R is the run-length posterior AFTER observing data[t].
    rl_index = np.arange(T + 1)
    cp_prob        = np.empty(T)
    exp_run_length = np.empty(T)
    w = short_window
    for t in range(T):
        col = R[:, t + 1]
        cp_prob[t]        = col[:w + 1].sum()        # P(run length <= w)
        exp_run_length[t] = float((rl_index * col).sum())

    return {"R": R, "cp_prob": cp_prob, "exp_run_length": exp_run_length}


def extract_candidates(weeks: pd.Series, series_values: np.ndarray,
                        cp_prob: np.ndarray, exp_run_length: np.ndarray,
                        threshold: float, min_dist: int,
                        series_col: str = "log_target") -> pd.DataFrame:
    """
    Extract change-point candidates.

    Primary signal: peaks in cp_prob above `threshold` with at least `min_dist`
    weeks between them.  `series_col` sets the name of the target series column
    in the output (defaults to the DATASET.target_log_col value passed by bocpd()).
    """
    peaks, _props = find_peaks(cp_prob, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        return pd.DataFrame(
            columns=["week", "cp_prob", "exp_run_length", series_col, "week_idx"])

    return (pd.DataFrame({
                "week":           weeks.iloc[peaks].values,
                "cp_prob":        cp_prob[peaks],
                "exp_run_length": exp_run_length[peaks],
                series_col:       series_values[peaks],
                "week_idx":       peaks,
            })
            .sort_values("cp_prob", ascending=False)
            .reset_index(drop=True))


def bocpd() -> None:
    print("=" * 60)
    print("  02  BOCPD  (bayesian_changepoint_detection)")
    print("=" * 60)

    mkt   = read_parquet(MARKET_SERIES)
    train = mkt[mkt["split"] == "train"].copy()

    log_col = DATASET.target_log_col
    log_sales_all   = mkt[log_col].values
    log_sales_train = train[log_col].values

    # Prior. NOTE: beta_0 from FIRST-DIFFERENCE variance (week-to-week noise),
    # not level variance — level variance is trend-inflated and mutes real shifts.
    mu_0    = float(log_sales_train.mean())
    kappa_0 = 1.0
    alpha_0 = 1.0
    diff_var = float(np.diff(log_sales_train).var())
    beta_0   = diff_var if diff_var > 0 else float(log_sales_train.var())

    print(f"\n  Prior: mu_0={mu_0:.3f}  beta_0={beta_0:.5f} (from diff-variance)")
    print(f"  Hazard lambda: {PARAMS['BOCPD_LAMBDA']} weeks  "
          f"(P(CP per week) = {1/PARAMS['BOCPD_LAMBDA']:.3f})")
    print(f"  Running BOCPD on {len(log_sales_all)} weeks ...")

    out = run_bocpd(
        log_sales_all,
        mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0,
        hazard_lam=PARAMS["BOCPD_LAMBDA"],
    )
    cp_prob        = out["cp_prob"]
    exp_run_length = out["exp_run_length"]

    # ── Save full probability series ─────────────────────────────────────────
    keep_cols = [DATASET.week_col, log_col, "split"]
    if DATASET.lifecycle_col and DATASET.lifecycle_col in mkt.columns:
        keep_cols.append(DATASET.lifecycle_col)
    comp_cols = [c for c in DATASET.competitor_channels if c in mkt.columns]
    keep_cols += comp_cols
    prob_df = mkt[keep_cols].copy()
    prob_df["cp_prob"]        = cp_prob
    prob_df["exp_run_length"] = exp_run_length
    write_parquet(prob_df, CP_PROBS)
    print(f"\n  CP probabilities written -> {CP_PROBS}")
    print(f"  cp_prob range: [{cp_prob.min():.3f}, {cp_prob.max():.3f}]  "
          f"(flat near 1/lambda would mean no signal)")

    # ── Extract candidates ────────────────────────────────────────────────────
    candidates = extract_candidates(
        mkt["week"], log_sales_all, cp_prob, exp_run_length,
        threshold=PARAMS["BOCPD_THRESHOLD"],
        min_dist=PARAMS["BOCPD_MIN_DIST"],
        series_col=log_col,
    )
    if hasattr(candidates, "to_csv"):
        write_csv(candidates, CP_CANDIDATES)
# 
#    from pipeline.config import write_csv  # UC-aware on Databricks, file locally
# 
#   (transition to parameterized models instead of hard coded data)

    print(f"  {len(candidates)} change-point candidates written -> {CP_CANDIDATES}")

    if len(candidates):
        print(f"\n  Top candidates:")
        print(candidates.head(8).to_string(index=False))

    # ── Quick validation against expected organic CPs (if configured) ────────
    if DATASET.organic_cp_timestamps:
        lag = pd.Timedelta(days=PARAMS["LAG_TOLERANCE_DAYS"])
        print("\n  Organic CP validation (+-6 week window):")
        for name, ts in DATASET.organic_cp_timestamps.items():
            if len(candidates):
                window = candidates[
                    (candidates["week"] >= ts - lag) &
                    (candidates["week"] <= ts + lag)
                ]
                hit = "DETECTED" if len(window) else "MISSED"
            else:
                hit = "MISSED"
            print(f"    {name} ({ts.date()}) -> {hit}")

    print("=" * 60)


if __name__ == "__main__":
    bocpd()
