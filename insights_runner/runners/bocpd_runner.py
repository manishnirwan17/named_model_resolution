"""
BOCPDRunner — bridges RouterResult to pipeline BOCPD implementation.

Runs BOCPD on ALL viable, non-redundant measure columns in the dataset.
Columns are ranked by the measure selector, then correlation-deduplicated
(threshold = _CORR_THRESHOLD_BOCPD = 0.85).  Results are stored per-column
under signals["by_measure"][col_name].

Translation:
  RouterResult.classification.columns (subtype == "date")    -> date column
  select_measure_columns() + dedup_by_correlation()          -> target columns
  RouterResult.column_profiles[target].skewness > 1.5        -> apply log1p
  RouterResult.column_profiles[date].date_grain              -> hazard_lam calibration
  _TRAIN_PERIODS = 180                                        -> train split (row count)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult

from .base import ModelRunner

_TRAIN_PERIODS    = 180       # row count for training window (grain-agnostic)
_HAZARD_LAM       = 52        # default expected run length (periods); overridden by grain
_CP_THRESHOLD     = 0.30
_CP_MIN_DIST      = 8         # minimum periods between changepoints
_CP_WINDOW        = 8         # half-width of context window around each changepoint

_CORR_THRESHOLD_BOCPD = 0.85  # drop measure columns more correlated than this

# Periods per year by grain — used to calibrate hazard_lam
_PERIODS_PER_YEAR: dict[str, int] = {
    "daily":   365,
    "weekly":   52,
    "monthly":  12,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_hazard_lam(
    df: pd.DataFrame,
    date_col: str,
    profiles: dict,
) -> int:
    """
    Determine hazard_lam (expected run length in periods) from the date column's
    grain.  Falls back to _HAZARD_LAM (52) when grain is unknown.
    """
    date_profile = profiles.get(date_col)
    grain = date_profile.date_grain if date_profile else None
    return _PERIODS_PER_YEAR.get(grain, _HAZARD_LAM)


def _run_bocpd_series(
    df: pd.DataFrame,
    date_col: str,
    measure_col: str,
    profiles: dict,
    hazard_lam: int,
    column_specs=None,
) -> dict:
    """
    Run the full BOCPD pipeline for a single (date_col, measure_col) pair.

    Returns a signals dict on success, or {"error": reason} on failure.
    The returned dict is stored under signals["by_measure"][measure_col].
    """
    from ._data_normalizer import normalize_to_series, parse_dates_flexible

    # ── Validate column presence ──────────────────────────────────────────────
    if measure_col not in df.columns:
        return {"error": f"column '{measure_col}' not in dataframe"}

    # ── Parse dates + collapse to single series ───────────────────────────────
    try:
        mkt = df[[date_col, measure_col]].copy()
        mkt[date_col] = parse_dates_flexible(mkt[date_col])
        mkt, agg_note = normalize_to_series(mkt, date_col, [measure_col], column_specs)
        mkt = mkt.sort_values(date_col).reset_index(drop=True)
    except Exception as exc:
        return {"error": f"date parsing / aggregation failed: {exc}"}

    # ── Log-transform target ──────────────────────────────────────────────────
    target_profile = profiles.get(measure_col)
    skewness = target_profile.skewness if target_profile else None
    if skewness is not None and skewness > 1.5:
        log_series    = np.log1p(mkt[measure_col].clip(lower=0).values.astype(float))
        transform_used = "log1p"
    else:
        log_series    = np.log(mkt[measure_col].clip(lower=1e-6).values.astype(float))
        transform_used = "log"

    # ── Priors from training window ───────────────────────────────────────────
    train_series = log_series[:_TRAIN_PERIODS]
    mu_0   = float(train_series.mean())
    diff_var = float(np.diff(train_series).var()) if len(train_series) > 1 else 0.0
    beta_0 = diff_var if diff_var > 0 else float(train_series.var())

    # ── Run BOCPD ─────────────────────────────────────────────────────────────
    try:
        from pipeline.bocpd import extract_candidates, run_bocpd
    except ImportError:
        return {
            "error": (
                "bayesian_changepoint_detection / pipeline.bocpd not installed; "
                "pip install -e .[pipeline]"
            )
        }

    try:
        out = run_bocpd(
            log_series,
            mu_0=mu_0,
            kappa_0=1.0,
            alpha_0=1.0,
            beta_0=beta_0,
            hazard_lam=hazard_lam,
        )
    except Exception as exc:
        return {"error": f"run_bocpd failed: {exc}"}

    cp_prob        = out["cp_prob"]
    exp_run_length = out["exp_run_length"]

    series_col_label = f"log_{measure_col}"
    try:
        candidates = extract_candidates(
            mkt[date_col],
            log_series,
            cp_prob,
            exp_run_length,
            threshold=_CP_THRESHOLD,
            min_dist=_CP_MIN_DIST,
            series_col=series_col_label,
        )
        cp_records = candidates.to_dict("records") if len(candidates) > 0 else []
        for rec in cp_records:
            for k, v in rec.items():
                if isinstance(v, pd.Timestamp):
                    rec[k] = str(v)
    except Exception:
        cp_records = []

    # ── Context windows ───────────────────────────────────────────────────────
    log_col_key = f"log_{measure_col}"
    raw_col_key = measure_col
    cp_context_windows = []
    for rec in cp_records:
        idx   = int(rec["week_idx"])
        start = max(0, idx - _CP_WINDOW)
        end   = min(len(log_series), idx + _CP_WINDOW + 1)

        window_series = [
            {
                "date":           str(mkt[date_col].iloc[i]),
                log_col_key:      round(float(log_series[i]), 4),
                raw_col_key:      round(float(mkt[measure_col].iloc[i]), 2),
                "cp_prob":        round(float(cp_prob[i]), 4),
                "exp_run_length": round(float(exp_run_length[i]), 2),
            }
            for i in range(start, end)
        ]

        cp_context_windows.append({
            "changepoint_date":       rec["week"],
            "cp_prob":                rec["cp_prob"],
            "exp_run_length":         rec["exp_run_length"],
            f"{log_col_key}_at_cp":   rec[series_col_label],
            f"{raw_col_key}_at_cp":   round(float(mkt[measure_col].iloc[idx]), 2),
            "window_periods":         _CP_WINDOW,
            "series":                 window_series,
        })

    cp_probs_series = [
        {
            "date":           str(mkt[date_col].iloc[i]),
            "cp_prob":        float(cp_prob[i]),
            "exp_run_length": float(exp_run_length[i]),
        }
        for i in range(len(cp_prob))
    ]

    signals = {
        "n_changepoints":     len(cp_records),
        "cp_candidates":      cp_records,
        "cp_context_windows": cp_context_windows,
        "cp_probs_series":    cp_probs_series,
        "model_params": {
            "mu_0":             round(mu_0, 6),
            "beta_0":           round(beta_0, 8),
            "hazard_lam":       hazard_lam,
            "threshold":        _CP_THRESHOLD,
            "min_dist":         _CP_MIN_DIST,
            "target_transform": transform_used,
        },
    }
    if agg_note:
        signals["aggregation_note"] = agg_note
    return signals


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BOCPDRunner(ModelRunner):
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        specs    = router_result.classification.columns
        profiles = {p.name: p for p in router_result.column_profiles}

        # ── Date column: score all candidates, pick the best ─────────────────
        from ._measure_selector import (
            dedup_by_correlation,
            select_date_column,
            select_measure_columns,
        )
        date_col, date_note = select_date_column(specs, profiles)
        if not date_col:
            return {"ran": False, "reason": "no date column identified"}
        if date_col not in df.columns:
            return {"ran": False, "reason": f"date column '{date_col}' missing from df"}

        notes: list[str] = []
        if date_note:
            notes.append(date_note)

        # ── Ranked + deduplicated measure columns ─────────────────────────────

        ranked   = select_measure_columns(specs, profiles)
        col_names = [c for c, _ in ranked]
        col_names = dedup_by_correlation(df, col_names, threshold=_CORR_THRESHOLD_BOCPD)

        if not col_names:
            return {
                "ran": False,
                "reason": "no viable measure columns identified",
            }

        # ── Grain / hazard_lam (same for all columns — date col is shared) ────
        hazard_lam = _detect_hazard_lam(df, date_col, profiles)

        # ── Grain label for model_params (shared across columns) ──────────────
        date_profile = profiles.get(date_col)
        grain = date_profile.date_grain if date_profile else None

        # ── Run per column ────────────────────────────────────────────────────
        by_measure: dict[str, dict] = {}

        # Collect measure-selection note for any unclassified_metric columns used
        unclassified_used = [
            c for c in col_names
            if any(s.name == c and s.semantic_subtype == "unclassified_metric" for s in specs)
        ]
        if unclassified_used:
            notes.append(
                f"Column(s) {unclassified_used} are unclassified_metric — "
                "selected via statistical scoring. Add to candidates.yaml to suppress."
            )

        for col in col_names:
            col_signals = _run_bocpd_series(df, date_col, col, profiles, hazard_lam,
                                            column_specs=specs)
            # Attach shared grain info into model_params (only when run succeeded)
            if "model_params" in col_signals:
                col_signals["model_params"]["date_grain"] = (
                    grain or "unknown (defaulted to weekly equivalent)"
                )
            by_measure[col] = col_signals

        result: dict = {
            "ran": True,
            "signals": {"by_measure": by_measure},
        }
        if notes:
            result["note"] = " ".join(notes)
        return result
