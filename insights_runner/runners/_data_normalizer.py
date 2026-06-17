"""
Pre-aggregation utilities shared across model runners.

normalize_to_series  — collapses territory/HCP-level multi-row-per-date data to a
                       single market-level time series.  When column_specs is provided,
                       detects the extra grain dimension (segment/key/geography columns)
                       for a two-step aggregation that preserves count information.

normalize_grain      — resamples a weekly dataset to monthly grain.  Triggered only
                       when thresholds.yaml has target_grain: "monthly" for the model.

_infer_agg_method    — heuristic: columns with rate/ratio/pct/share keywords → mean,
                       everything else → sum.
"""

from __future__ import annotations

import re

import pandas as pd

_RATE_KEYWORDS = ("rate", "ratio", "pct", "percent", "share", "avg", "mean")

# Column subtypes that identify an extra grain dimension beyond the date axis
_EXTRA_GRAIN_SUBTYPES = frozenset({"segment", "key", "geography"})

# Quarter string: "2025 Q4", "2025Q4", "Q4 2025", "Q4/2025" etc.
_QUARTER_RE = re.compile(
    r"(?:(\d{4})\s*[Qq](\d)|[Qq](\d)\s*[/\-]?\s*(\d{4}))"
)


def parse_dates_flexible(series: pd.Series) -> pd.Series:
    """
    Parse a date Series, with fallback handling for quarter strings like
    '2025 Q4' or 'Q4 2025' that pandas cannot parse natively.

    Returns a datetime64[ns] Series (NaT for unparseable values).
    """
    try:
        return pd.to_datetime(series)
    except Exception:
        pass

    # Quarter-string fallback: map each value individually
    def _parse_one(v):
        if v is None or (isinstance(v, float) and v != v):  # NaN check
            return pd.NaT
        s = str(v).strip()
        m = _QUARTER_RE.search(s)
        if m:
            year   = m.group(1) or m.group(4)
            qdigit = m.group(2) or m.group(3)
            try:
                return pd.Period(f"{year}Q{qdigit}", freq="Q").start_time
            except Exception:
                pass
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    try:
        return series.apply(_parse_one)
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def _infer_agg_method(col_name: str) -> str:
    """Sum for counts/volumes; mean for rates/ratios."""
    lower = col_name.lower()
    return "mean" if any(kw in lower for kw in _RATE_KEYWORDS) else "sum"


def normalize_to_series(
    df: pd.DataFrame,
    date_col: str,
    measure_cols: list[str],
    column_specs=None,
) -> tuple[pd.DataFrame, str | None]:
    """
    If df has multiple rows per date (territory/HCP-level data), aggregate to
    one row per date.

    When column_specs is provided (list[ColumnSpec]), columns classified as
    segment / key / geography are treated as the extra grain dimension:
      Step 1 — groupby([date, *grain_cols]) collapses within-segment duplicates
      Step 2 — national rollup groupby(date) + nunique count of the primary grain

    Without specs (or when no matching grain cols exist in df), falls back to
    groupby-date-only with an n_obs count column.

    Returns (aggregated_df, note_or_None).
    - Rate/ratio/pct columns → mean;  all others → sum
    - Count column added: n_{primary_grain_col} (segment-aware) or n_obs (fallback)
    - If no duplicates: returns original df unchanged with note=None.
    """
    if not df[date_col].duplicated().any():
        return df, None

    # ── Detect extra grain dimension from column_specs ────────────────────────
    extra_grain_cols: list[str] = []
    if column_specs:
        extra_grain_cols = [
            s.name for s in column_specs
            if s.semantic_subtype in _EXTRA_GRAIN_SUBTYPES
            and s.name in df.columns
            and s.name != date_col
        ]

    agg_dict = {
        col: _infer_agg_method(col)
        for col in measure_cols
        if col in df.columns
    }
    if not agg_dict:
        return df, None

    n_rows_before = len(df)

    if extra_grain_cols:
        primary_grain = extra_grain_cols[0]

        # Step 1: collapse within-segment-period duplicates (if any)
        step1 = (
            df.groupby([date_col] + extra_grain_cols, as_index=False)
            .agg(agg_dict)
        )

        # Step 2: national rollup — sum/mean across segments + unique count
        count_per_date = (
            step1.groupby(date_col)[primary_grain]
            .nunique()
            .rename(f"n_{primary_grain}")
            .reset_index()
        )
        result = (
            step1.groupby(date_col, as_index=False)
            .agg(agg_dict)
            .sort_values(date_col)
            .reset_index(drop=True)
            .merge(count_per_date, on=date_col, how="left")
        )
        n_unique = int(count_per_date[f"n_{primary_grain}"].max())
        note = (
            f"Aggregated {n_rows_before:,} {primary_grain}-level rows "
            f"({n_unique} unique {primary_grain}s) to {len(result)} national periods."
        )
    else:
        # Fallback: no grain info — group by date, attach row-count per period
        count_per_date = (
            df.groupby(date_col)
            .size()
            .rename("n_obs")
            .reset_index()
        )
        result = (
            df.groupby(date_col, as_index=False)
            .agg(agg_dict)
            .sort_values(date_col)
            .reset_index(drop=True)
            .merge(count_per_date, on=date_col, how="left")
        )
        n_max = int(count_per_date["n_obs"].max())
        methods = ", ".join(f"'{c}' ({m})" for c, m in agg_dict.items())
        note = (
            f"Multiple rows per date detected (max {n_max} per date) -- "
            f"aggregated to market-level time series: {methods}."
        )

    return result, note


def normalize_grain(
    df: pd.DataFrame,
    date_col: str,
    measure_cols: list[str],
    target_grain: str = "monthly",
) -> tuple[pd.DataFrame, str | None]:
    """
    Resample df to a coarser grain.  Currently only weekly -> monthly.
    Grain is auto-detected from median inter-observation gap: if the data is
    already monthly (median gap >= 25 days) nothing is done.

    Returns (df, note_or_None).
    """
    if target_grain != "monthly":
        return df, None

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        dates = df[date_col].sort_values()
        gaps = dates.diff().dropna().dt.days
        if gaps.empty or float(gaps.median()) >= 25:
            return df, None  # already monthly or coarser

        agg_dict = {
            col: _infer_agg_method(col)
            for col in measure_cols
            if col in df.columns
        }
        if not agg_dict:
            return df, None

        df["_period"] = df[date_col].dt.to_period("M")
        agg_dict_with_date = {date_col: "last", **agg_dict}
        result = (
            df.groupby("_period", as_index=False)
            .agg(agg_dict_with_date)
            .drop(columns=["_period"])
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        n_before, n_after = len(df), len(result)
        note = (
            f"Resampled from weekly to monthly grain "
            f"({n_before} weekly rows -> {n_after} monthly rows). "
            f"Month-end dates used as representative dates."
        )
        return result, note
    except Exception as exc:
        return df, f"Grain normalization skipped: {exc}"
