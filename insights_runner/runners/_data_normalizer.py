"""
Pre-aggregation utilities shared across model runners.

normalize_to_series  — collapses territory/HCP-level multi-row-per-date data to a
                       single market-level time series.

normalize_grain      — resamples a weekly dataset to monthly grain.  Triggered only
                       when thresholds.yaml has target_grain: "monthly" for the model.

_infer_agg_method    — heuristic: columns with rate/ratio/pct/share keywords → mean,
                       everything else → sum.
"""

from __future__ import annotations

import re

import pandas as pd

_RATE_KEYWORDS = ("rate", "ratio", "pct", "percent", "share", "avg", "mean")

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
) -> tuple[pd.DataFrame, str | None]:
    """
    If df has multiple rows per date (territory/HCP-level data), aggregate to
    one row per date.

    Returns (df, note_or_None).
    - Columns with rate/ratio/pct names → mean
    - All other measure columns → sum
    - If no duplicates: returns original df unchanged with note=None.
    """
    if not df[date_col].duplicated().any():
        return df, None

    agg_dict = {
        col: _infer_agg_method(col)
        for col in measure_cols
        if col in df.columns
    }
    if not agg_dict:
        return df, None

    result = (
        df.groupby(date_col, as_index=False)
        .agg(agg_dict)
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    methods = ", ".join(f"'{c}' ({m})" for c, m in agg_dict.items())
    note = (
        f"Multiple rows per date detected -- likely territory/HCP-level data. "
        f"Aggregated to market-level time series: {methods}."
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
