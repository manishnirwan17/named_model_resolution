"""
Measure column selection utilities — shared across all model runners.

select_measure_column   — pick ONE best column (backward-compat, used by existing callers)
select_measure_columns  — return ALL viable columns ranked by score
dedup_by_correlation    — remove near-duplicate columns (high Pearson corr) from a ranked list

Scoring (applied to both 'measure' and 'unclassified_metric' columns):
  measure subtype                 score = +inf  (always ranked above unclassified_metric)
  business_hint present           +2.0
  null_pct                        -3.0 * null_pct
  CV = std / |mean|  (capped 2.0) +0.5 * min(cv, 2.0)
  unique_count > 50               +0.5
  unique_count > 10               +0.2
  skewness in (0, 5)              +0.2
  value_max > 0                   +0.1
"""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ColumnProfile, ColumnSpec


# ---------------------------------------------------------------------------
# Shared scoring helper
# ---------------------------------------------------------------------------

def _score_spec(spec: ColumnSpec, profiles: dict[str, ColumnProfile]) -> float:
    """Return a numeric score for one measure/unclassified_metric column spec."""
    if spec.semantic_subtype == "measure":
        return float("inf")
    # unclassified_metric scoring
    score = 0.0
    if spec.business_hint:
        score += 2.0
    p = profiles.get(spec.name)
    if p is None:
        return score
    score -= p.null_pct * 3.0
    if p.std is not None and p.mean is not None and abs(p.mean) > 1e-6:
        score += min(p.std / abs(p.mean), 2.0) * 0.5
    if p.unique_count > 50:
        score += 0.5
    elif p.unique_count > 10:
        score += 0.2
    if p.skewness is not None and 0 < p.skewness < 5:
        score += 0.2
    if p.value_max is not None and p.value_max > 0:
        score += 0.1
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_measure_column(
    specs: list[ColumnSpec],
    profiles: dict[str, ColumnProfile],
) -> tuple[str | None, str | None]:
    """
    Return (column_name, warning_or_None) for a single best target column.

    If a proper 'measure' column exists, return it with no warning.
    Otherwise score all 'unclassified_metric' candidates and return the best
    with a warning explaining the selection and scores.
    """
    for s in specs:
        if s.semantic_subtype == "measure":
            return s.name, None

    candidates = [s for s in specs if s.semantic_subtype == "unclassified_metric"]
    if not candidates:
        return None, None

    best = max(candidates, key=lambda s: _score_spec(s, profiles))
    scores = {s.name: round(_score_spec(s, profiles), 3) for s in candidates}
    hint_note = f" (business hint: \"{best.business_hint}\")" if best.business_hint else ""
    warning = (
        f"No 'measure' column found. Selected '{best.name}' from unclassified_metric "
        f"candidates via statistical scoring{hint_note}. "
        f"Scores: {scores}. "
        f"Add '{best.name}' to measure_candidates in candidates.yaml to suppress this warning."
    )
    return best.name, warning


def select_measure_columns(
    specs: list[ColumnSpec],
    profiles: dict[str, ColumnProfile],
) -> list[tuple[str, float]]:
    """
    Return ALL viable measure/unclassified_metric columns sorted by score descending.

    Returns a list of (col_name, score) tuples.
    - 'measure' columns always outrank 'unclassified_metric' (score = +inf).
    - Empty list if no candidates exist.
    """
    candidates = [
        s for s in specs
        if s.semantic_subtype in {"measure", "unclassified_metric"}
    ]
    if not candidates:
        return []

    scored = [(s.name, _score_spec(s, profiles)) for s in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def dedup_by_correlation(
    df: pd.DataFrame,
    ranked_cols: list[str],
    threshold: float = 0.85,
) -> list[str]:
    """
    Greedy correlation deduplication of a ranked column list.

    Walk `ranked_cols` from highest-scored to lowest.  Keep each column unless
    its absolute Pearson correlation with any already-kept column exceeds
    `threshold`.  Columns that cannot be correlated (all-NaN, < 3 valid rows,
    or not present in df) are kept unconditionally.

    Returns a subset of `ranked_cols` preserving the original order.
    """
    kept: list[str] = []

    for col in ranked_cols:
        if col not in df.columns:
            kept.append(col)
            continue

        col_data = pd.to_numeric(df[col], errors="coerce")
        if col_data.count() < 3:
            # Can't compute a reliable correlation — treat as independent
            kept.append(col)
            continue

        drop = False
        for k in kept:
            if k not in df.columns:
                continue
            k_data = pd.to_numeric(df[k], errors="coerce")
            # Align on common non-null rows
            valid = col_data.notna() & k_data.notna()
            if valid.sum() < 3:
                continue
            try:
                r = col_data[valid].corr(k_data[valid])
                if r is not None and abs(r) > threshold:
                    drop = True
                    break
            except Exception:
                continue

        if not drop:
            kept.append(col)

    return kept
