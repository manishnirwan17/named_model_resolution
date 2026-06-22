"""
Column selection utilities — shared across all model runners and quality checks.

select_date_column      — pick the single BEST date column (scores all date candidates)
select_measure_column   — pick ONE best measure column (backward-compat)
select_measure_columns  — return ALL viable measure columns ranked by score
dedup_by_correlation    — remove near-duplicate columns (high Pearson corr) from a ranked list

Date scoring (higher = better time-series column):
  match_source == "heuristic_dtype" (real date/timestamp dtype)  +3.0
  match_source in ("candidate_list", "abbreviation_expanded")    +2.0
  match_source == "heuristic_token"                              +1.0
  confidence bonus                                               +0.5 * confidence
  null_pct penalty                                               -5.0 * null_pct
  date_grain detected                                            +1.0
  unique_count > 52                                              +0.5
  unique_count > 12                                              +0.3

Measure scoring (applied to both 'measure' and 'unclassified_metric' columns):
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
# Date column selection
# ---------------------------------------------------------------------------

def select_date_column(
    specs: list[ColumnSpec],
    profiles: dict[str, ColumnProfile],
) -> tuple[str | None, str | None]:
    """
    Return (best_date_col_name, warning_or_None).

    Scores all date-subtype columns and returns the highest scorer.
    Strongly prefers columns with a proper date/timestamp dtype
    (match_source="heuristic_dtype"), high fill rate, and a detected
    date grain.  Emits a warning when the selected column was matched
    only via token heuristic (low-confidence selection).

    Used by runners (BOCPD, MMM), quality checks (date_continuity,
    autocorrelation, fill_rate), and the pipeline window slicer so
    every component uses the same date column.
    """
    date_specs = [s for s in specs if s.semantic_subtype == "date"]
    if not date_specs:
        return None, None

    def _score(spec: ColumnSpec) -> float:
        score = 0.0
        ms = spec.match_source
        if ms == "heuristic_dtype":
            score += 3.0
        elif ms in ("candidate_list", "abbreviation_expanded"):
            score += 2.0
        elif ms == "heuristic_token":
            score += 1.0
        score += spec.confidence * 0.5

        p = profiles.get(spec.name)
        if p is None:
            return score
        score -= p.null_pct * 5.0
        if p.date_grain is not None:
            score += 1.0
        if p.unique_count > 52:
            score += 0.5
        elif p.unique_count > 12:
            score += 0.3
        return score

    best = max(date_specs, key=_score)
    warning: str | None = None
    if best.match_source == "heuristic_token":
        warning = (
            f"Date column '{best.name}' selected via token heuristic (low confidence). "
            f"Consider adding it to date_candidates in candidates.yaml."
        )
    return best.name, warning


# ---------------------------------------------------------------------------
# Shared measure-scoring helper
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

    # ── CV bonus (meaningful variation relative to mean) ──────────────────────
    if p.std is not None and p.mean is not None and abs(p.mean) > 1e-6:
        cv = p.std / abs(p.mean)
        if cv < 0.02:
            score -= 1.0    # near-constant (also surfaced by zero_variance check)
        else:
            score += min(cv, 2.0) * 0.5

    # ── Unique-count: penalty for degenerate, bonus for rich ─────────────────
    if p.unique_count <= 2:
        score -= 3.0    # constant or binary — useless as a continuous time-series target
    elif p.unique_count <= 5:
        score -= 1.0    # very few levels — likely ordinal/categorical, not a continuous measure
    elif p.unique_count > 50:
        score += 0.5
    elif p.unique_count > 10:
        score += 0.2

    # ── Distribution shape bonus ──────────────────────────────────────────────
    # Calibrated on real datasets:
    #   engagement counts (f2f, touches): sk 1–7
    #   sales volume (qty_sold):          sk ~132
    #   claims volume:                    sk 193–228
    #   price (WAC):                      sk –1.3 (near-normal)
    if p.skewness is not None:
        sk = p.skewness
        if 0.3 <= sk <= 15:
            score += 0.4    # moderate right-skew — engagement/sales pattern
        elif sk > 15:
            score += 0.2    # extreme right-skew (claims volume, order qty) — valid measure
        elif abs(sk) <= 1.5:
            score += 0.2    # near-normal (WAC, rates) — good for ARIMA

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
