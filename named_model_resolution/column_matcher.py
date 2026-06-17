"""
Layer 2 — Column semantic subtype matcher.

Matching pipeline (in order, first match wins):
  1. Normalize column name (lowercase, special chars → _)
  2. Abbreviation expansion (abbreviations.yaml)
  3. Candidate list matching (candidates.yaml) — exact / token-boundary substring
  4. Structural heuristics:
       - token overlap with candidate lists → partial match
       - dtype = date/timestamp → "date"
       - dtype = numeric + not a key suffix → "unclassified_metric"  [guardrail gate]
       - dtype = object + low unique count → "dimension_attribute"
  5. Guardrail gate: emit warning for unclassified_metric and unknown columns

Confidence scale:
  1.0 — candidate list exact match
  0.9 — dtype-based date heuristic
  0.8 — abbreviation expansion + then candidate list hit, OR token-boundary substring match
  0.6 — token overlap / partial heuristic
  0.5 — guardrail_metric (numeric fallback)
  0.4 — dimension_attribute (low-cardinality string)
  0.0 — unknown (genuinely unclassifiable)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .models import ColumnSpec

_DATE_DTYPES = {"date", "datetime", "datetime64", "timestamp", "datetime64[ns]", "datetime64[us]"}

# Token-overlap date classification is gated: a column name must contain at least one
# of these tokens to qualify as `date` via token overlap.  Without this gate, columns
# like HCP_FIRST_NAME match `first_diagnosis_date` on the token "first" alone.
_DATE_GATE_TOKENS = frozenset({
    "date", "dt", "day", "week", "wk", "month", "mth",
    "year", "yr", "quarter", "qtr", "period", "time",
})
_NUMERIC_DTYPES = {"int", "int64", "int32", "float", "float64", "float32", "double", "number", "numeric", "bigint"}
_STRING_DTYPES = {"object", "str", "string", "varchar", "char", "text"}

# Date-suffix override: column names ending with these tokens are always dates,
# regardless of earlier tokens (e.g. CALL_OR_RTE_SENT_DATE has "rte" channel token
# but ends in "date" → must be a date).
_DATE_SUFFIX_TOKENS_GUARD = frozenset({"date", "dt", "datetime", "timestamp", "ts"})

# Rolling metric guard: _last_90d, _last_30d patterns are count/rate windows,
# never channels.  Returning None lets the heuristic dtype path take over
# (numeric → unclassified_metric; string → dimension_attribute).
_ROLLING_METRIC_RE = re.compile(r"_last_\d+[dD]$")


def _normalize_name(name: str) -> str:
    """Lowercase and replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _dtype_family(dtype: str) -> str:
    """Coarse dtype family: 'date' | 'numeric' | 'string' | 'other'."""
    d = dtype.strip().lower()
    if d in _DATE_DTYPES or "date" in d or "time" in d:
        return "date"
    if d in _NUMERIC_DTYPES or d.startswith("int") or d.startswith("float"):
        return "numeric"
    if d in _STRING_DTYPES:
        return "string"
    return "other"


def _has_key_suffix(name: str, suffixes: list[str]) -> bool:
    return any(name.endswith(s) for s in suffixes)


def _has_flag_pattern(name: str, prefixes: list[str], suffixes: list[str]) -> bool:
    return any(name.startswith(p) for p in prefixes) or any(name.endswith(s) for s in suffixes)


def _is_numeric_indicator(
    name: str,
    numeric_suffixes: list[str],
    numeric_prefixes: list[str],
) -> bool:
    """True for names like week_idx, n_patients, row_num — numeric indices/counts, not dates."""
    return (
        any(name.endswith(s) for s in numeric_suffixes)
        or any(name.startswith(p) for p in numeric_prefixes)
    )


def _token_boundary_match(candidate: str, name: str) -> bool:
    """
    True when candidate's tokens are a subset of name's tokens (or vice versa),
    using underscore as the word delimiter.

    This prevents 'week' matching 'n_weeks' (different token: 'weeks' ≠ 'week')
    and 'y' matching 'year' (not a token in 'year').
    """
    c_tokens = {t for t in candidate.split("_") if t}
    n_tokens = {t for t in name.split("_") if t}
    if not c_tokens or not n_tokens:
        return False
    return c_tokens.issubset(n_tokens) or n_tokens.issubset(c_tokens)


def _candidate_match(
    name: str,
    candidates: dict[str, list[str]],
    key_suffixes: list[str],
    flag_prefixes: list[str],
    flag_suffixes: list[str],
    numeric_ind_suffixes: list[str],
    numeric_ind_prefixes: list[str],
) -> tuple[str, float] | None:
    """Return (subtype, confidence) if name matches any candidate list, else None."""
    # Key suffix check first (surrogate/natural keys should not be mistaken for measures)
    if _has_key_suffix(name, key_suffixes):
        return ("key", 1.0)

    # Flag prefix/suffix
    if _has_flag_pattern(name, flag_prefixes, flag_suffixes):
        return ("flag", 1.0)

    # Date suffix override: column ending in _date, _dt, _datetime, _ts, etc.
    # is always a date regardless of channel/measure tokens earlier in the name.
    # Example: CALL_OR_RTE_SENT_DATE has "rte" (channel token) but ends in "date".
    last_token = name.split("_")[-1] if "_" in name else name
    if last_token in _DATE_SUFFIX_TOKENS_GUARD:
        return ("date", 0.8)

    # Rolling metric guard: _last_90d, _last_30d, etc. are count/rate windows,
    # not channels.  Return None so heuristic dtype path handles them.
    if _ROLLING_METRIC_RE.search(name):
        return None

    is_numeric_ind = _is_numeric_indicator(name, numeric_ind_suffixes, numeric_ind_prefixes)

    _ORDERED_SUBTYPES = [
        # channel before measure so f2f/email columns don't accidentally hit measure via token
        ("channel", candidates.get("channel_candidates", [])),
        ("date", candidates.get("date_candidates", [])),
        ("geography", candidates.get("geography_candidates", [])),
        ("measure", candidates.get("measure_candidates", [])),
        ("segment", candidates.get("segment_candidates", [])),
    ]

    # Exact match
    for subtype, candidate_list in _ORDERED_SUBTYPES:
        # Skip date matching for known numeric-indicator columns
        if subtype == "date" and is_numeric_ind:
            continue
        if name in candidate_list:
            return (subtype, 1.0)

    # Token-boundary substring match
    for subtype, candidate_list in _ORDERED_SUBTYPES:
        if subtype == "date" and is_numeric_ind:
            continue
        for c in candidate_list:
            if c != name and _token_boundary_match(c, name):
                return (subtype, 0.8)

    return None


def _token_overlap_match(
    name: str,
    candidates: dict[str, list[str]],
    numeric_ind_suffixes: list[str],
    numeric_ind_prefixes: list[str],
) -> tuple[str, float] | None:
    """Split name on _ and check if any token appears in any candidate list."""
    tokens = set(name.split("_"))
    tokens.discard("")

    is_numeric_ind = _is_numeric_indicator(name, numeric_ind_suffixes, numeric_ind_prefixes)

    flat: dict[str, list[str]] = {
        "channel": candidates.get("channel_candidates", []),
        "date": candidates.get("date_candidates", []),
        "geography": candidates.get("geography_candidates", []),
        "measure": candidates.get("measure_candidates", []),
        "segment": candidates.get("segment_candidates", []),
    }
    for subtype, candidate_list in flat.items():
        if subtype == "date" and is_numeric_ind:
            continue
        # Date gate: only allow date classification via token overlap when the column
        # name itself contains a recognisable date-specific token.  This prevents false
        # positives such as HCP_FIRST_NAME matching first_diagnosis_date on "first".
        if subtype == "date" and not (tokens & _DATE_GATE_TOKENS):
            continue
        for c in candidate_list:
            c_tokens = set(c.split("_"))
            if tokens & c_tokens:
                return (subtype, 0.6)
    return None


class ColumnMatcher:
    def __init__(self, configs_dir: str | Path) -> None:
        configs_dir = Path(configs_dir)
        with (configs_dir / "abbreviations.yaml").open() as f:
            self._abbr: dict[str, str] = yaml.safe_load(f) or {}
        with (configs_dir / "candidates.yaml").open() as f:
            self._candidates: dict[str, Any] = yaml.safe_load(f) or {}

        self._key_suffixes: list[str] = self._candidates.get("key_candidates_suffixes", [])
        self._flag_prefixes: list[str] = self._candidates.get("flag_prefixes", [])
        self._flag_suffixes: list[str] = self._candidates.get("flag_suffixes", [])
        self._numeric_ind_suffixes: list[str] = self._candidates.get("numeric_indicator_suffixes", [])
        self._numeric_ind_prefixes: list[str] = self._candidates.get("numeric_indicator_prefixes", [])
        self._business_hints: dict[str, str] = self._candidates.get("business_hints") or {}

    def _get_hint(self, name: str, norm: str) -> str | None:
        """Look up a business hint by original name or normalized name."""
        return self._business_hints.get(name) or self._business_hints.get(norm) or None

    def match(
        self,
        name: str,
        dtype: str,
        unique_count: int = 0,
        warnings: list[str] | None = None,
    ) -> ColumnSpec:
        if warnings is None:
            warnings = []

        # Step 1 — normalize
        norm = _normalize_name(name)

        # Step 2 — abbreviation expansion
        expanded_name: str | None = None
        match_source = "unmatched"
        if norm in self._abbr:
            expanded_name = self._abbr[norm]
            norm_for_matching = _normalize_name(expanded_name)
            _abbr_expanded = True
        else:
            norm_for_matching = norm
            _abbr_expanded = False

        # Step 3 — candidate list matching
        result = _candidate_match(
            norm_for_matching,
            self._candidates,
            self._key_suffixes,
            self._flag_prefixes,
            self._flag_suffixes,
            self._numeric_ind_suffixes,
            self._numeric_ind_prefixes,
        )
        if result:
            subtype, confidence = result
            if _abbr_expanded:
                match_source = "abbreviation_expanded"
                confidence = min(confidence, 0.8)
            else:
                match_source = "candidate_list"
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype=subtype,
                match_source=match_source,
                expanded_name=expanded_name,
                confidence=confidence,
                business_hint=self._get_hint(name, norm),
            )

        # Step 4 — heuristic fallbacks
        dtype_fam = _dtype_family(dtype)

        # 4a — dtype date (only for actual date dtypes, not numeric indicators)
        if dtype_fam == "date" and not _is_numeric_indicator(
            norm_for_matching, self._numeric_ind_suffixes, self._numeric_ind_prefixes
        ):
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype="date",
                match_source="heuristic_dtype",
                expanded_name=expanded_name,
                confidence=0.9,
                business_hint=self._get_hint(name, norm),
            )

        # 4b — token overlap (skipped for rolling-metric columns — they go straight
        #      to the numeric guardrail so _last_90d patterns don't hit channel tokens)
        if _ROLLING_METRIC_RE.search(norm_for_matching):
            result = None
        else:
            result = _token_overlap_match(
                norm_for_matching,
                self._candidates,
                self._numeric_ind_suffixes,
                self._numeric_ind_prefixes,
            )
        if result:
            subtype, confidence = result
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype=subtype,
                match_source="heuristic_token",
                expanded_name=expanded_name,
                confidence=confidence,
                business_hint=self._get_hint(name, norm),
            )

        # 4c — numeric fallback → unclassified_metric (guardrail gate)
        if dtype_fam == "numeric" and not _has_key_suffix(norm_for_matching, self._key_suffixes):
            warnings.append(
                f"Column '{name}' (dtype={dtype}) could not be matched to a known subtype. "
                f"Classified as 'unclassified_metric' — will be offered to models that accept "
                f"generic measures. Expand candidates.yaml or abbreviations.yaml if this is a "
                f"known measure."
            )
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype="unclassified_metric",
                match_source="guardrail_metric",
                expanded_name=expanded_name,
                confidence=0.5,
                business_hint=self._get_hint(name, norm),
            )

        # 4d — low-cardinality string → dimension_attribute
        if dtype_fam == "string" and 0 < unique_count <= 50:
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype="dimension_attribute",
                match_source="heuristic_cardinality",
                expanded_name=expanded_name,
                confidence=0.4,
                business_hint=self._get_hint(name, norm),
            )

        # 5 — genuinely unclassifiable
        warnings.append(
            f"Column '{name}' (dtype={dtype}) could not be classified. "
            f"It will not be used for model routing. "
            f"Consider adding it to candidates.yaml or abbreviations.yaml."
        )
        return ColumnSpec(
            name=name,
            dtype=dtype,
            semantic_subtype="unknown",
            match_source="unmatched",
            expanded_name=expanded_name,
            confidence=0.0,
            business_hint=self._get_hint(name, norm),
        )

    def match_all(
        self,
        schema: dict[str, str],
        unique_counts: dict[str, int] | None = None,
        warnings: list[str] | None = None,
    ) -> list[ColumnSpec]:
        """Match every column in a schema dict {col_name: dtype}."""
        if unique_counts is None:
            unique_counts = {}
        if warnings is None:
            warnings = []
        return [
            self.match(col, dtype, unique_counts.get(col, 0), warnings)
            for col, dtype in schema.items()
        ]
