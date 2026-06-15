"""
Layer 2 — Column semantic subtype matcher.

Matching pipeline (in order, first match wins):
  1. Normalize column name (lowercase, special chars → _)
  2. Abbreviation expansion (abbreviations.yaml)
  3. Candidate list matching (candidates.yaml) — exact / substring
  4. Structural heuristics:
       - token overlap with candidate lists → partial match
       - dtype = date/timestamp → "date"
       - dtype = numeric + not a key suffix → "unclassified_metric"  [guardrail gate]
       - dtype = object + low unique count → "dimension_attribute"
  5. Guardrail gate: emit warning for unclassified_metric and unknown columns

Confidence scale:
  1.0 — candidate list exact match
  0.9 — dtype-based date heuristic
  0.8 — abbreviation expansion + then candidate list hit
  0.6 — token overlap / substring heuristic
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
_NUMERIC_DTYPES = {"int", "int64", "int32", "float", "float64", "float32", "double", "number", "numeric", "bigint"}
_STRING_DTYPES = {"object", "str", "string", "varchar", "char", "text"}


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


def _candidate_match(
    name: str,
    candidates: dict[str, list[str]],
    key_suffixes: list[str],
    flag_prefixes: list[str],
    flag_suffixes: list[str],
) -> tuple[str, float] | None:
    """Return (subtype, confidence) if name matches any candidate list, else None."""
    # Key suffix check first (surrogate/natural keys should not be mistaken for measures)
    if _has_key_suffix(name, key_suffixes):
        return ("key", 1.0)

    # Flag prefix/suffix
    if _has_flag_pattern(name, flag_prefixes, flag_suffixes):
        return ("flag", 1.0)

    # Exact match against named lists
    for subtype, candidate_list in [
        ("date", candidates.get("date_candidates", [])),
        ("geography", candidates.get("geography_candidates", [])),
        ("measure", candidates.get("measure_candidates", [])),
        ("segment", candidates.get("segment_candidates", [])),
    ]:
        if name in candidate_list:
            return (subtype, 1.0)

    # Substring match (name contains or is contained by a candidate)
    for subtype, candidate_list in [
        ("date", candidates.get("date_candidates", [])),
        ("geography", candidates.get("geography_candidates", [])),
        ("measure", candidates.get("measure_candidates", [])),
        ("segment", candidates.get("segment_candidates", [])),
    ]:
        for c in candidate_list:
            if c in name or name in c:
                return (subtype, 0.8)

    return None


def _token_overlap_match(
    name: str,
    candidates: dict[str, list[str]],
) -> tuple[str, float] | None:
    """Split name on _ and check if any token appears in any candidate list."""
    tokens = set(name.split("_"))
    tokens.discard("")

    flat: dict[str, list[str]] = {
        "date": candidates.get("date_candidates", []),
        "geography": candidates.get("geography_candidates", []),
        "measure": candidates.get("measure_candidates", []),
        "segment": candidates.get("segment_candidates", []),
    }
    for subtype, candidate_list in flat.items():
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
            )

        # Step 4 — heuristic fallbacks
        dtype_fam = _dtype_family(dtype)

        # 4a — dtype date
        if dtype_fam == "date":
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype="date",
                match_source="heuristic_dtype",
                expanded_name=expanded_name,
                confidence=0.9,
            )

        # 4b — token overlap
        result = _token_overlap_match(norm_for_matching, self._candidates)
        if result:
            subtype, confidence = result
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype=subtype,
                match_source="heuristic_token",
                expanded_name=expanded_name,
                confidence=confidence,
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
            )

        # 4d — low-cardinality string → dimension_attribute
        if dtype_fam == "string" and 0 < unique_count <= 50:
            return ColumnSpec(
                name=name,
                dtype=dtype,
                semantic_subtype="dimension_attribute",
                match_source="heuristic_token",
                expanded_name=expanded_name,
                confidence=0.4,
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
