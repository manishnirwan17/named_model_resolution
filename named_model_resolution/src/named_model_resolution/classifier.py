"""
Layer 1 — Table classifier: fact vs dimension vs unknown.

Rules:
  Fact      → ≥1 date column AND ≥1 (measure OR unclassified_metric) column
  Dimension → ≥1 key column AND majority of columns are dimension_attribute/flag/key,
              AND at most 1 date column
  Unknown   → neither rule satisfied

Tie-break: if a table name or its columns show > 0.5 Jaccard overlap with a
known DatamartCatalog entry, the catalog's implied type takes precedence.
(Catalog entries that end with _Fact are fact; _Summary/_Dim are dimension.)
"""

from __future__ import annotations

from .models import ColumnSpec, DatamartCatalog, TableClassification

_FACT_SUFFIXES = ("_fact", "_claims", "_fact_table")
_DIMENSION_SUFFIXES = ("_dim", "_dimension", "_summary", "_lookup", "_ref", "_map")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _catalog_type_hint(table_name: str) -> str | None:
    """Infer implied type from catalog naming convention."""
    lower = table_name.lower()
    if any(lower.endswith(s) for s in _FACT_SUFFIXES):
        return "fact"
    if any(lower.endswith(s) for s in _DIMENSION_SUFFIXES):
        return "dimension"
    return None


def classify(
    table_name: str,
    column_specs: list[ColumnSpec],
    catalog: DatamartCatalog,
) -> TableClassification:
    subtypes = [c.semantic_subtype for c in column_specs]
    subtype_set = set(subtypes)

    has_date = "date" in subtype_set
    has_measure = bool(subtype_set & {"measure", "unclassified_metric"})
    has_key = "key" in subtype_set
    n_cols = len(column_specs)
    n_dim_like = sum(
        1 for s in subtypes if s in {"dimension_attribute", "flag", "key", "segment"}
    )

    # ── Structural classification ─────────────────────────────────────────────
    if has_date and has_measure:
        structural_type = "fact"
    elif has_key and n_cols > 0 and (n_dim_like / n_cols) >= 0.5:
        structural_type = "dimension"
    else:
        structural_type = "unknown"

    # ── Catalog cross-check ───────────────────────────────────────────────────
    incoming_cols = {c.name.lower() for c in column_specs}
    # Also include expanded names
    incoming_cols |= {c.expanded_name.lower() for c in column_specs if c.expanded_name}

    best_match: str | None = None
    best_score: float = 0.0
    for dm_name, dm_spec in catalog.datamarts.items():
        catalog_cols = {col.lower() for col in dm_spec.columns}
        score = _jaccard(incoming_cols, catalog_cols)
        if score > best_score:
            best_score = score
            best_match = dm_name

    # If strong catalog match exists, use naming-convention hint from catalog entry
    final_type = structural_type
    if best_score >= 0.5 and best_match:
        hint = _catalog_type_hint(best_match)
        if hint:
            final_type = hint

    # If still unknown, try naming convention on the incoming table itself
    if final_type == "unknown":
        hint = _catalog_type_hint(table_name)
        if hint:
            final_type = hint

    return TableClassification(
        table_name=table_name,
        table_type=final_type,
        columns=column_specs,
        matched_catalog_entry=best_match if best_score > 0.0 else None,
        catalog_match_score=best_score,
    )
