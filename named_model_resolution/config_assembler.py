"""
Layer 3 — Config assembler.

Turns TableClassification objects (one per dataset in a run) into ranked
ModelConfig lists by scoring each model's routing rule against the columns
present in the fact table.

Scoring formula:
  score = (required subtypes ALL satisfied → 1.0 per required, else 0 total)
        + (optional subtypes present × 0.3)
        + (preferred_measures matched × 0.5)

A model is only a candidate when ALL required subtypes are satisfied.

Returns ModelConfigs ranked descending by score.  Callers decide how many
to use (e.g. top-1 for hard routing, top-N for multi-model evaluation).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import DatamartCatalog, ModelConfig, TableClassification


def _subtypes_present(classification: TableClassification) -> set[str]:
    return {c.semantic_subtype for c in classification.columns}


def _measure_names_present(classification: TableClassification) -> set[str]:
    """Return normalised column names for measure/unclassified_metric columns."""
    names: set[str] = set()
    for c in classification.columns:
        if c.semantic_subtype in {"measure", "unclassified_metric"}:
            names.add(c.name.lower())
            if c.expanded_name:
                names.add(c.expanded_name.lower())
    return names


def _infer_join_keys(
    fact: TableClassification,
    dim: TableClassification,
) -> dict[str, str]:
    """Return {fact_col: dim_col} for columns of subtype 'key' shared between tables."""
    fact_keys = {c.name.lower(): c.name for c in fact.columns if c.semantic_subtype == "key"}
    dim_keys = {c.name.lower(): c.name for c in dim.columns if c.semantic_subtype == "key"}
    shared = set(fact_keys) & set(dim_keys)
    return {fact_keys[k]: dim_keys[k] for k in shared}


def assemble(
    target: TableClassification,
    all_classifications: list[TableClassification],
    catalog: DatamartCatalog,
    configs_dir: str | Path,
) -> list[ModelConfig]:
    """
    Build ranked ModelConfigs for `target` (must be a fact table).
    `all_classifications` is the full list so we can find joinable dimensions.
    """
    configs_dir = Path(configs_dir)
    with (configs_dir / "model_routing.yaml").open() as f:
        routing_rules: dict = yaml.safe_load(f) or {}

    if target.table_type != "fact":
        return []

    subtypes = _subtypes_present(target)
    measure_names = _measure_names_present(target)
    unclassified_cols = [
        c.name for c in target.columns if c.semantic_subtype == "unclassified_metric"
    ]

    # Identify joinable dimension tables
    dim_tables = [tc for tc in all_classifications if tc.table_type == "dimension"]

    # Derive use-case hints from matched catalog entry
    catalog_use_cases: list[str] = []
    if target.matched_catalog_entry and target.matched_catalog_entry in catalog.datamarts:
        desc = catalog.datamarts[target.matched_catalog_entry].description
        if desc:
            catalog_use_cases.append(desc)

    results: list[ModelConfig] = []

    for model_name, rule in routing_rules.items():
        required: list[str] = rule.get("required", [])
        optional: list[str] = rule.get("optional", [])
        accepts: list[str] = rule.get("accepts", [])
        preferred_measures: list[str] = rule.get("preferred_measures", [])
        rule_use_cases: list[str] = rule.get("use_case_hints", [])

        # Gate: all required subtypes must be present
        if not all(r in subtypes for r in required):
            continue

        # Score
        score = float(len(required))  # 1.0 per required (all satisfied at this point)
        score += sum(0.3 for opt in optional if opt in subtypes)
        score += sum(0.5 for pm in preferred_measures if pm in measure_names)

        # Normalise by theoretical max to get a 0-1 confidence
        max_score = len(required) + len(optional) * 0.3 + len(preferred_measures) * 0.5
        confidence = round(score / max_score, 3) if max_score > 0 else 0.0

        # Infer join keys across dimension tables
        join_keys: dict[str, str] = {}
        matched_dims: list[str] = []
        for dim in dim_tables:
            keys = _infer_join_keys(target, dim)
            if keys:
                join_keys.update(keys)
                matched_dims.append(dim.table_name)

        # Pass-through unclassified metrics if model accepts them
        flagged = unclassified_cols if "unclassified_metric" in accepts else []

        use_cases = catalog_use_cases + rule_use_cases

        results.append(
            ModelConfig(
                model_name=model_name,
                confidence=confidence,
                fact_table=target.table_name,
                dimension_tables=matched_dims,
                join_keys=join_keys,
                use_cases=use_cases,
                flagged_unclassified_columns=flagged,
            )
        )

    results.sort(key=lambda m: m.confidence, reverse=True)
    return results
