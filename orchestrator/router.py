"""
Router — the main orchestration entry point.

Steps per dataset:
  1. List available datasets via connector
  2. Get schema → run ColumnMatcher → get ColumnSpecs
  3. Classify table (fact / dimension) against DatamartCatalog
  4. Assemble ModelConfigs (ranked by routing confidence)
  5. Sample rows → run Profiler → get ColumnProfiles
  6. Collect warnings from the guardrail gate

Returns a list of RouterResult objects, one per dataset.
"""

from __future__ import annotations

from pathlib import Path

from collections import defaultdict

import pandas as pd

from named_model_resolution.catalog_parser import parse_catalog
from named_model_resolution.classifier import classify
from named_model_resolution.column_matcher import ColumnMatcher
from named_model_resolution.config_assembler import assemble
from named_model_resolution.models import DatamartCatalog, RouterResult, TableClassification

from .connectors.base import CatalogConnector
from .profiler import Profiler

# Subtypes that carry routing signal (exclude low-information subtypes)
_ROUTING_SUBTYPES = {"date", "measure", "geography", "segment", "channel", "key", "flag"}


def _coerce_numeric_schema(
    schema: dict[str, str],
    sample: pd.DataFrame,
    min_numeric_frac: float = 0.9,
) -> dict[str, str]:
    """
    Patch schema dtype for object columns where all values are present and
    most values parse as numbers.

    Heuristic: dtype == "object"  AND  null_pct == 0  AND  >= min_numeric_frac
    of values convert successfully with pd.to_numeric(errors="coerce").

    Columns that pass are reported as "float64" so the column matcher treats
    them as numeric rather than string/categorical — preventing string measures
    from being silently classified as dimension_attribute or unknown.
    """
    patched = dict(schema)
    for col, dtype in schema.items():
        if dtype != "object" or col not in sample.columns:
            continue
        series = sample[col]
        if series.isna().any():
            continue  # must have no missing values
        numeric_frac = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_frac >= min_numeric_frac:
            patched[col] = "float64"
    return patched


def _routing_signature(result: RouterResult) -> frozenset[str]:
    """Fingerprint = set of routing-relevant subtypes present in the fact table."""
    return frozenset(
        c.semantic_subtype
        for c in result.classification.columns
        if c.semantic_subtype in _ROUTING_SUBTYPES
    )


def _deduplicate_results(results: list[RouterResult]) -> list[RouterResult]:
    """
    Group fact tables by (routing_signature, top_model).
    Within each group, the table with the most columns is the 'primary'.
    Others are marked is_duplicate_signal=True.
    Dimension and unknown tables are never deduplicated.
    """
    fact_results = [r for r in results if r.classification.table_type == "fact" and r.model_configs]
    non_fact = [r for r in results if r not in fact_results]

    # Group by signature + top model
    groups: dict[tuple, list[RouterResult]] = defaultdict(list)
    for r in fact_results:
        sig = _routing_signature(r)
        top_model = r.model_configs[0].model_name if r.model_configs else ""
        groups[(sig, top_model)].append(r)

    for group in groups.values():
        if len(group) <= 1:
            continue
        # Primary = most columns (most informative routing surface)
        group.sort(key=lambda r: len(r.classification.columns), reverse=True)
        primary = group[0]
        for r in group[1:]:
            r.is_duplicate_signal = True
            r.signal_group_primary = primary.dataset_name

    return non_fact + fact_results


class Router:
    def __init__(
        self,
        connector: CatalogConnector,
        spec_path: str | Path,
        configs_dir: str | Path,
    ) -> None:
        self._connector = connector
        self._catalog: DatamartCatalog = parse_catalog(spec_path)
        self._matcher = ColumnMatcher(configs_dir)
        self._profiler = Profiler(configs_dir)
        self._configs_dir = Path(configs_dir)

    def run(self, datasets: list[str] | None = None, deduplicate: bool = False) -> list[RouterResult]:
        """
        Run the full routing pipeline.

        Args:
            datasets: Optional explicit list of dataset names to process.
                      If None, all datasets from the connector are used.
        """
        if datasets is None:
            datasets = self._connector.list_datasets()

        # ── Pass 1: classify all tables (needed so config assembler can find dims) ──
        all_classifications: list[TableClassification] = []
        warnings_map: dict[str, list[str]] = {}

        for name in datasets:
            warnings: list[str] = []
            schema = self._connector.get_schema(name)

            # Get unique counts from a sample for dimension_attribute heuristic
            try:
                sample = self._connector.sample_rows(name, n=500)
                unique_counts = {col: int(sample[col].nunique()) for col in sample.columns}
                schema = _coerce_numeric_schema(schema, sample)
            except Exception:
                sample = None
                unique_counts = {}

            specs = self._matcher.match_all(schema, unique_counts, warnings)
            classification = classify(name, specs, self._catalog)
            all_classifications.append(classification)
            warnings_map[name] = warnings

        # ── Pass 2: assemble configs + profile (facts only, but return all) ──
        results: list[RouterResult] = []

        for classification in all_classifications:
            name = classification.table_name
            warnings = warnings_map.get(name, [])

            model_configs = assemble(
                target=classification,
                all_classifications=all_classifications,
                catalog=self._catalog,
                configs_dir=self._configs_dir,
            )

            # Profile only if we can sample
            column_profiles = []
            try:
                sample = self._connector.sample_rows(name, n=1000)
                column_profiles = self._profiler.profile(sample, classification.columns)

                # Nullity scrub: completely-null non-date/non-key columns cannot serve as
                # measures/channels. Demote them to "unknown" so the quality gate and runners
                # never see them as key columns. (All-null SQL columns often arrive as float64
                # via pandas NaN inference, which the numeric fallback misclassifies.)
                _prof_lookup = {p.name: p for p in column_profiles}
                for spec in classification.columns:
                    prof = _prof_lookup.get(spec.name)
                    if (prof is not None
                            and prof.null_pct >= 1.0
                            and spec.semantic_subtype not in {"date", "key"}):
                        spec.semantic_subtype = "unknown"
                        spec.confidence = 0.0
            except Exception as exc:
                warnings.append(f"Profiling skipped for '{name}': {exc}")

            results.append(
                RouterResult(
                    dataset_name=name,
                    classification=classification,
                    model_configs=model_configs,
                    column_profiles=column_profiles,
                    warnings=warnings,
                )
            )

        if deduplicate:
            results = _deduplicate_results(results)

        return results
