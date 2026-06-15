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

from named_model_resolution.catalog_parser import parse_catalog
from named_model_resolution.classifier import classify
from named_model_resolution.column_matcher import ColumnMatcher
from named_model_resolution.config_assembler import assemble
from named_model_resolution.models import DatamartCatalog, RouterResult, TableClassification

from .connectors.base import CatalogConnector
from .profiler import Profiler


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

    def run(self, datasets: list[str] | None = None) -> list[RouterResult]:
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

        return results
