"""
Profiler — samples N rows from a dataset and computes per-column statistics.

For measure / unclassified_metric columns:
  - skewness and kurtosis via scipy.stats
  - null percentage
  - value range (min/max)

For date columns:
  - infer grain from median gap between consecutive sorted dates

Transform rule evaluation is config-driven (transform_rules.yaml).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from named_model_resolution.models import ColumnProfile, ColumnSpec


def _infer_date_grain(series: pd.Series) -> str | None:
    """Infer time grain from sorted unique dates: daily / weekly / monthly / quarterly / None.

    Uses unique dates only — deduplicating before computing gaps prevents HCP-level
    data (many rows per month with the same date) from producing a median gap of 0
    and being wrongly classified as daily.
    """
    try:
        s = pd.to_datetime(series.dropna()).drop_duplicates().sort_values()
        if len(s) < 2:
            return None
        gaps = s.diff().dropna().dt.days
        median_gap = gaps.median()
        if median_gap <= 2:
            return "daily"
        if 5 <= median_gap <= 9:
            return "weekly"
        if 25 <= median_gap <= 35:
            return "monthly"
        if 80 <= median_gap <= 100:
            return "quarterly"
        return None
    except Exception:
        return None


class Profiler:
    def __init__(self, configs_dir: str | Path) -> None:
        configs_dir = Path(configs_dir)
        with (configs_dir / "transform_rules.yaml").open() as f:
            self._rules: dict = yaml.safe_load(f) or {}

    def profile(
        self,
        df: pd.DataFrame,
        column_specs: list[ColumnSpec],
    ) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        spec_map = {s.name: s for s in column_specs}

        for col in df.columns:
            spec = spec_map.get(col)
            subtype = spec.semantic_subtype if spec else "unknown"
            series = df[col]
            null_pct = float(series.isna().mean())
            unique_count = int(series.nunique(dropna=True))

            skewness: float | None = None
            kurtosis: float | None = None
            value_max: float | None = None
            date_grain: str | None = None
            mean: float | None = None
            median: float | None = None
            std: float | None = None
            outlier_rate: float | None = None

            if subtype in {"measure", "unclassified_metric"}:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 1:
                    from scipy import stats as scipy_stats  # lazy import

                    skewness = float(scipy_stats.skew(numeric))
                    kurtosis = float(scipy_stats.kurtosis(numeric))
                    value_max = float(numeric.max())
                    mean = float(numeric.mean())
                    median = float(numeric.median())
                    std = float(numeric.std())
                    q1 = float(numeric.quantile(0.25))
                    q3 = float(numeric.quantile(0.75))
                    iqr = q3 - q1
                    outlier_rate = float(
                        ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).mean()
                    )

            elif subtype == "date":
                date_grain = _infer_date_grain(series)

            suggested = self._evaluate_rules(
                col_name=col,
                subtype=subtype,
                null_pct=null_pct,
                skewness=skewness,
                kurtosis=kurtosis,
                value_max=value_max,
                unique_count=unique_count,
                dtype=str(series.dtype),
                date_grain=date_grain,
            )

            profiles.append(
                ColumnProfile(
                    name=col,
                    dtype=str(series.dtype),
                    null_pct=null_pct,
                    skewness=skewness,
                    kurtosis=kurtosis,
                    value_max=value_max,
                    unique_count=unique_count,
                    date_grain=date_grain,
                    suggested_transforms=suggested,
                    mean=mean,
                    median=median,
                    std=std,
                    outlier_rate=outlier_rate,
                )
            )

        return profiles

    def _evaluate_rules(
        self,
        col_name: str,
        subtype: str,
        null_pct: float,
        skewness: float | None,
        kurtosis: float | None,
        value_max: float | None,
        unique_count: int,
        dtype: str,
        date_grain: str | None,
    ) -> list[str]:
        suggestions: list[str] = []

        for rule_name, rule in self._rules.items():
            trigger: dict = rule.get("trigger", {})
            applies_to: list[str] = rule.get("applies_to", [])
            suggestion: str = rule.get("suggestion", "")

            # Check applies_to filter
            if applies_to and subtype not in applies_to:
                continue

            # Evaluate trigger conditions
            matched = True

            if "skewness_gt" in trigger:
                if skewness is None or skewness <= trigger["skewness_gt"]:
                    matched = False

            if "kurtosis_gt" in trigger:
                if kurtosis is None or kurtosis <= trigger["kurtosis_gt"]:
                    matched = False

            if "null_pct_gt" in trigger:
                if null_pct <= trigger["null_pct_gt"]:
                    matched = False

            if "date_grain" in trigger:
                if date_grain != trigger["date_grain"]:
                    matched = False

            if "unique_count_gt" in trigger:
                if unique_count <= trigger["unique_count_gt"]:
                    matched = False

            if "dtype" in trigger:
                if trigger["dtype"] not in dtype.lower():
                    matched = False

            if "value_max_gt" in trigger:
                if value_max is None or value_max <= trigger["value_max_gt"]:
                    matched = False

            # Optional column-name filter for ratio_bound_check etc.
            if "column_name_contains" in rule:
                col_lower = col_name.lower()
                if not any(kw in col_lower for kw in rule["column_name_contains"]):
                    matched = False

            if matched:
                suggestions.append(f"[{rule_name}] {suggestion}")

        return suggestions
