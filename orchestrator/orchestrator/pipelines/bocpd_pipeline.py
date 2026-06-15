"""
BOCPD Pipeline — Bayesian Online Changepoint Detection.

Requires: date column + at least one measure column.
Optional: geography column (filters to a territory if present).

Use cases detected:
  - Weekly persistency trend across territory (Patient Analytics)
  - Weekly TRx/NRx trend across territory (Provider Analytics)
"""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ColumnProfile, RouterResult

from ..connectors.base import CatalogConnector
from .base import ModelPipeline

_PERSISTENCY_MEASURES = {"persistency", "adherence_ratio", "persistence_days"}
_TRX_MEASURES = {"trx", "nrx", "nbrx", "trx_count", "nrx_count"}


class BOCPDPipeline(ModelPipeline):
    def detect_use_cases(self, result: RouterResult) -> list[str]:
        use_cases: list[str] = []
        col_names = {c.name.lower() for c in result.classification.columns}
        expanded = {c.expanded_name.lower() for c in result.classification.columns if c.expanded_name}
        all_names = col_names | expanded

        if _PERSISTENCY_MEASURES & all_names:
            use_cases.append("Weekly persistency trend monitoring across territory (Patient Analytics)")
        if _TRX_MEASURES & all_names:
            use_cases.append("Weekly TRx/NRx trend monitoring across territory (Provider Analytics)")
        if not use_cases:
            use_cases.append("General time-series changepoint detection on available measures")
        return use_cases

    def prep(self, connector: CatalogConnector, result: RouterResult) -> pd.DataFrame:
        df = connector.sample_rows(result.dataset_name, n=10_000)

        # Select date column
        date_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "date"]
        # Select measure + unclassified_metric columns
        measure_cols = [
            c.name
            for c in result.classification.columns
            if c.semantic_subtype in {"measure", "unclassified_metric"}
        ]
        geo_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "geography"]

        keep = date_cols + measure_cols + geo_cols
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()

    def transform(self, df: pd.DataFrame, profiles: list[ColumnProfile]) -> pd.DataFrame:
        profile_map = {p.name: p for p in profiles}

        # Identify date column (first date column)
        date_col = next(
            (c for c in df.columns if profile_map.get(c) and profile_map[c].date_grain is not None),
            None,
        )

        # Apply suggested transforms to measure columns
        for col in df.select_dtypes(include="number").columns:
            profile = profile_map.get(col)
            if profile:
                for suggestion in profile.suggested_transforms:
                    if "log1p" in suggestion:
                        df[col] = df[col].clip(lower=0).pipe(lambda s: s.apply(lambda x: x if pd.isna(x) else __import__("math").log1p(x)))
                    if "clip at 1st/99th" in suggestion:
                        lo, hi = df[col].quantile([0.01, 0.99])
                        df[col] = df[col].clip(lo, hi)

        # Resample to weekly if date grain is daily
        if date_col and date_col in df.columns:
            profile = profile_map.get(date_col)
            if profile and profile.date_grain == "daily":
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    measure_cols = df.select_dtypes(include="number").columns.tolist()
                    df = (
                        df.set_index(date_col)[measure_cols]
                        .resample("W-SAT")
                        .sum()
                        .reset_index()
                    )
                except Exception:
                    pass  # leave as-is if resampling fails

        return df

    def tune(self, df: pd.DataFrame, result: RouterResult) -> dict:
        return {
            "hazard": 1 / 250,
            "truncate": 500,
            "observation_model": "StudentT",
            "suggested_columns": list(df.select_dtypes(include="number").columns),
            "note": "Tune hazard based on expected changepoint frequency; 1/250 ≈ one change per ~5 years weekly.",
        }
