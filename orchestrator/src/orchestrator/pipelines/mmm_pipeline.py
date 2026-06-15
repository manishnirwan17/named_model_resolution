"""
MMM Pipeline — Marketing Mix Modeling.

Requires: date column + measure column.
Optional: geography, segment/channel columns.

Use cases detected:
  - Decompose sales/TRx drivers across channels
  - Estimate ROI of promotional spend
"""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ColumnProfile, RouterResult

from ..connectors.base import CatalogConnector
from .base import ModelPipeline

_CHANNEL_KEYWORDS = {"channel", "payer_channel", "chnl", "payer", "fulfillment", "rejection"}
_SPEND_KEYWORDS = {"spend", "cost", "budget", "investment", "detailing", "sample"}


class MMMPipeline(ModelPipeline):
    def detect_use_cases(self, result: RouterResult) -> list[str]:
        col_names = {c.name.lower() for c in result.classification.columns}
        use_cases: list[str] = []
        if col_names & _CHANNEL_KEYWORDS:
            use_cases.append("Decompose TRx/sales drivers across payer channels")
        if col_names & _SPEND_KEYWORDS:
            use_cases.append("Estimate ROI and contribution of promotional spend")
        if not use_cases:
            use_cases.append("Marketing mix decomposition on available measures")
        return use_cases

    def prep(self, connector: CatalogConnector, result: RouterResult) -> pd.DataFrame:
        df = connector.sample_rows(result.dataset_name, n=10_000)
        date_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "date"]
        measure_cols = [
            c.name
            for c in result.classification.columns
            if c.semantic_subtype in {"measure", "unclassified_metric"}
        ]
        segment_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "segment"]
        geo_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "geography"]
        keep = [c for c in date_cols + measure_cols + segment_cols + geo_cols if c in df.columns]
        return df[keep].copy()

    def transform(self, df: pd.DataFrame, profiles: list[ColumnProfile]) -> pd.DataFrame:
        profile_map = {p.name: p for p in profiles}
        for col in df.select_dtypes(include="number").columns:
            profile = profile_map.get(col)
            if profile:
                for suggestion in profile.suggested_transforms:
                    if "log1p" in suggestion:
                        df[col] = df[col].clip(lower=0).apply(
                            lambda x: x if pd.isna(x) else __import__("math").log1p(x)
                        )
                    if "clip at 1st/99th" in suggestion:
                        lo, hi = df[col].quantile([0.01, 0.99])
                        df[col] = df[col].clip(lo, hi)
        return df

    def tune(self, df: pd.DataFrame, result: RouterResult) -> dict:
        measure_cols = list(df.select_dtypes(include="number").columns)
        return {
            "kpi": measure_cols[0] if measure_cols else None,
            "channels": [c for c in measure_cols[1:] if any(kw in c.lower() for kw in _CHANNEL_KEYWORDS | _SPEND_KEYWORDS)],
            "baseline_formula": "intercept + seasonality",
            "note": "Add adstock/lag transforms for spend columns before fitting.",
        }
