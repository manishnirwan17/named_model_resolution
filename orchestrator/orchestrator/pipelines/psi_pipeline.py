"""
PSI Pipeline — Population Stability Index / Segment Drift Detection.

Requires: date column + measure column + segment column.

Use cases detected:
  - Patient segment distribution drift over time
  - LOT / treatment status progression drift
"""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ColumnProfile, RouterResult

from ..connectors.base import CatalogConnector
from .base import ModelPipeline


class PSIPipeline(ModelPipeline):
    def detect_use_cases(self, result: RouterResult) -> list[str]:
        col_names = {c.name.lower() for c in result.classification.columns}
        use_cases: list[str] = []
        if "line_of_therapy" in col_names or "lot" in col_names:
            use_cases.append("LOT / drug holiday progression drift (Patient Analytics)")
        if "patient_status" in col_names or "treatment_status" in col_names:
            use_cases.append("Patient segment distribution drift over time")
        if "payer_channel" in col_names:
            use_cases.append("Payer channel mix drift")
        if not use_cases:
            use_cases.append("General population stability monitoring across segments")
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
        keep = [c for c in date_cols + measure_cols + segment_cols if c in df.columns]
        return df[keep].copy()

    def transform(self, df: pd.DataFrame, profiles: list[ColumnProfile]) -> pd.DataFrame:
        # PSI works on distributions — apply null imputation if flagged
        profile_map = {p.name: p for p in profiles}
        for col in df.select_dtypes(include="number").columns:
            profile = profile_map.get(col)
            if profile:
                for suggestion in profile.suggested_transforms:
                    if "forward-fill or median" in suggestion:
                        df[col] = df[col].fillna(df[col].median())
        return df

    def tune(self, df: pd.DataFrame, result: RouterResult) -> dict:
        date_cols = [c for c in df.columns if "date" in c.lower() or "period" in c.lower()]
        return {
            "reference_period": "first quartile of date range",
            "comparison_period": "last quartile of date range",
            "date_column": date_cols[0] if date_cols else None,
            "n_bins": 10,
            "psi_threshold": 0.2,
            "note": "PSI < 0.1 = stable; 0.1–0.2 = moderate shift; > 0.2 = significant drift.",
        }
