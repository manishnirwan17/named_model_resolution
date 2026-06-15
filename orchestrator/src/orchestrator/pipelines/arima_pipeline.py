"""
ARIMA Pipeline — Seasonal trend modeling (ARIMA / SARIMA / Prophet).

Fallback time-series model for any fact table with a date + continuous measure
when there is no clear changepoint hypothesis for BOCPD.

Use cases detected:
  - Forecast weekly TRx / market share over next N periods
  - Decompose seasonality in patient counts or market size
"""

from __future__ import annotations

import pandas as pd

from named_model_resolution.models import ColumnProfile, RouterResult

from ..connectors.base import CatalogConnector
from .base import ModelPipeline


class ARIMAPipeline(ModelPipeline):
    def detect_use_cases(self, result: RouterResult) -> list[str]:
        col_names = {c.name.lower() for c in result.classification.columns}
        use_cases: list[str] = []
        if any(kw in col_names for kw in {"trx", "nrx", "nbrx", "trx_count", "nrx_count"}):
            use_cases.append("Forecast weekly TRx/NRx over next N periods")
        if any(kw in col_names for kw in {"market_share", "market_size", "projected_market_size"}):
            use_cases.append("Decompose and forecast seasonality in market size / share")
        if any(kw in col_names for kw in {"patient_count", "total_patients"}):
            use_cases.append("Patient count trend forecasting with seasonal decomposition")
        if not use_cases:
            use_cases.append("General seasonal trend modeling on available time-series measures")
        return use_cases

    def prep(self, connector: CatalogConnector, result: RouterResult) -> pd.DataFrame:
        df = connector.sample_rows(result.dataset_name, n=10_000)
        date_cols = [c.name for c in result.classification.columns if c.semantic_subtype == "date"]
        measure_cols = [
            c.name
            for c in result.classification.columns
            if c.semantic_subtype in {"measure", "unclassified_metric"}
        ]
        keep = [c for c in date_cols + measure_cols if c in df.columns]
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
                    if "forward-fill" in suggestion:
                        df[col] = df[col].ffill()
        return df

    def tune(self, df: pd.DataFrame, result: RouterResult) -> dict:
        measure_cols = list(df.select_dtypes(include="number").columns)
        return {
            "suggested_target": measure_cols[0] if measure_cols else None,
            "order_hints": {"p": "check PACF", "d": 1, "q": "check ACF"},
            "seasonal_period": 52,  # weekly data → 52-week seasonality
            "framework_options": ["statsmodels SARIMA", "Prophet", "NeuralProphet"],
            "note": "Run ADF test on target series first; d=1 if non-stationary.",
        }
