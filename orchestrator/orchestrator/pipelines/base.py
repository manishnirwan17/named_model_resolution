"""
ModelPipeline abstract base class.

Each model (BOCPD, MMM, PSI, ARIMA, ...) is a pipeline with four stages:
  1. detect_use_cases  → what specific use cases apply to this dataset
  2. prep              → extract / filter required columns
  3. transform         → apply recommended statistical transforms
  4. tune              → produce model hyperparameter config dict

Adding a new model = create a new file implementing ModelPipeline + add a
routing rule entry in model_routing.yaml.  No changes to Router needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from named_model_resolution.models import ColumnProfile, RouterResult

    from ..connectors.base import CatalogConnector


class ModelPipeline(ABC):
    @abstractmethod
    def detect_use_cases(self, result: "RouterResult") -> list[str]:
        """Return applicable use-case descriptions for this dataset."""
        ...

    @abstractmethod
    def prep(
        self,
        connector: "CatalogConnector",
        result: "RouterResult",
    ) -> pd.DataFrame:
        """Load and filter data to the columns required by this model."""
        ...

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        profiles: list["ColumnProfile"],
    ) -> pd.DataFrame:
        """Apply profile-driven and model-specific transformations."""
        ...

    @abstractmethod
    def tune(self, df: pd.DataFrame, result: "RouterResult") -> dict:
        """Return model hyperparameter / config suggestions as a dict."""
        ...

    def run(
        self,
        connector: "CatalogConnector",
        result: "RouterResult",
    ) -> dict:
        """Orchestrate the full pipeline and return a summary dict."""
        use_cases = self.detect_use_cases(result)
        df = self.prep(connector, result)
        df = self.transform(df, result.column_profiles)
        params = self.tune(df, result)
        return {
            "dataset": result.dataset_name,
            "model": self.__class__.__name__,
            "use_cases": use_cases,
            "model_params": params,
            "row_count": len(df),
            "columns_used": list(df.columns),
        }
