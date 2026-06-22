"""
QualityAssessor — dispatches per-model quality checks.

CHECK_REGISTRY maps check names (from thresholds.yaml) to check functions.
Adding a new check = add function to checks.py + register here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from named_model_resolution.models import RouterResult

from .checks import (
    autocorrelation,
    channel_collinearity,
    date_continuity,
    distribution_shape,
    fill_rate,
    min_row_count,
    segment_balance,
    zero_variance,
)
from .models import ModelQualityReport, QualityCheckResult

CHECK_REGISTRY: dict[str, Callable] = {
    "fill_rate":             fill_rate,
    "zero_variance":         zero_variance,
    "date_continuity":       date_continuity,
    "channel_collinearity":  channel_collinearity,
    "segment_balance":       segment_balance,
    "autocorrelation":       autocorrelation,
    "min_row_count":         min_row_count,
    "distribution_shape":    distribution_shape,
}


def _aggregate_decision(checks: list[QualityCheckResult]) -> str:
    statuses = {r.status for r in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


class QualityAssessor:
    def __init__(self, thresholds_path: str | Path) -> None:
        with open(thresholds_path) as f:
            self._thresholds: dict = yaml.safe_load(f) or {}

    def assess(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_name: str,
    ) -> ModelQualityReport:
        """
        Run all checks registered for model_name and return a ModelQualityReport.
        """
        global_params = dict(self._thresholds.get("global", {}))
        model_params = dict(self._thresholds.get(model_name, {}))
        params = {**global_params, **model_params}   # model overrides global

        check_names: list[str] = model_params.get("checks", [])
        if not check_names:
            # No checks registered for this model — PASS with a note
            return ModelQualityReport(
                model_name=model_name,
                decision="PASS",
                checks=[QualityCheckResult(
                    check_name="no_checks",
                    status="PASS",
                    detail=f"no quality checks configured for model '{model_name}'",
                    metric=None,
                )],
                skip_reason=None,
            )

        results: list[QualityCheckResult] = []
        for name in check_names:
            fn = CHECK_REGISTRY.get(name)
            if fn is None:
                results.append(QualityCheckResult(
                    check_name=name,
                    status="WARN",
                    detail=f"check '{name}' not registered in CHECK_REGISTRY",
                    metric=None,
                ))
                continue
            try:
                result = fn(
                    df,
                    router_result.classification.columns,
                    router_result.column_profiles,
                    params,
                )
            except Exception as exc:
                result = QualityCheckResult(
                    check_name=name,
                    status="WARN",
                    detail=f"check '{name}' raised an exception: {exc}",
                    metric=None,
                )
            results.append(result)

        decision = _aggregate_decision(results)
        skip_reason = None
        if decision == "FAIL":
            skip_reason = next(
                (r.detail for r in results if r.status == "FAIL"), None
            )

        return ModelQualityReport(
            model_name=model_name,
            decision=decision,
            checks=results,
            skip_reason=skip_reason,
        )
