"""
insights_runner.pipeline — entry point.

Usage:
    from insights_runner.pipeline import run
    payload = run(connector, router_result, catalog, configs_dir)
    print(payload.to_json())
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from named_model_resolution.models import DatamartCatalog, RouterResult
from orchestrator.connectors.base import CatalogConnector

from .output.builder import build
from .output.models import InsightsPayload
from .quality_gate.assessor import QualityAssessor
from .quality_gate.models import QualityReport
from .runners import RUNNER_REGISTRY

_DEFAULT_THRESHOLDS = Path(__file__).parent / "quality_gate" / "thresholds.yaml"
_DEFAULT_SAMPLE_N = 5000


def _aggregate_decision(per_model: dict) -> str:
    decisions = {r.decision for r in per_model.values()}
    if "PASS" in decisions:
        return "PASS"   # at least one model can run
    if "WARN" in decisions:
        return "WARN"
    return "FAIL"


def _read_window_config(thresholds_path: Path) -> dict[str, int]:
    """
    Read candidate_window_years per model from thresholds.yaml.
    Returns {model_name: years} for models that have the key set.
    """
    try:
        with open(thresholds_path) as f:
            raw = yaml.safe_load(f)
        return {
            model: int(cfg["candidate_window_years"])
            for model, cfg in raw.items()
            if isinstance(cfg, dict) and "candidate_window_years" in cfg
        }
    except Exception:
        return {}


def _window_df(
    df: pd.DataFrame,
    router_result: RouterResult,
    window_years: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Slice df to rows with date >= (max_date - window_years).

    Returns:
        (sliced_df, window_info)  — window_info is embedded in the payload
        so the LLM knows exactly which date range was analysed.
        Falls back to (full df, {}) if no date column or slicing fails.
    """
    date_col = next(
        (s.name for s in router_result.classification.columns
         if s.semantic_subtype == "date"),
        None,
    )
    if date_col is None or date_col not in df.columns or window_years <= 0:
        return df, {}

    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        max_date = dates.max()
        cutoff = max_date - pd.DateOffset(years=window_years)
        mask = dates >= cutoff
        sliced = df[mask].copy()
        if len(sliced) == 0:
            return df, {}
        return sliced, {
            "years": window_years,
            "cutoff_date": str(cutoff.date()),
            "max_date": str(max_date.date()),
            "n_rows": len(sliced),
        }
    except Exception:
        return df, {}


def run(
    connector: CatalogConnector,
    router_result: RouterResult,
    catalog: DatamartCatalog,
    configs_dir: str | Path,
    models_to_run: list[str] | None = None,
    thresholds_path: str | Path | None = None,
    sample_n: int = _DEFAULT_SAMPLE_N,
) -> InsightsPayload:
    """
    Full insights pipeline for a single RouterResult.

    Args:
        connector:       Platform-agnostic data connector (same as router used).
        router_result:   Output from orchestrator.Router.run() for one dataset.
        catalog:         DatamartCatalog (from parse_catalog).
        configs_dir:     Path to pharma_knowledge_base/configs/ (for future use).
        models_to_run:   Restrict to a subset of routed models (None = all).
        thresholds_path: Override path to thresholds.yaml.
        sample_n:        Number of rows to sample from the dataset.

    Returns:
        InsightsPayload (call .to_json() for the LLM-ingestible string).
    """
    configs_dir = Path(configs_dir)
    thresholds_path = Path(thresholds_path) if thresholds_path else _DEFAULT_THRESHOLDS

    # ── Sample data ───────────────────────────────────────────────────────────
    try:
        df: pd.DataFrame = connector.sample_rows(
            router_result.dataset_name, n=sample_n
        )
    except Exception as exc:
        from datetime import datetime, timezone
        return InsightsPayload(
            dataset_name=router_result.dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
            columns=[],
            quality_gate={"overall_decision": "FAIL", "per_model": {}},
            model_signals={},
            transform_context=[],
            warnings=list(router_result.warnings) + [f"sampling failed: {exc}"],
            knowledge_base_context={},
        )

    # ── Candidate window config (per model, from thresholds.yaml) ─────────────
    window_config = _read_window_config(thresholds_path)

    assessor = QualityAssessor(thresholds_path)

    model_configs = router_result.model_configs
    if models_to_run:
        model_configs = [mc for mc in model_configs if mc.model_name in models_to_run]

    quality_per_model: dict = {}
    signals_map: dict = {}

    for mc in model_configs:
        # ── Candidate generation: slice to model-specific date window ─────────
        window_years = window_config.get(mc.model_name, 0)
        if window_years > 0:
            df_model, window_info = _window_df(df, router_result, window_years)
        else:
            df_model, window_info = df, {}

        # ── Quality gate (runs on the windowed slice) ─────────────────────────
        quality = assessor.assess(df_model, router_result, mc.model_name)
        quality_per_model[mc.model_name] = quality

        if quality.decision == "FAIL":
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": quality.skip_reason or "quality gate FAIL",
                "candidate_window": window_info or None,
            }
            continue

        # ── Run model (also on the windowed slice) ────────────────────────────
        runner_cls = RUNNER_REGISTRY.get(mc.model_name)
        if runner_cls is None:
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": f"no runner registered for model '{mc.model_name}'",
                "candidate_window": window_info or None,
            }
            continue

        try:
            result = runner_cls().run(df_model, router_result, mc)
            if window_info:
                result["candidate_window"] = window_info
            signals_map[mc.model_name] = result
        except Exception as exc:
            signals_map[mc.model_name] = {
                "ran": False,
                "reason": f"runner raised an exception: {exc}",
                "candidate_window": window_info or None,
            }

    quality_report = QualityReport(
        overall_decision=_aggregate_decision(quality_per_model) if quality_per_model else "PASS",
        per_model=quality_per_model,
    )

    return build(router_result, quality_report, signals_map, catalog)
