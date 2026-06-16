"""
OutputBuilder — assembles InsightsPayload from all pipeline parts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from named_model_resolution.models import DatamartCatalog, RouterResult

from ..quality_gate.models import QualityReport
from .models import InsightsPayload


def build(
    router_result: RouterResult,
    quality_report: QualityReport,
    signals_map: dict[str, dict],
    catalog: DatamartCatalog,
) -> InsightsPayload:
    now = datetime.now(timezone.utc).isoformat()

    # ── Metadata ──────────────────────────────────────────────────────────────
    top_model = (
        router_result.model_configs[0].model_name
        if router_result.model_configs
        else None
    )
    top_conf = (
        router_result.model_configs[0].confidence
        if router_result.model_configs
        else None
    )
    all_candidates = [
        {"model": mc.model_name, "confidence": round(mc.confidence, 4)}
        for mc in router_result.model_configs
    ]

    # Star schema info from top model config
    dim_tables: list[str] = []
    join_keys: dict = {}
    if router_result.model_configs:
        mc = router_result.model_configs[0]
        dim_tables = mc.dimension_tables
        join_keys = mc.join_keys

    # Catalog match
    matched_entry = router_result.classification.matched_catalog_entry
    catalog_spec = catalog.datamarts.get(matched_entry) if matched_entry else None

    metadata = {
        "table_type": router_result.classification.table_type,
        "routing": {
            "top_model": top_model,
            "confidence": round(top_conf, 4) if top_conf is not None else None,
            "all_candidates": all_candidates,
        },
        "catalog_match": {
            "entry": matched_entry,
            "score": round(router_result.classification.catalog_match_score, 4),
        },
        "star_schema": {
            "dimension_tables": dim_tables,
            "join_keys": join_keys,
        },
        "duplicate_signal": router_result.is_duplicate_signal,
        "signal_group_primary": router_result.signal_group_primary,
    }

    # ── Column list ───────────────────────────────────────────────────────────
    profile_map = {p.name: p for p in router_result.column_profiles}
    columns = []
    for spec in router_result.classification.columns:
        p = profile_map.get(spec.name)
        col_entry: dict = {
            "name": spec.name,
            "dtype": spec.dtype,
            "subtype": spec.semantic_subtype,
            "confidence": round(spec.confidence, 4),
            "match_source": spec.match_source,
        }
        if p:
            col_entry["profile"] = {
                "null_pct": round(p.null_pct, 4),
                "unique_count": p.unique_count,
                "skewness": round(p.skewness, 4) if p.skewness is not None else None,
                "kurtosis": round(p.kurtosis, 4) if p.kurtosis is not None else None,
                "date_grain": p.date_grain,
                "value_max": p.value_max,
            }
        columns.append(col_entry)

    # ── Quality gate ──────────────────────────────────────────────────────────
    quality_gate: dict = {
        "overall_decision": quality_report.overall_decision,
        "per_model": {},
    }
    for model_name, report in quality_report.per_model.items():
        quality_gate["per_model"][model_name] = {
            "decision": report.decision,
            "skip_reason": report.skip_reason,
            "checks": {
                r.check_name: {
                    "status": r.status,
                    "metric": r.metric,
                    "detail": r.detail,
                }
                for r in report.checks
            },
        }

    # ── Model signals ─────────────────────────────────────────────────────────
    model_signals: dict = {}
    for model_name, result in signals_map.items():
        qr = quality_report.per_model.get(model_name)
        model_signals[model_name] = {
            "ran": result.get("ran", False),
            "quality_decision": qr.decision if qr else "UNKNOWN",
            "candidate_window": result.get("candidate_window"),
            "signals": result.get("signals"),
            "reason": result.get("reason"),
            "note": result.get("note"),
        }
        # Drop None keys to keep JSON tidy
        model_signals[model_name] = {
            k: v for k, v in model_signals[model_name].items() if v is not None
        }

    # ── Transform context ─────────────────────────────────────────────────────
    transform_context = [
        {
            "column": p.name,
            "suggestions": p.suggested_transforms,
        }
        for p in router_result.column_profiles
        if p.suggested_transforms
    ]

    # ── Knowledge base context ────────────────────────────────────────────────
    use_cases: list[str] = []
    if router_result.model_configs:
        mc = router_result.model_configs[0]
        use_cases = mc.use_cases

    knowledge_base_context = {
        "matched_datamart": matched_entry,
        "category": catalog_spec.category if catalog_spec else None,
        "description": catalog_spec.description if catalog_spec else None,
        "use_cases": use_cases,
    }

    return InsightsPayload(
        dataset_name=router_result.dataset_name,
        generated_at=now,
        metadata=metadata,
        columns=columns,
        quality_gate=quality_gate,
        model_signals=model_signals,
        transform_context=transform_context,
        warnings=list(router_result.warnings),
        knowledge_base_context=knowledge_base_context,
    )
