"""
MMMRunner — bridges RouterResult to insights_generation MMM pipeline.

Runs MMM on ALL viable, non-redundant measure columns (target variables).
Columns are ranked by the measure selector, then correlation-deduplicated
(threshold = _CORR_THRESHOLD_MMM = 0.90).  Results are stored per-column
under signals["by_measure"][col_name].

Translation flow per target column:
  1. Build DatasetConfig from RouterResult column subtypes
  2. Aggregate HCP-level -> market-level (if key column present)
  3. Add log-target, week_idx, Fourier terms, split column
  4. Call transform_channels(mkt, dataset_config=ds)
  5. Call build_model_matrix(mkt, dataset_config=ds)
  6. Optionally fit PyMC model (build_pymc_model + pm.sample)

mmm_data_prep.transform_channels / build_model_matrix accept an explicit
dataset_config parameter (refactored from global DATASET) so no global mutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from named_model_resolution.models import ModelConfig, RouterResult

from .base import ModelRunner

_TRAIN_WEEKS    = 180
_FOURIER_K      = 2
_FOURIER_PERIOD = 52

# PyMC sampling defaults (conservative for runner context)
_MCMC_DRAWS          = 500
_MCMC_TUNE           = 500
_MCMC_CHAINS         = 2
_MCMC_TARGET_ACCEPT  = 0.90
_MCMC_SEED           = 42

_CORR_THRESHOLD_MMM  = 0.90  # drop measure columns more correlated than this


# ---------------------------------------------------------------------------
# DatasetConfig builder
# ---------------------------------------------------------------------------

def _build_dataset_config(
    router_result: RouterResult,
    target_col: str | None = None,
) -> tuple:
    """
    Construct a DatasetConfig from RouterResult without reading any JSON.

    `target_col` — if provided, use this column as the MMM target (dependent
    variable).  If None, falls back to select_measure_column() (single-best
    selection) for backward compatibility.

    Channel field vs broadcast classification:
      Columns containing broadcast hints -> broadcast/mean.
      Everything else -> field/sum.

    Returns (DatasetConfig | None, measure_note | None).
    """
    try:
        from pipeline.dataset_config import ChannelSpec, DatasetConfig
    except ImportError:
        return None, None

    specs    = router_result.classification.columns
    profiles = {p.name: p for p in router_result.column_profiles}

    _broadcast_hints = {
        "impression", "grp", "tv", "display", "programmatic",
        "social", "digital", "banner",
    }

    # Determine target column
    measure_note: str | None = None
    if target_col is None:
        from ._measure_selector import select_measure_column
        target_col, measure_note = select_measure_column(specs, profiles)

    from ._measure_selector import select_date_column
    week_col, _ = select_date_column(specs, profiles)
    hcp_id_col = next((s.name for s in specs if s.semantic_subtype == "key"), None)

    if not target_col or not week_col:
        return None, None

    channels = []
    for s in specs:
        if s.semantic_subtype != "channel":
            continue
        name_lower = s.name.lower()
        is_broadcast = any(h in name_lower for h in _broadcast_hints)
        is_competitor = any(
            h in name_lower for h in ("competitor", "compet", "rival", "generic")
        )
        if is_competitor:
            group, agg, decay = "competitor", "mean", 0.0
        elif is_broadcast:
            group, agg, decay = "broadcast", "mean", 0.60
        else:
            group, agg, decay = "field", "sum", 0.30
        channels.append(ChannelSpec(name=s.name, group=group, agg=agg, decay=decay))

    return DatasetConfig(
        target_col=target_col,
        week_col=week_col,
        hcp_id_col=hcp_id_col,
        channels=channels,
        fourier_k=_FOURIER_K,
        fourier_period=_FOURIER_PERIOD,
        train_weeks=_TRAIN_WEEKS,
    ), measure_note


# ---------------------------------------------------------------------------
# Market-level aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_to_market(df: pd.DataFrame, ds) -> pd.DataFrame:
    """Collapse HCP x week -> weekly market-level aggregate."""
    agg_dict = {}
    sum_cols  = ds.sum_channels + [ds.target_col]
    mean_cols = ds.mean_channels

    for c in sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"
    for c in mean_cols:
        if c in df.columns:
            agg_dict[c] = "mean"
    if ds.lifecycle_col and ds.lifecycle_col in df.columns:
        agg_dict[ds.lifecycle_col] = "first"

    if not agg_dict:
        return df.copy()

    group_cols = [ds.week_col]
    if ds.product_col and ds.product_col in df.columns:
        group_cols.append(ds.product_col)

    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def _add_features(mkt: pd.DataFrame, ds) -> pd.DataFrame:
    """Add log-target, lifecycle numeric, week_idx, Fourier terms."""
    mkt = mkt.sort_values(ds.week_col).reset_index(drop=True)

    mkt[ds.target_log_col] = np.log(mkt[ds.target_col].clip(lower=1e-6))

    if ds.lifecycle_col and ds.lifecycle_col in mkt.columns:
        stage_map = {"pre_launch": 0, "launch": 1, "growth": 2, "maturity": 3, "decline": 4}
        mkt["lc_num"] = mkt[ds.lifecycle_col].map(stage_map).fillna(2).astype(int)

    mkt["week_idx"] = np.arange(len(mkt))

    t = mkt["week_idx"].values
    for k in range(1, ds.fourier_k + 1):
        mkt[f"sin_{k}"] = np.sin(2 * np.pi * k * t / ds.fourier_period)
        mkt[f"cos_{k}"] = np.cos(2 * np.pi * k * t / ds.fourier_period)

    for col in ds.broadcast_channels:
        if col in mkt.columns:
            mkt[f"{col}_log"] = np.log1p(mkt[col])

    return mkt


# ---------------------------------------------------------------------------
# Single-target MMM pipeline
# ---------------------------------------------------------------------------

def _run_single_mmm(df: pd.DataFrame, ds, column_specs=None) -> dict:
    """
    Run the full MMM pipeline for one DatasetConfig (one target column).

    Returns a signals dict on success, or {"ran": False, "reason": str} on
    failure.  The returned dict is stored under signals["by_measure"][target_col].
    """
    from ._data_normalizer import normalize_grain, normalize_to_series, parse_dates_flexible

    measure_note: str | None = None

    # ── Validate required columns in df ──────────────────────────────────────
    if ds.week_col not in df.columns:
        return {"ran": False, "reason": f"week column '{ds.week_col}' not in df"}
    if ds.target_col not in df.columns:
        return {"ran": False, "reason": f"target column '{ds.target_col}' not in df"}

    try:
        df = df.copy()
        df[ds.week_col] = parse_dates_flexible(df[ds.week_col])
    except Exception as exc:
        return {"ran": False, "reason": f"date conversion failed: {exc}"}

    # ── Aggregate to market level if HCP-level ────────────────────────────────
    if ds.hcp_id_col and ds.hcp_id_col in df.columns:
        mkt = _aggregate_to_market(df, ds)
    else:
        mkt = df.copy()

    # ── Normalize to single time series (territory-level rollup) ─────────────
    _all_measure_cols = [ds.target_col] + [s.name for s in ds.channels]
    mkt, agg_note = normalize_to_series(mkt, ds.week_col, _all_measure_cols, column_specs)
    if agg_note:
        measure_note = (
            (measure_note + " " + agg_note).strip() if measure_note else agg_note
        )

    # ── Grain normalization (weekly -> monthly if configured) ─────────────────
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        _raw = _yaml.safe_load(
            (_Path(__file__).parent.parent / "quality_gate" / "thresholds.yaml").read_text()
        ) or {}
        _target_grain = {**_raw.get("global", {}), **_raw.get("MMM", {})}.get(
            "target_grain", "auto"
        )
    except Exception:
        _target_grain = "auto"

    if _target_grain == "monthly":
        mkt, grain_note = normalize_grain(mkt, ds.week_col, _all_measure_cols, "monthly")
        if grain_note:
            measure_note = (
                (measure_note + " " + grain_note).strip() if measure_note else grain_note
            )

    # ── Add features ──────────────────────────────────────────────────────────
    try:
        mkt = _add_features(mkt, ds)
    except Exception as exc:
        return {"ran": False, "reason": f"feature engineering failed: {exc}"}

    mkt["split"] = "train"
    mkt.loc[mkt.index >= ds.train_weeks, "split"] = "test"

    # ── Transform channels + build model matrix ───────────────────────────────
    try:
        from pipeline.mmm_data_prep import build_model_matrix, transform_channels
    except ImportError:
        return {
            "ran": False,
            "reason": (
                "pipeline.mmm_data_prep not importable; "
                "pip install -e insights_generation/"
            ),
        }

    try:
        mkt, channel_meta = transform_channels(mkt, dataset_config=ds)
        X, y, feature_cols, scaler = build_model_matrix(mkt, dataset_config=ds)
    except Exception as exc:
        return {"ran": False, "reason": f"MMM data prep failed: {exc}"}

    train_mask = (mkt["split"] == "train").values
    X_train, y_train = X[train_mask], y[train_mask]

    # ── Fit PyMC model (optional — skipped if PyMC not installed) ─────────────
    try:
        import arviz as az
        import pymc as pm
        from pipeline.mmm_fit import build_pymc_model
    except ImportError:
        pymc_note = "PyMC not installed -- data prep completed but model not sampled"
        combined_note = (
            f"{pymc_note}. {measure_note}" if measure_note else pymc_note
        )
        result = {
            "ran": True,
            "channel_meta":          channel_meta,
            "feature_cols":          feature_cols,
            "n_train_rows":          int(train_mask.sum()),
            "model_fit":             None,
            "channel_contributions": None,
        }
        if combined_note:
            result["note"] = combined_note
        return result

    try:
        mmm_model, ch_idx, comp_idx, ctrl_idx = build_pymc_model(
            X_train, y_train, feature_cols, scaler,
            dataset_config=ds,
            alpha_prior_mu=float(y_train.mean()),
        )

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            with mmm_model:
                trace = pm.sample(
                    draws=_MCMC_DRAWS,
                    tune=_MCMC_TUNE,
                    chains=_MCMC_CHAINS,
                    target_accept=_MCMC_TARGET_ACCEPT,
                    random_seed=_MCMC_SEED,
                    return_inferencedata=True,
                    progressbar=False,
                )
    except Exception as exc:
        return {"ran": False, "reason": f"PyMC sampling failed: {exc}"}

    # ── Diagnostics ───────────────────────────────────────────────────────────
    try:
        summary   = az.summary(
            trace, var_names=["alpha", "beta_ch", "sigma"], round_to=4
        )
        max_rhat  = float(summary["r_hat"].max())
        min_ess   = float(summary["ess_bulk"].min())
        divs      = int(trace.sample_stats["diverging"].sum().item())
    except Exception:
        max_rhat = min_ess = divs = None

    # ── In-sample MAPE ────────────────────────────────────────────────────────
    try:
        with mmm_model:
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
        y_hat = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
        mape  = float(
            np.mean(np.abs(np.exp(y_hat) - np.exp(y_train)) / np.exp(y_train)) * 100
        )
    except Exception:
        mape = None

    # ── Contribution decomposition ────────────────────────────────────────────
    contributions = []
    try:
        beta_ch_mean   = trace.posterior["beta_ch"].mean(dim=["chain", "draw"]).values
        beta_ctrl_mean = trace.posterior["beta_ctrl"].mean(dim=["chain", "draw"]).values
        alpha_mean     = float(trace.posterior["alpha"].mean())

        contrib_df = pd.DataFrame({"week": mkt[ds.week_col].values})
        contrib_df["baseline"] = alpha_mean
        for i, col_idx in enumerate(ch_idx):
            col_name = feature_cols[col_idx]
            contrib_df[f"contrib_{col_name}"] = beta_ch_mean[i] * X[:, col_idx]
        for i, col_idx in enumerate(ctrl_idx):
            col_name = feature_cols[col_idx]
            contrib_df[f"contrib_{col_name}"] = beta_ctrl_mean[i] * X[:, col_idx]

        contrib_df["y_actual"] = y
        contrib_df["split"]    = mkt["split"].values
        contributions = contrib_df.to_dict("records")
        for rec in contributions:
            for k, v in rec.items():
                if isinstance(v, pd.Timestamp):
                    rec[k] = str(v)
    except Exception:
        contributions = []

    result = {
        "ran":                   True,
        "channel_meta":          channel_meta,
        "feature_cols":          feature_cols,
        "n_train_rows":          int(train_mask.sum()),
        "model_fit": {
            "in_sample_mape": round(mape, 4)     if mape     is not None else None,
            "rhat_max":       round(max_rhat, 4) if max_rhat is not None else None,
            "ess_min":        round(min_ess, 1)  if min_ess  is not None else None,
            "divergences":    divs,
        },
        "channel_contributions": contributions,
    }
    if measure_note:
        result["note"] = measure_note
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MMMRunner(ModelRunner):
    def run(
        self,
        df: pd.DataFrame,
        router_result: RouterResult,
        model_config: ModelConfig,
    ) -> dict:
        specs    = router_result.classification.columns
        profiles = {p.name: p for p in router_result.column_profiles}

        # ── Ranked + deduplicated target columns ──────────────────────────────
        from ._measure_selector import dedup_by_correlation, select_measure_columns

        ranked    = select_measure_columns(specs, profiles)
        col_names = [c for c, _ in ranked]
        col_names = dedup_by_correlation(df, col_names, threshold=_CORR_THRESHOLD_MMM)

        if not col_names:
            return {"ran": False, "reason": "no viable measure columns for MMM"}

        # ── Loop over target columns ──────────────────────────────────────────
        by_measure: dict[str, dict] = {}
        notes: list[str] = []

        for target_col in col_names:
            ds, measure_note = _build_dataset_config(
                router_result, target_col=target_col
            )
            if measure_note:
                notes.append(measure_note)

            if ds is None:
                by_measure[target_col] = {
                    "ran": False,
                    "reason": (
                        "could not construct DatasetConfig -- "
                        "pipeline.dataset_config not importable or missing date/measure"
                    ),
                }
                continue

            if not ds.channels:
                by_measure[target_col] = {
                    "ran": False,
                    "reason": "no channel columns identified for MMM",
                }
                continue

            by_measure[target_col] = _run_single_mmm(df.copy(), ds, column_specs=specs)

        if not by_measure:
            return {"ran": False, "reason": "no viable measure columns for MMM"}

        result: dict = {
            "ran": True,
            "signals": {"by_measure": by_measure},
        }
        if notes:
            result["note"] = " ".join(notes)
        return result
