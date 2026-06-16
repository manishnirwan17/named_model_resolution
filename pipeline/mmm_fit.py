"""
04_mmm_fit.py — Bayesian MMM: fit PyMC model, diagnose, decompose contributions.

Requires: pymc>=5.0   (uv add pymc)
          arviz>=0.17
Optional: pip install numpyro  (10x faster NUTS via JAX backend)

Reads:  MODEL_MATRIX   (output of 03_mmm_data_prep.py)
        mmm_meta.json  (scaler + channel metadata)
Writes: MMM_TRACE      (ArviZ InferenceData, NetCDF)
        CONTRIBUTIONS  (weekly contribution decomposition per channel)

Run locally:  uv run python models/04_mmm_fit.py
On Databricks: %run ./04_mmm_fit
  -> Use a GPU or High-Memory single-node cluster for NUTS sampling.
  -> Set MCMC_CHAINS=2, MCMC_DRAWS=1000 for a quick diagnostic first pass.
"""

from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import (
    MODEL_MATRIX, MODEL_OUT, MMM_TRACE, CONTRIBUTIONS,
    read_parquet, write_parquet, PARAMS, DATASET,
)
try:
    from databricks.sdk.runtime import *  # noqa: F401, F403
except Exception:
    pass


def build_pymc_model(X_train: np.ndarray, y_train: np.ndarray,
                     feature_cols: list[str], scaler: dict,
                     dataset_config=None,
                     alpha_prior_mu: float | None = None):
    """
    Bayesian linear MMM on log(target).

    log(target_t) = alpha + beta_channels @ X_t + beta_ctrl @ ctrl_t + eps_t

    Sign constraints:
      - Field/broadcast channels (_sat suffix): HalfNormal (forced positive)
      - Competitor channels:                    -HalfNormal (forced negative)
      - Lifecycle, seasonality, trend:           Normal (unconstrained)

    Intercept prior: auto-calibrated to mean(log(target)) on the training set
    so the model works correctly regardless of the target variable's scale.
    """
    import pymc as pm

    ds = dataset_config if dataset_config is not None else DATASET

    # Identify feature groups by column name patterns
    channel_idx = [i for i, c in enumerate(feature_cols) if "_sat" in c]
    comp_idx    = [i for i, c in enumerate(feature_cols)
                   if any(ch in c for ch in ds.competitor_channels)]
    ctrl_idx    = [i for i, c in enumerate(feature_cols)
                   if i not in channel_idx and i not in comp_idx]

    n_ch   = len(channel_idx)
    n_comp = len(comp_idx)
    n_ctrl = len(ctrl_idx)

    X_ch   = X_train[:, channel_idx]
    X_comp = X_train[:, comp_idx]
    X_ctrl = X_train[:, ctrl_idx]

    # Intercept prior centred at training-set mean of log(target)
    mu_prior = alpha_prior_mu if alpha_prior_mu is not None else float(y_train.mean())

    with pm.Model() as mmm:
        # Intercept — auto-calibrated to the dataset's log(target) scale
        alpha = pm.Normal("alpha", mu=mu_prior, sigma=0.5)

        # Channel coefficients — forced positive (HalfNormal)
        beta_ch = pm.HalfNormal("beta_ch", sigma=0.5, shape=n_ch)

        # Competitor spend — forced negative
        if n_comp > 0:
            beta_comp_raw = pm.HalfNormal("beta_comp_raw", sigma=0.3, shape=n_comp)
            beta_comp     = pm.Deterministic("beta_comp", -beta_comp_raw)
        else:
            beta_comp = pm.Data("beta_comp", np.zeros((len(y_train), 0)))

        # Control covariates (lifecycle, trend, Fourier) — unconstrained
        beta_ctrl = pm.Normal("beta_ctrl", mu=0, sigma=0.5, shape=n_ctrl)

        # Expected log(sales)
        mu = (alpha
              + X_ch   @ beta_ch
              + (X_comp @ beta_comp  if n_comp > 0 else 0)
              + X_ctrl  @ beta_ctrl)

        # Noise prior: HalfNormal(0.15) matches generator's lognormal sigma
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        # Likelihood
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

    return mmm, channel_idx, comp_idx, ctrl_idx


def mmm_fit() -> None:
    print("=" * 60)
    print("  04  MMM FIT")
    print("=" * 60)

    # ── Load model matrix and metadata ────────────────────────────────────────
    mat = read_parquet(MODEL_MATRIX)
    meta_path = Path(MODEL_OUT) / "mmm_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    scaler       = meta["scaler"]
    feature_cols = scaler["feature_cols"]

    # Prefix used when storing X columns in parquet
    X_cols = [f"X_{c}" for c in feature_cols]
    X      = mat[X_cols].values
    y      = mat["y"].values

    train_mask = mat["split"] == "train"
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]

    print(f"  X shape: {X.shape}  "
          f"(train={train_mask.sum()}, test={(~train_mask).sum()})")

    # ── Build and sample model ────────────────────────────────────────────────
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("  [error] PyMC/ArviZ not installed. Run: uv add pymc arviz")
        return

    alpha_prior_mu = float(y_train.mean())
    print(f"  Intercept prior: Normal(mu={alpha_prior_mu:.3f}, sigma=0.5) "
          f"[auto-calibrated from training mean of {DATASET.target_log_col}]")

    mmm, ch_idx, comp_idx, ctrl_idx = build_pymc_model(
        X_train, y_train, feature_cols, scaler, alpha_prior_mu=alpha_prior_mu)

    print(f"\n  Sampling: {PARAMS['MCMC_CHAINS']} chains x "
          f"{PARAMS['MCMC_DRAWS']} draws (tune={PARAMS['MCMC_TUNE']}) ...")

    sampler_kwargs = dict(
        draws         = PARAMS["MCMC_DRAWS"],
        tune          = PARAMS["MCMC_TUNE"],
        chains        = PARAMS["MCMC_CHAINS"],
        target_accept = PARAMS["MCMC_TARGET_ACCEPT"],
        random_seed   = PARAMS["MCMC_SEED"],
        return_inferencedata = True,
    )
    # Use numpyro backend if available (10x faster via JAX)
    try:
        import numpyro  # noqa: F401
        sampler_kwargs["nuts_sampler"] = "numpyro"
        print("  numpyro backend detected — using JAX for faster sampling")
    except ImportError:
        pass

    with mmm:
        trace = pm.sample(**sampler_kwargs)

    # ── Convergence diagnostics ───────────────────────────────────────────────
    summary = az.summary(trace, var_names=["alpha", "beta_ch", "sigma"],
                         round_to=4)
    print(f"\n  Convergence summary (top rows):\n{summary.head(12).to_string()}")

    max_rhat = summary["r_hat"].max()
    min_ess  = summary["ess_bulk"].min()
    divs     = trace.sample_stats["diverging"].sum().item()

    print(f"\n  max R-hat:    {max_rhat:.4f}  (target < 1.01)")
    print(f"  min ESS_bulk: {min_ess:.0f}   (target > 400)")
    print(f"  Divergences:  {divs}           (target = 0)")

    if max_rhat > 1.05:
        warnings.warn("R-hat > 1.05: chains have NOT converged. "
                      "Increase draws or tighten priors.")
    if divs > 0:
        warnings.warn(f"{divs} divergences — raise target_accept to 0.95 "
                      "or add stronger priors for correlated channels.")

    # ── Posterior predictive check ─────────────────────────────────────────────
    with mmm:
        ppc = pm.sample_posterior_predictive(trace)

    y_hat_train = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
    mape_train  = float(np.mean(np.abs(np.exp(y_hat_train) - np.exp(y_train))
                                / np.exp(y_train)) * 100)
    print(f"\n  In-sample MAPE: {mape_train:.1f}%  (target < 10%)")

    # ── Save trace ─────────────────────────────────────────────────────────────
    try:
        trace.to_netcdf(MMM_TRACE)
        print(f"\n  Trace saved -> {MMM_TRACE}")
    except Exception as e: 
        print("="*50)
        print(f"[error] {e}")
        print("="*50)
        
    # ── Contribution decomposition ─────────────────────────────────────────────
    # posterior mean of each parameter
    beta_ch_mean   = trace.posterior["beta_ch"].mean(dim=["chain", "draw"]).values
    beta_ctrl_mean = trace.posterior["beta_ctrl"].mean(dim=["chain", "draw"]).values
    alpha_mean     = float(trace.posterior["alpha"].mean())

    contrib = pd.DataFrame({"week": mat["week"].values})
    contrib["baseline"] = alpha_mean

    for i, col_idx in enumerate(ch_idx):
        col_name = feature_cols[col_idx]
        contrib[f"contrib_{col_name}"] = beta_ch_mean[i] * X[:, col_idx]

    for i, col_idx in enumerate(ctrl_idx):
        col_name = feature_cols[col_idx]
        contrib[f"contrib_{col_name}"] = beta_ctrl_mean[i] * X[:, col_idx]

    if comp_idx:
        beta_comp_mean = trace.posterior["beta_comp"].mean(dim=["chain", "draw"]).values
        for i, col_idx in enumerate(comp_idx):
            col_name = feature_cols[col_idx]
            contrib[f"contrib_{col_name}"] = beta_comp_mean[i] * X[:, col_idx]

    contrib["y_actual"]    = y
    contrib["y_predicted"] = contrib[[c for c in contrib.columns
                                      if c.startswith("contrib_")]].sum(axis=1) + alpha_mean
    contrib["residual"]    = contrib["y_actual"] - contrib["y_predicted"]
    contrib["split"]       = mat["split"].values

    write_parquet(contrib, CONTRIBUTIONS)
    print(f"  Contributions written -> {CONTRIBUTIONS}")

    # ── Scaler for back-transform (saved to JSON in 03, used in 06) ───────────
    print("\n  Done. Run 05_integration.py to map CPs to attribution shifts.")
    print("=" * 60)


if __name__ == "__main__":
    mmm_fit()
