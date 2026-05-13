#!/usr/bin/env python3
"""
Export dashboard data from fitted Bayesian food production models.

Reads ArviZ InferenceData (.nc files) and exports lightweight
CSV/JSON files for an interactive HTML dashboard.

Usage:
    pixi run python scripts/export_dashboard_data.py --variant base
    pixi run python scripts/export_dashboard_data.py --all
    pixi run python scripts/export_dashboard_data.py --variant base --skip-shapley
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyprojroot import here
from scipy.special import expit

RESULTS_DIR = here("results")

PRIOR_POSTERIOR_PARAMS = [
    "m0",
    "k0",
    "b0",
    "intercept_mu",
    "b_groupsize_mu0",
    "eta_mu0",
    "shape",
    "effort_intercept",
    "effort_age",
    "effort_age2",
    "intercept_success",
    "eta_success0",
    "gp_lengthscale",
    "gp_sigma_effort",
    "gp_sigma_success",
    "gp_sigma_returns",
    "sigma_gender_effort",
    "sigma_gender_skill",
    "sigma_gender_success",
]

QUANTILE_PROBS = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
QUANTILE_NAMES = ["q025", "q10", "q25", "median", "q75", "q90", "q975"]


def _flatten_chains(x: np.ndarray) -> np.ndarray:
    """Reshape (chain, draw, ...) to (chain*draw, ...)."""
    n_chains, n_draws = x.shape[:2]
    return x.reshape(n_chains * n_draws, *x.shape[2:])


def _subsample(x: np.ndarray, n: int = 1000, rng_seed: int = 42) -> np.ndarray:
    """Subsample first axis to n rows."""
    if x.shape[0] <= n:
        return x
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(x.shape[0], size=n, replace=False)
    idx.sort()
    return x[idx]


def _quantile_row(
    samples: np.ndarray,
) -> dict[str, float]:
    """Compute quantiles and mean from a 1-D array of samples."""
    qs = np.quantile(samples, QUANTILE_PROBS)
    row = {name: float(q) for name, q in zip(QUANTILE_NAMES, qs)}
    row["mean"] = float(np.mean(samples))
    return row


def _get_gender_coords(idata) -> list[str]:
    """Return gender coordinate labels as strings."""
    gender_coords = idata.posterior.coords["gender"].values
    return [str(g) for g in gender_coords]


# =========================================================================
# 1. Manifest
# =========================================================================


def export_manifest(
    idata: az.InferenceData,
    config: dict,
    export_dir: Path,
) -> None:
    post = idata.posterior
    manifest = {
        "variant": config.get("name", "unknown"),
        "exported_at": datetime.now().isoformat(),
        "n_foragers": int(post.dims["forager"]),
        "n_groups": int(post.dims["group"]),
        "n_dates": int(post.dims["date"]),
        "n_chains": int(post.dims["chain"]),
        "n_draws": int(post.dims["draw"]),
        "kcal_scale": float(post.attrs.get("kcal_scale", 1.0)),
        "age_scale": float(post.attrs.get("age_scale", 1.0)),
        "config": config,
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("  wrote manifest.json")


# =========================================================================
# 1b. Key estimates for manuscript
# =========================================================================


def _hdi(samples, prob=0.95):
    """Compute HDI from flat array."""
    hdi = az.hdi(samples, hdi_prob=prob)
    return float(hdi[0]), float(hdi[1])


def _est(samples, name, prob=0.95):
    """Build an estimate dict: mean, median, hdi_lo, hdi_hi."""
    lo, hi = _hdi(samples, prob)
    return {
        "name": name,
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "hdi_lo": lo,
        "hdi_hi": hi,
        "sd": float(np.std(samples)),
    }


def export_estimates(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Export key manuscript estimates as JSON."""
    post = idata.posterior
    kcal_scale = float(post.attrs.get("kcal_scale", 1.0))
    age_scale = float(post.attrs.get("age_scale", 1.0))

    estimates = []

    # --- Skill curve ---
    m0 = _flatten_chains(post["m0"].values)
    k0 = _flatten_chains(post["k0"].values)
    b0 = _flatten_chains(post["b0"].values)
    m = np.exp(m0)
    k = np.exp(k0)
    b = np.exp(b0)

    # Peak skill age: x_peak = (1/k) * log(1 + k*b/m)
    x_peak_scaled = (1 / k) * np.log(1 + k * b / m)
    peak_years = x_peak_scaled * age_scale
    estimates.append(_est(peak_years, "Peak skill age (years)"))

    # Age at 50% of peak skill
    ages_grid = np.linspace(0.01, 1.0, 1000)
    S_curves = np.exp(-m[:, None] * ages_grid[None, :]) * (
        np.clip(1 - np.exp(-k[:, None] * ages_grid[None, :]), 1e-12, None)
    ) ** b[:, None]
    peak_S = S_curves.max(axis=1)
    age_50 = np.full(len(m), np.nan)
    age_80 = np.full(len(m), np.nan)
    for i in range(len(m)):
        above_50 = np.where(S_curves[i] >= 0.5 * peak_S[i])[0]
        above_80 = np.where(S_curves[i] >= 0.8 * peak_S[i])[0]
        if len(above_50) > 0:
            age_50[i] = ages_grid[above_50[0]] * age_scale
        if len(above_80) > 0:
            age_80[i] = ages_grid[above_80[0]] * age_scale
    estimates.append(_est(age_50[~np.isnan(age_50)], "Age at 50% peak skill (years)"))
    estimates.append(_est(age_80[~np.isnan(age_80)], "Age at 80% peak skill (years)"))

    # --- Effort ---
    effort_age = _flatten_chains(post["effort_age"].values)
    effort_age2 = _flatten_chains(post["effort_age2"].values)
    estimates.append(_est(effort_age, "Effort age (linear, logit)"))
    estimates.append(_est(effort_age2, "Effort age² (quadratic, logit)"))

    # --- Effort-skill correlation ---
    if "chol_cov_corr" in post:
        corr = _flatten_chains(post["chol_cov_corr"].values[:, :, 0, 1])
        est = _est(corr, "Effort-skill correlation (ψ)")
        est["prob_negative"] = float(np.mean(corr < 0))
        estimates.append(est)

    # --- RE standard deviations ---
    if "chol_cov_stds" in post:
        stds = _flatten_chains(post["chol_cov_stds"].values)
        estimates.append(_est(stds[:, 0], "σ_effort (RE)"))
        estimates.append(_est(stds[:, 1], "σ_skill (RE)"))

    # --- Returns ---
    intercept_mu = _flatten_chains(post["intercept_mu"].values)
    estimates.append(_est(intercept_mu, "Returns intercept"))

    b_gs = _flatten_chains(post["b_groupsize_mu"].values)
    estimates.append(_est(b_gs, "Group size elasticity"))

    eta_mu = _flatten_chains(post["eta_mu"].values)
    estimates.append(_est(eta_mu, "Skill elasticity on returns (η)"))

    if "shape" in post:
        shape = _flatten_chains(post["shape"].values)
        estimates.append(_est(shape, "Dispersion (shape/σ)"))

    # --- Kcal scale ---
    estimates.append({
        "name": "Kcal scale (mean non-zero returns)",
        "mean": kcal_scale, "median": kcal_scale,
        "hdi_lo": kcal_scale, "hdi_hi": kcal_scale, "sd": 0.0,
    })

    (export_dir / "estimates.json").write_text(
        json.dumps(estimates, indent=2)
    )
    print("  wrote estimates.json")


# =========================================================================
# 2. Prior / Posterior comparison
# =========================================================================


def export_prior_posterior(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    rows = []
    n_target = 1000

    for param in PRIOR_POSTERIOR_PARAMS:
        if param not in idata.posterior:
            warnings.warn(f"Skipping {param}: not in posterior")
            continue
        post_vals = _subsample(
            _flatten_chains(idata.posterior[param].values), n_target
        )
        if param in idata.prior:
            prior_all = _flatten_chains(idata.prior[param].values)
            prior_vals = _subsample(prior_all, min(n_target, len(prior_all)))
        else:
            prior_vals = np.array([])

        n_out = max(len(post_vals), len(prior_vals))
        for i in range(n_out):
            rows.append(
                {
                    "parameter": param,
                    "sample_idx": i,
                    "prior_value": float(prior_vals[i]) if i < len(prior_vals) else None,
                    "posterior_value": float(post_vals[i]) if i < len(post_vals) else None,
                }
            )

    # Effort-skill correlation from chol_cov_corr
    if "chol_cov_corr" in idata.posterior:
        post_corr = _subsample(
            _flatten_chains(
                idata.posterior["chol_cov_corr"].values[:, :, 0, 1]
            ),
            n_target,
        )
        if "chol_cov_corr" in idata.prior:
            prior_all = _flatten_chains(
                idata.prior["chol_cov_corr"].values[:, :, 0, 1]
            )
            prior_corr = _subsample(prior_all, min(n_target, len(prior_all)))
        else:
            prior_corr = np.array([])
        n_out = max(len(post_corr), len(prior_corr))
        for i in range(n_out):
            rows.append(
                {
                    "parameter": "effort_skill_corr",
                    "sample_idx": i,
                    "prior_value": float(prior_corr[i]) if i < len(prior_corr) else None,
                    "posterior_value": float(post_corr[i]) if i < len(post_corr) else None,
                }
            )

    pd.DataFrame(rows).to_csv(
        export_dir / "prior_posterior.csv", index=False
    )

    # Also export forest plot summary (mean + HDI per parameter)
    forest_rows = []
    all_params = PRIOR_POSTERIOR_PARAMS + ["effort_skill_corr"]
    for param in all_params:
        for group_name, group_data in [
            ("posterior", idata.posterior),
            ("prior", idata.prior),
        ]:
            if param == "effort_skill_corr":
                if "chol_cov_corr" not in group_data:
                    continue
                vals = _flatten_chains(
                    group_data["chol_cov_corr"].values[:, :, 0, 1]
                )
            elif param in group_data:
                vals = _flatten_chains(group_data[param].values)
            else:
                continue
            hdi = az.hdi(vals, hdi_prob=0.94)
            forest_rows.append({
                "parameter": param,
                "group": group_name,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "hdi_3": float(hdi[0]),
                "hdi_97": float(hdi[1]),
                "sd": float(np.std(vals)),
            })
    pd.DataFrame(forest_rows).to_csv(
        export_dir / "forest_plot.csv", index=False
    )
    print("  wrote prior_posterior.csv + forest_plot.csv")


# =========================================================================
# 3. Skill curves: S, M, K vs age
# =========================================================================


def _skill_curves_from_group(
    group_data, idata, age_years, age_scaled, source_label,
):
    """Compute skill curve rows from a posterior or prior group."""
    m0 = _flatten_chains(group_data["m0"].values)
    k0 = _flatten_chains(group_data["k0"].values)
    b0 = _flatten_chains(group_data["b0"].values)

    gender_labels = _get_gender_coords(idata)
    m0_gender = _flatten_chains(group_data["m0_gender"].values)
    k0_gender = _flatten_chains(group_data["k0_gender"].values)
    b0_gender = _flatten_chains(group_data["b0_gender"].values)

    rows = []

    def _compute(m_log, k_log, b_log, gender_label):
        m = np.exp(m_log)
        k = np.exp(k_log)
        b = np.exp(b_log)
        for j, age_y in enumerate(age_years):
            x = age_scaled[j]
            M_vals = np.exp(-m * x)
            K_vals = np.clip(1.0 - np.exp(-k * x), 1e-12, None)
            S_vals = M_vals * K_vals ** b
            for curve_name, vals in [("S", S_vals), ("M", M_vals), ("K", K_vals)]:
                row = {
                    "age_years": age_y, "gender": gender_label,
                    "curve": curve_name, "source": source_label,
                }
                row.update(_quantile_row(vals))
                rows.append(row)

    _compute(m0, k0, b0, "population")
    for g_idx, g_label in enumerate(gender_labels):
        _compute(
            m0 + m0_gender[:, g_idx],
            k0 + k0_gender[:, g_idx],
            b0 + b0_gender[:, g_idx],
            g_label,
        )
    return rows


def export_skill_curves(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    post = idata.posterior
    age_scale = float(post.attrs.get("age_scale", 1.0))
    max_age_years = age_scale

    age_years = np.linspace(0.01, max_age_years, 100)
    age_scaled = age_years / age_scale

    rows = _skill_curves_from_group(post, idata, age_years, age_scaled, "posterior")

    if hasattr(idata, "prior"):
        rows += _skill_curves_from_group(
            idata.prior, idata, age_years, age_scaled, "prior",
        )

    pd.DataFrame(rows).to_csv(export_dir / "skill_curves.csv", index=False)

    # Compute age landmarks for population S(x): peak and 80% of peak
    m0 = _flatten_chains(post["m0"].values)
    k0 = _flatten_chains(post["k0"].values)
    b0 = _flatten_chains(post["b0"].values)
    m = np.exp(m0)
    k = np.exp(k0)
    b = np.exp(b0)

    fine_grid = np.linspace(0.01, max_age_years, 500)
    fine_scaled = fine_grid / age_scale
    # (S, grid)
    M_all = np.exp(-m[:, None] * fine_scaled[None, :])
    K_all = np.clip(1.0 - np.exp(-k[:, None] * fine_scaled[None, :]), 1e-12, None)
    S_all = M_all * K_all ** b[:, None]

    peak_idx = S_all.argmax(axis=1)
    peak_ages = fine_grid[peak_idx]
    peak_vals = S_all[np.arange(len(m)), peak_idx]

    age_80_samples = []
    for i in range(len(m)):
        threshold = 0.8 * peak_vals[i]
        above = np.where(S_all[i] >= threshold)[0]
        age_80_samples.append(fine_grid[above[0]] if len(above) > 0 else np.nan)
    age_80 = np.array(age_80_samples)

    landmarks = {
        "peak_age_mean": float(np.nanmean(peak_ages)),
        "peak_age_median": float(np.nanmedian(peak_ages)),
        "peak_age_q025": float(np.nanquantile(peak_ages, 0.025)),
        "peak_age_q975": float(np.nanquantile(peak_ages, 0.975)),
        "age_80pct_mean": float(np.nanmean(age_80)),
        "age_80pct_median": float(np.nanmedian(age_80)),
        "age_80pct_q025": float(np.nanquantile(age_80, 0.025)),
        "age_80pct_q975": float(np.nanquantile(age_80, 0.975)),
    }
    (export_dir / "skill_landmarks.json").write_text(json.dumps(landmarks, indent=2))

    print("  wrote skill_curves.csv + skill_landmarks.json")


# =========================================================================
# 4. Effort curves: P(effort) vs age
# =========================================================================


def export_effort_curves(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    post = idata.posterior
    age_scale = float(post.attrs.get("age_scale", 1.0))
    max_age_years = age_scale

    age_raw = idata.constant_data["age_raw"].values
    age_mean = float(np.mean(age_raw))
    age_std = float(np.std(age_raw))

    age_years = np.linspace(0.01, max_age_years, 100)
    age_z = (age_years - age_mean) / age_std  # (100,)

    gender_labels = _get_gender_coords(idata)
    rows = []

    for src, group in [("posterior", post)] + (
        [("prior", idata.prior)] if hasattr(idata, "prior") else []
    ):
        intercept = _flatten_chains(group["effort_intercept"].values)
        b1 = _flatten_chains(group["effort_age"].values)
        b2 = _flatten_chains(group["effort_age2"].values)
        intercept_g = _flatten_chains(group["effort_intercept_gender"].values)
        b1_g = _flatten_chains(group["effort_age_gender"].values)
        b2_g = _flatten_chains(group["effort_age2_gender"].values)

        def _compute_effort(intercept_v, b1_v, b2_v, gender_label):
            for j, age_y in enumerate(age_years):
                z = age_z[j]
                p = expit(intercept_v + b1_v * z + b2_v * z**2)
                row = {
                    "age_years": age_y, "gender": gender_label,
                    "curve": "effort", "source": src,
                }
                row.update(_quantile_row(p))
                rows.append(row)

        _compute_effort(intercept, b1, b2, "population")
        for g_idx, g_label in enumerate(gender_labels):
            _compute_effort(
                intercept + intercept_g[:, g_idx],
                b1 + b1_g[:, g_idx],
                b2 + b2_g[:, g_idx],
                g_label,
            )

    pd.DataFrame(rows).to_csv(export_dir / "effort_curves.csv", index=False)
    print("  wrote effort_curves.csv")


# =========================================================================
# 5. Success curves: P(success) vs age
# =========================================================================


def export_success_curves(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    post = idata.posterior
    age_scale = float(post.attrs.get("age_scale", 1.0))
    max_age_years = age_scale

    age_years = np.linspace(0.01, max_age_years, 100)
    age_scaled = age_years / age_scale

    gender_labels = _get_gender_coords(idata)
    rows = []

    for src, group in [("posterior", post)] + (
        [("prior", idata.prior)] if hasattr(idata, "prior") else []
    ):
        m0 = _flatten_chains(group["m0"].values)
        k0 = _flatten_chains(group["k0"].values)
        b0 = _flatten_chains(group["b0"].values)
        intercept_success = _flatten_chains(group["intercept_success"].values)
        eta_success0 = _flatten_chains(group["eta_success0"].values)
        m0_g = _flatten_chains(group["m0_gender"].values)
        k0_g = _flatten_chains(group["k0_gender"].values)
        b0_g = _flatten_chains(group["b0_gender"].values)
        intercept_success_g = _flatten_chains(group["intercept_success_gender"].values)
        eta_success0_g = _flatten_chains(group["eta_success0_gender"].values)

        def _compute_success(m_log, k_log, b_log, alpha_log, eta_log, gender_label):
            m = np.exp(m_log)
            k = np.exp(k_log)
            b = np.exp(b_log)
            alpha = np.exp(alpha_log)
            eta = np.exp(eta_log)
            for j, age_y in enumerate(age_years):
                x = age_scaled[j]
                K_vals = np.clip(1.0 - np.exp(-k * x), 1e-12, None)
                S = np.exp(-m * x) * K_vals ** b
                S_safe = np.clip(S, 1e-12, None)
                linear = np.log(S_safe**eta * alpha + 1e-10)
                p = expit(linear)
                row = {
                    "age_years": age_y, "gender": gender_label,
                    "curve": "success", "source": src,
                }
                row.update(_quantile_row(p))
                rows.append(row)

        _compute_success(m0, k0, b0, intercept_success, eta_success0, "population")
        for g_idx, g_label in enumerate(gender_labels):
            _compute_success(
                m0 + m0_g[:, g_idx],
                k0 + k0_g[:, g_idx],
                b0 + b0_g[:, g_idx],
                intercept_success + intercept_success_g[:, g_idx],
                eta_success0 + eta_success0_g[:, g_idx],
                g_label,
            )

    pd.DataFrame(rows).to_csv(
        export_dir / "success_curves.csv", index=False
    )
    print("  wrote success_curves.csv")


# =========================================================================
# 6. Diagnostics
# =========================================================================


def export_diagnostics(
    idata: az.InferenceData,
    variant_dir: Path,
    export_dir: Path,
) -> dict:
    """Export diagnostics CSV and meta JSON. Returns meta dict for reuse."""
    summary_path = variant_dir / "model_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        warnings.warn(
            "model_summary.csv not found, computing summary (slow)"
        )
        df = az.summary(idata).reset_index()

    # Rename unnamed first column to 'parameter'
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "parameter"})

    # Filter out internal/non-centered params that have misleading diagnostics
    internal_patterns = ["_raw", "__", "_log__", "zerosum__"]
    mask = ~df["parameter"].str.contains(
        "|".join(internal_patterns), regex=True, na=False
    )
    df_filtered = df[mask].copy()
    df_filtered.to_csv(export_dir / "diagnostics.csv", index=False)
    # Also save unfiltered for completeness
    df.to_csv(export_dir / "diagnostics_full.csv", index=False)

    # Divergences per chain
    diverging = idata.sample_stats["diverging"].values  # (chain, draw)
    div_per_chain = [int(diverging[c].sum()) for c in range(diverging.shape[0])]

    # Rhat and ESS from filtered summary
    rhat_col = "r_hat" if "r_hat" in df_filtered.columns else "rhat"
    ess_bulk_col = "ess_bulk"
    ess_tail_col = "ess_tail"

    max_rhat = float(df_filtered[rhat_col].max()) if rhat_col in df_filtered.columns else None
    min_ess_bulk = (
        float(df_filtered[ess_bulk_col].min()) if ess_bulk_col in df_filtered.columns else None
    )
    min_ess_tail = (
        float(df_filtered[ess_tail_col].min()) if ess_tail_col in df_filtered.columns else None
    )

    meta = {
        "n_chains": int(idata.posterior.dims["chain"]),
        "n_draws": int(idata.posterior.dims["draw"]),
        "divergences_per_chain": div_per_chain,
        "total_divergences": int(sum(div_per_chain)),
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
    }
    (export_dir / "diagnostics_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print("  wrote diagnostics.csv + diagnostics_meta.json")
    return meta


# =========================================================================
# 6b. Trace plots
# =========================================================================

TRACE_GROUPS = {
    "skill": ["m0", "k0", "b0"],
    "returns": ["intercept_mu", "b_groupsize_mu0", "eta_mu0", "shape"],
    "effort": ["effort_intercept", "effort_age", "effort_age2"],
    "success": ["intercept_success", "eta_success0"],
    "gp": ["gp_lengthscale", "gp_sigma_effort", "gp_sigma_success", "gp_sigma_returns"],
    "re": ["chol_cov"],
    "gender_skill": ["sigma_gender_skill", "m0_gender", "k0_gender", "b0_gender"],
    "gender_effort": ["sigma_gender_effort", "effort_intercept_gender", "effort_age_gender", "effort_age2_gender"],
    "gender_success": ["sigma_gender_success", "intercept_success_gender", "eta_success0_gender"],
}


def export_trace_plots(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Pre-render ArviZ trace plots as PNGs, one per parameter group."""
    trace_names = []
    for group_name, var_names in TRACE_GROUPS.items():
        present = [v for v in var_names if v in idata.posterior]
        if not present:
            continue
        try:
            axes = az.plot_trace(idata, var_names=present, compact=True)
            fig = axes.flatten()[0].get_figure()
            fig.suptitle(group_name.replace("_", " ").title(), fontsize=14, y=1.01)
            fig.tight_layout()
            fig.savefig(
                export_dir / f"trace_{group_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            trace_names.append(group_name)
        except Exception as e:
            warnings.warn(f"trace_{group_name} failed: {e}")

    (export_dir / "trace_index.json").write_text(
        json.dumps(trace_names)
    )
    print(f"  wrote {len(trace_names)} trace plot PNGs")


# =========================================================================
# 7. Pareto k
# =========================================================================


def export_pareto_k(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    try:
        loo_result = az.loo(idata, var_name="kcal", pointwise=True)
    except Exception as e:
        warnings.warn(f"LOO computation failed: {e}")
        return

    pareto_k_vals = loo_result.pareto_k.values
    loo_i_vals = loo_result.loo_i.values

    df = pd.DataFrame(
        {
            "group_idx": np.arange(len(pareto_k_vals)),
            "pareto_k": pareto_k_vals,
            "loo_i": loo_i_vals,
        }
    )
    df.to_csv(export_dir / "pareto_k.csv", index=False)

    summary = {
        "loo_estimate": float(loo_result.elpd_loo),
        "loo_se": float(loo_result.se),
        "p_loo": float(loo_result.p_loo),
        "n_bad_k": int((pareto_k_vals > 0.7).sum()),
        "max_k": float(np.max(pareto_k_vals)),
    }
    (export_dir / "pareto_k_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("  wrote pareto_k.csv + pareto_k_summary.json")


# =========================================================================
# 7b. Power-scaling sensitivity (psens)
# =========================================================================

# Population-level parameters to include in psens
PSENS_PARAMS = [
    "m0", "k0", "b0",
    "intercept_mu", "b_groupsize_mu0", "eta_mu0", "shape",
    "effort_intercept", "effort_age", "effort_age2",
    "intercept_success", "eta_success0",
    "gp_lengthscale", "gp_sigma_effort", "gp_sigma_success", "gp_sigma_returns",
    # Gender offsets (fixed effects, not individual RE)
    "m0_gender", "k0_gender", "b0_gender",
    "effort_intercept_gender", "effort_age_gender", "effort_age2_gender",
    "intercept_success_gender", "eta_success0_gender",
]


def export_psens(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Export power-scaling sensitivity for population-level parameters."""
    # Filter to params that exist in this model's posterior
    available = [p for p in PSENS_PARAMS if p in idata.posterior]
    if not available:
        warnings.warn("No population-level params found for psens")
        return

    rows = []
    for component in ["likelihood", "prior"]:
        try:
            result = az.psens(idata, component=component, var_names=available)
            for var_name in result.data_vars:
                vals = result[var_name].values
                if vals.ndim == 0:
                    rows.append({
                        "parameter": str(var_name),
                        "component": component,
                        "sensitivity": float(vals),
                    })
                else:
                    # Vector param: report each element
                    for idx in np.ndindex(vals.shape):
                        label = f"{var_name}[{','.join(str(i) for i in idx)}]"
                        rows.append({
                            "parameter": label,
                            "component": component,
                            "sensitivity": float(vals[idx]),
                        })
        except Exception as e:
            warnings.warn(f"psens({component}) failed: {e}")

    if not rows:
        warnings.warn("No psens results produced")
        return

    df_psens = pd.DataFrame(rows)
    df_psens.to_csv(export_dir / "psens.csv", index=False)

    # Pre-render psens plot as annotated PNG
    # ZeroSumNormal pairs [0]/[1] are identical — keep only [0]
    df_plot = df_psens[~df_psens["parameter"].str.endswith("[1]")].copy()
    df_plot["parameter"] = df_plot["parameter"].str.replace("[0]", "", regex=False)

    params_ordered = (
        df_plot[df_plot.component == "likelihood"]
        .sort_values("sensitivity")["parameter"]
        .tolist()
    )
    if not params_ordered:
        params_ordered = df_plot["parameter"].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, max(4, len(params_ordered) * 0.35)))
    y_pos = np.arange(len(params_ordered))
    bar_h = 0.35

    for component, color, offset, label in [
        ("likelihood", "#3b82f6", -bar_h / 2, "Likelihood"),
        ("prior", "#9ca3af", bar_h / 2, "Prior"),
    ]:
        sub = df_plot[df_plot.component == component]
        vals = [
            float(sub.loc[sub.parameter == p, "sensitivity"].values[0])
            if p in sub.parameter.values
            else 0.0
            for p in params_ordered
        ]
        ax.barh(y_pos + offset, vals, bar_h, label=label, color=color, alpha=0.85)

    ax.axvline(
        0.05, color="#dc2626", linestyle="--", linewidth=1.2,
        label="Threshold (0.05)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_ordered, fontsize=9)
    ax.set_xlabel("Sensitivity")
    ax.set_title(
        "Power-Scaling Sensitivity\n"
        "(prior > 0.05 = informative prior; "
        "likelihood < 0.05 = weak data signal)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(export_dir / "psens_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  wrote psens.csv + psens_plot.png")

    # Power-scaling posterior plots for key parameters
    _export_powerscale_posteriors(idata, export_dir)


# Parameters to show power-scaled posteriors for, with extraction logic
POWERSCALE_PARAMS = {
    "effort_skill_corr": {
        "label": "Effort-Skill Correlation",
        "extract": lambda post: post["chol_cov_corr"].values[:, :, 0, 1].flatten(),
    },
    "m0": {"label": "Senescence Rate (m₀, log scale)"},
    "k0": {"label": "Learning Rate (k₀, log scale)"},
    "b0": {"label": "Skill Elasticity (b₀, log scale)"},
    "shape": {"label": "LogNormal Dispersion (σ)"},
    "intercept_mu": {"label": "Returns Intercept"},
    "eta_mu0": {"label": "Skill Elasticity on Returns (log)"},
    "b_groupsize_mu0": {"label": "Group Size Effect (log)"},
}

POWERSCALE_ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
POWERSCALE_COLORS = {
    0.5: "#93c5fd", 0.75: "#fdba74", 1.0: "#3b82f6",
    1.25: "#86efac", 1.5: "#fca5a5", 2.0: "#c4b5fd",
}


def _export_powerscale_posteriors(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Render power-scaled posterior density plots for key parameters."""
    if not hasattr(idata, "log_prior"):
        warnings.warn("log_prior not in idata, skipping powerscale plots")
        return

    post = idata.posterior
    log_prior = idata.log_prior

    # Sum all log-prior components into a single vector
    n_samples = post.sizes["chain"] * post.sizes["draw"]
    lp_total = np.zeros(n_samples)
    for v in log_prior.data_vars:
        vals = log_prior[v].values
        flat = vals.reshape(vals.shape[0] * vals.shape[1], *vals.shape[2:])
        if flat.ndim > 1:
            flat = flat.sum(axis=tuple(range(1, flat.ndim)))
        lp_total += flat

    n_params = len(POWERSCALE_PARAMS)
    ncols = 2
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = axes.flatten()

    plot_idx = 0
    for key, spec in POWERSCALE_PARAMS.items():
        ax = axes[plot_idx]
        plot_idx += 1

        # Extract samples
        if "extract" in spec:
            samples = spec["extract"](post)
        elif key in post:
            samples = post[key].values.flatten()
        else:
            ax.set_visible(False)
            continue

        from scipy.stats import gaussian_kde

        for alpha in POWERSCALE_ALPHAS:
            color = POWERSCALE_COLORS.get(alpha, "#94a3b8")
            if alpha == 1.0:
                kde = gaussian_kde(samples)
                lw, style, label = 2.5, "-", f"α={alpha:.1f} (original)"
            else:
                log_w = (alpha - 1) * lp_total
                log_w -= log_w.max()
                w = np.exp(log_w)
                w /= w.sum()
                kde = gaussian_kde(samples, weights=w)
                lw = 1.5
                style = "--" if alpha < 1.0 else ":"
                label = f"α={alpha:.1f}"

            x_grid = np.linspace(
                np.quantile(samples, 0.001),
                np.quantile(samples, 0.999),
                200,
            )
            ax.plot(x_grid, kde(x_grid), color=color, lw=lw, ls=style, label=label)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(spec["label"], fontsize=10)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Prior Power-Scaling Sensitivity\n"
        "(α<1 weakens prior → data-driven; α>1 strengthens prior)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        export_dir / "psens_posteriors.png", dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  wrote psens_posteriors.png")


# =========================================================================
# 8. Predictive checks
# =========================================================================


def export_predictive_checks(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Pre-render ArviZ predictive check plots as PNG images."""
    components = {
        "kcal": "kcal",
        "effort": "effort",
        "success": "non_zero_prod",
    }

    plot_size = (7, 4)
    binary_components = {"effort", "success"}

    # --- Continuous (kcal): density, ECDF, prior PPC, LOO-PIT ---
    for label, var_name in components.items():
        if label in binary_components:
            continue
        # Posterior PPC density
        try:
            fig, ax = plt.subplots(figsize=plot_size)
            az.plot_ppc(idata, var_names=[var_name], kind="kde", mean=False, ax=ax)
            ax.set_xscale("log")
            ax.set_title(f"Posterior Predictive: {label}")
            fig.tight_layout()
            fig.savefig(export_dir / f"ppc_dens_{label}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"ppc_dens_{label} failed: {e}")

        # Posterior PPC ECDF
        try:
            fig, ax = plt.subplots(figsize=plot_size)
            az.plot_ppc(idata, var_names=[var_name], kind="cumulative", mean=False, ax=ax)
            ax.set_xscale("log")
            ax.set_title(f"Posterior Predictive ECDF: {label}")
            fig.tight_layout()
            fig.savefig(export_dir / f"ppc_ecdf_{label}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"ppc_ecdf_{label} failed: {e}")

        # Prior PPC density
        try:
            fig, ax = plt.subplots(figsize=plot_size)
            az.plot_ppc(
                idata, var_names=[var_name], kind="kde",
                group="prior", mean=False, ax=ax,
            )
            ax.set_xscale("log")
            ax.set_title(f"Prior Predictive: {label}")
            fig.tight_layout()
            fig.savefig(export_dir / f"prior_ppc_dens_{label}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"prior_ppc_dens_{label} failed: {e}")

        # LOO-PIT (only valid for continuous outcomes)
        try:
            fig, ax = plt.subplots(figsize=plot_size)
            az.plot_loo_pit(idata, y=var_name, ecdf=True, ecdf_fill=True, ax=ax)
            ax.set_title(f"LOO-PIT ECDF Difference: {label}")
            fig.tight_layout()
            fig.savefig(export_dir / f"loo_pit_ecdf_{label}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"loo_pit_ecdf_{label} failed: {e}")

    # --- Binary (effort, success): proportion bar PPC + calibration ---
    for label, var_name in components.items():
        if label not in binary_components:
            continue

        for group_label, group_name in [("posterior", "ppc_bars"), ("prior", "prior_ppc_bars")]:
            try:
                obs = idata.observed_data[var_name].values
                pp_group = (
                    idata.posterior_predictive if group_label == "posterior"
                    else idata.prior_predictive
                )
                pp = pp_group[var_name].values  # (chain, draw, obs)
                pp_flat = pp.reshape(-1, pp.shape[-1])
                obs_prop = obs.mean()
                pp_props = pp_flat.mean(axis=1)  # proportion of 1s per draw

                fig, ax = plt.subplots(figsize=plot_size)
                ax.hist(pp_props, bins=30, density=True, alpha=0.6,
                        color="#3b82f6", label="Predictive draws")
                ax.axvline(obs_prop, color="#dc2626", lw=2, ls="--",
                           label=f"Observed ({obs_prop:.2f})")
                ax.set_xlabel(f"Proportion ({label} = 1)")
                ax.set_ylabel("Density")
                ax.set_title(f"{'Posterior' if group_label == 'posterior' else 'Prior'} Predictive: {label}")
                ax.legend(fontsize=9)
                fig.tight_layout()
                fig.savefig(export_dir / f"{group_name}_{label}.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"{group_name}_{label} failed: {e}")

    # --- Calibration plots for binary components ---
    for label, var_name in [("effort", "effort"), ("success", "non_zero_prod")]:
        try:
            obs = idata.observed_data[var_name].values
            pp = idata.posterior_predictive[var_name].values
            pp_mean = pp.reshape(-1, pp.shape[-1]).mean(axis=0)

            # Bin predicted probabilities and compute observed frequencies
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            obs_freq = np.zeros(n_bins)
            pred_freq = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (pp_mean >= bin_edges[b]) & (pp_mean < bin_edges[b + 1])
                if mask.sum() > 0:
                    obs_freq[b] = obs[mask].mean()
                    pred_freq[b] = pp_mean[mask].mean()
                    counts[b] = mask.sum()

            fig, ax = plt.subplots(figsize=plot_size)
            valid = counts > 0
            ax.bar(bin_centers[valid], obs_freq[valid], width=0.08,
                   alpha=0.6, color="#3b82f6", label="Observed frequency")
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
            ax.scatter(pred_freq[valid], obs_freq[valid], color="#dc2626",
                       s=counts[valid] * 2, zorder=3, label="Bin (size ∝ n)")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"Calibration: {label}")
            ax.legend(fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(export_dir / f"calibration_{label}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"calibration_{label} failed: {e}")

    print("  wrote predictive check PNGs")


# =========================================================================
# 9. Observed data
# =========================================================================


def export_observed_data(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    cd = idata.constant_data

    # Group-level observed data
    obs_kcal = idata.observed_data["kcal"].values
    group_date_idx = cd["group_date_idx"].values

    # Group size: count valid (>=0) entries per row in forager_ids
    forager_ids = cd["forager_ids"].values  # (group, forager_in_group)
    group_size = (forager_ids >= 0).sum(axis=1)

    df_obs = pd.DataFrame(
        {
            "group_idx": np.arange(len(obs_kcal)),
            "kcal": obs_kcal,
            "group_size": group_size,
            "date_idx": group_date_idx,
        }
    )
    df_obs.to_csv(export_dir / "observed_data.csv", index=False)

    # Forager metadata
    age_raw = cd["age_raw"].values
    gender_idx = cd["gender_idx"].values
    gender_map = {0: "male", 1: "female"}

    df_forager = pd.DataFrame(
        {
            "forager_idx": np.arange(len(age_raw)),
            "age": age_raw,
            "gender": [gender_map.get(int(g), str(g)) for g in gender_idx],
        }
    )
    df_forager.to_csv(export_dir / "forager_metadata.csv", index=False)
    print("  wrote observed_data.csv + forager_metadata.csv")


# =========================================================================
# 10. GP temporal effects
# =========================================================================


def export_gp_temporal(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    cd = idata.constant_data
    date_numeric = cd["date_numeric"].values  # (date,)
    post = idata.posterior

    gp_vars = ["gp_effort", "gp_success", "gp_returns"]
    results: dict[str, dict[str, np.ndarray]] = {}

    for gp_name in gp_vars:
        vals = _flatten_chains(post[gp_name].values)  # (S, date)
        results[gp_name] = {
            "mean": np.mean(vals, axis=0),
            "q025": np.quantile(vals, 0.025, axis=0),
            "q975": np.quantile(vals, 0.975, axis=0),
        }

    n_dates = len(date_numeric)
    df = pd.DataFrame({"date_idx": np.arange(n_dates), "date_numeric": date_numeric})
    for gp_name in gp_vars:
        for stat in ["mean", "q025", "q975"]:
            df[f"{gp_name}_{stat}"] = results[gp_name][stat]

    df.to_csv(export_dir / "gp_temporal.csv", index=False)
    print("  wrote gp_temporal.csv")


# =========================================================================
# 11. Random effects
# =========================================================================


def export_random_effects(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    post = idata.posterior
    cd = idata.constant_data

    age_raw = cd["age_raw"].values
    gender_idx = cd["gender_idx"].values
    gender_map = {0: "male", 1: "female"}

    # RE: (chain, draw, forager, re_response)
    re_vals = _flatten_chains(post["re"].values)  # (S, forager, 2)
    re_effort = re_vals[:, :, 0]  # (S, forager)
    re_skill = re_vals[:, :, 1]

    # S and S_base: (chain, draw, forager)
    S_vals = _flatten_chains(post["S"].values)
    S_base_vals = _flatten_chains(post["S_base"].values)

    n_foragers = re_effort.shape[1]
    rows = []
    for i in range(n_foragers):
        rows.append(
            {
                "forager_idx": i,
                "age": float(age_raw[i]),
                "gender": gender_map.get(int(gender_idx[i]), str(gender_idx[i])),
                "re_effort_mean": float(np.mean(re_effort[:, i])),
                "re_effort_sd": float(np.std(re_effort[:, i])),
                "re_skill_mean": float(np.mean(re_skill[:, i])),
                "re_skill_sd": float(np.std(re_skill[:, i])),
                "S_mean": float(np.mean(S_vals[:, i])),
                "S_sd": float(np.std(S_vals[:, i])),
                "S_base_mean": float(np.mean(S_base_vals[:, i])),
                "S_base_sd": float(np.std(S_base_vals[:, i])),
            }
        )

    pd.DataFrame(rows).to_csv(
        export_dir / "random_effects.csv", index=False
    )
    print("  wrote random_effects.csv")


# =========================================================================
# 12. Effort–skill tradeoff (marginal effects)
# =========================================================================


def export_effort_skill_tradeoff(
    idata: az.InferenceData,
    export_dir: Path,
) -> None:
    """Export skill-gap compensation table with posterior uncertainty.

    For a range of skill deficits, compute:
    - kcal loss per trip
    - extra subsistence days/month needed to compensate
    - whether compensation is feasible given effort headroom

    Exports overall + by-gender tables, using full posterior of eta.
    """
    post = idata.posterior
    cd = idata.constant_data

    eta_mu = _flatten_chains(post["eta_mu"].values)  # (S,)
    gender_idx = cd["gender_idx"].values

    # Observed effort per forager
    effort_forager_idx = cd["effort_forager_idx"].values
    effort_obs = cd["forager_effort"].values
    df_eff = pd.DataFrame({"f": effort_forager_idx, "e": effort_obs})
    per_forager_effort = df_eff.groupby("f")["e"].mean()

    gender_map = {0: "male", 1: "female"}
    days_per_month = 30
    gap_pcts = [2, 5, 10, 15, 20, 25, 30, 40, 50]

    def _compute_table(eta_samples, baseline_effort, label):
        rows = []
        headroom = (1 - baseline_effort) * days_per_month
        baseline_days = baseline_effort * days_per_month
        for gap in gap_pcts:
            effort_mult = 1 / (1 - gap / 100) ** eta_samples
            extra_days = (effort_mult - 1) * baseline_days
            kcal_loss = (1 - (1 - gap / 100) ** eta_samples) * 100
            rows.append({
                "group": label,
                "skill_gap_pct": gap,
                "baseline_effort": round(baseline_effort, 3),
                "baseline_days_per_month": round(baseline_days, 1),
                "headroom_days": round(headroom, 1),
                "kcal_loss_per_trip_mean": round(float(np.mean(kcal_loss)), 1),
                "kcal_loss_per_trip_q025": round(float(np.quantile(kcal_loss, 0.025)), 1),
                "kcal_loss_per_trip_q975": round(float(np.quantile(kcal_loss, 0.975)), 1),
                "extra_days_mean": round(float(np.mean(extra_days)), 1),
                "extra_days_q025": round(float(np.quantile(extra_days, 0.025)), 1),
                "extra_days_q975": round(float(np.quantile(extra_days, 0.975)), 1),
                "feasible": bool(np.mean(extra_days) <= headroom),
            })

        # Max compensable gap
        target_ratio = days_per_month / baseline_days
        max_gap = (1 - target_ratio ** (-1 / eta_samples)) * 100
        rows.append({
            "group": label,
            "skill_gap_pct": -1,  # sentinel for max compensable
            "baseline_effort": round(baseline_effort, 3),
            "baseline_days_per_month": round(baseline_days, 1),
            "headroom_days": round(headroom, 1),
            "kcal_loss_per_trip_mean": 0,
            "kcal_loss_per_trip_q025": 0,
            "kcal_loss_per_trip_q975": 0,
            "extra_days_mean": round(float(np.mean(max_gap)), 1),
            "extra_days_q025": round(float(np.quantile(max_gap, 0.025)), 1),
            "extra_days_q975": round(float(np.quantile(max_gap, 0.975)), 1),
            "feasible": True,
        })
        return rows

    # Overall
    baseline_all = float(per_forager_effort.median())
    all_rows = _compute_table(eta_mu, baseline_all, "all")

    # By gender
    for g_code, g_label in gender_map.items():
        foragers_g = [i for i in range(len(gender_idx)) if gender_idx[i] == g_code]
        efforts_g = per_forager_effort.reindex(foragers_g).dropna()
        baseline_g = float(efforts_g.median()) if len(efforts_g) > 0 else baseline_all
        all_rows.extend(_compute_table(eta_mu, baseline_g, g_label))

    df = pd.DataFrame(all_rows)
    df.to_csv(export_dir / "effort_skill_tradeoff.csv", index=False)

    # Also export eta summary by gender (eta_success varies)
    eta_success = _flatten_chains(post["eta_success"].values)  # (S, forager)
    male_idx = np.where(gender_idx == 0)[0]
    female_idx = np.where(gender_idx == 1)[0]

    eta_summary = {
        "eta_mu": {
            "mean": round(float(np.mean(eta_mu)), 3),
            "q025": round(float(np.quantile(eta_mu, 0.025)), 3),
            "q975": round(float(np.quantile(eta_mu, 0.975)), 3),
            "description": "Skill elasticity on per-trip returns (no gender interaction)",
        },
        "eta_success_male": {
            "mean": round(float(np.mean(eta_success[:, male_idx])), 3),
            "q025": round(float(np.quantile(eta_success[:, male_idx].mean(axis=1), 0.025)), 3),
            "q975": round(float(np.quantile(eta_success[:, male_idx].mean(axis=1), 0.975)), 3),
        },
        "eta_success_female": {
            "mean": round(float(np.mean(eta_success[:, female_idx])), 3),
            "q025": round(float(np.quantile(eta_success[:, female_idx].mean(axis=1), 0.025)), 3),
            "q975": round(float(np.quantile(eta_success[:, female_idx].mean(axis=1), 0.975)), 3),
        },
        "baseline_effort_all": round(baseline_all, 3),
    }

    with open(export_dir / "effort_skill_elasticities.json", "w") as f:
        json.dump(eta_summary, f, indent=2)

    print("  wrote effort_skill_tradeoff.csv + effort_skill_elasticities.json")


# =========================================================================
# 13. Shapley values
# =========================================================================


def export_shapley_values(
    idata: az.InferenceData,
    variant_dir: Path,
    export_dir: Path,
) -> None:
    import xarray as xr

    shapley_path = variant_dir / "shapley.nc"
    if not shapley_path.exists():
        warnings.warn(
            f"Shapley file not found at {shapley_path}, skipping"
        )
        return

    ds = xr.open_dataset(shapley_path)
    shap = ds["shapley_contribution"]
    vals = shap.values  # (sample, group, forager)

    # Flatten to per-group-forager rows
    rows = []
    if vals.ndim == 3:
        n_samples, n_groups, n_per_group = vals.shape
        for g in range(n_groups):
            for f in range(n_per_group):
                col = vals[:, g, f]
                if np.all(np.isnan(col)):
                    continue
                rows.append(
                    {
                        "group_idx": g,
                        "forager_idx": f,
                        "shapley_mean": float(np.nanmean(col)),
                        "shapley_sd": float(np.nanstd(col)),
                        "shapley_q025": float(np.nanquantile(col, 0.025)),
                        "shapley_q975": float(np.nanquantile(col, 0.975)),
                    }
                )
    elif vals.ndim == 2:
        n_groups, n_per_group = vals.shape
        for g in range(n_groups):
            for f in range(n_per_group):
                v = vals[g, f]
                if np.isnan(v):
                    continue
                rows.append(
                    {
                        "group_idx": g,
                        "forager_idx": f,
                        "shapley_mean": float(v),
                        "shapley_sd": 0.0,
                        "shapley_q025": float(v),
                        "shapley_q975": float(v),
                    }
                )

    if not rows:
        warnings.warn("No valid Shapley values found")
        return

    df_shap = pd.DataFrame(rows)

    # Enrich with metadata before saving
    cd = idata.constant_data
    age_raw = cd["age_raw"].values
    gender_idx = cd["gender_idx"].values
    group_date_idx = cd["group_date_idx"].values
    gender_map = {0: "male", 1: "female"}

    df_shap["date_idx"] = df_shap["group_idx"].map(
        lambda g: int(group_date_idx[g])
    )
    df_shap.to_csv(export_dir / "shapley_values.csv", index=False)

    def age_bin(age):
        if age < 10:
            return "<10"
        elif age < 20:
            return "10-20"
        return ">20"

    df_shap["age"] = df_shap["forager_idx"].map(lambda f: float(age_raw[f]))
    df_shap["gender"] = df_shap["forager_idx"].map(
        lambda f: gender_map.get(int(gender_idx[f]), "unknown")
    )
    df_shap["age_bin"] = df_shap["age"].map(age_bin)
    df_shap["category"] = df_shap["gender"] + " " + df_shap["age_bin"]

    # Build date labels from date_numeric (days since start)
    date_numeric = cd["date_numeric"].values  # (date,)
    start_date = pd.Timestamp("2018-07-23")
    date_labels = {
        int(i): (start_date + pd.Timedelta(days=int(d))).strftime("%b %d")
        for i, d in enumerate(date_numeric)
    }

    # Aggregate: sum shapley_mean per date x category
    area_data = (
        df_shap.groupby(["date_idx", "category"])["shapley_mean"]
        .sum()
        .reset_index()
    )
    area_data["date_label"] = area_data["date_idx"].map(date_labels)
    area_data.to_csv(export_dir / "shapley_area.csv", index=False)

    print("  wrote shapley_values.csv + shapley_area.csv")


# =========================================================================
# Main orchestration
# =========================================================================


def export_all(variant: str, skip_shapley: bool = False) -> None:
    variant_dir = RESULTS_DIR / variant
    idata_path = variant_dir / "idata.nc"
    config_path = variant_dir / "config.json"
    export_dir = variant_dir / "exports"

    if not idata_path.exists():
        print(f"Skipping {variant}: {idata_path} not found")
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {idata_path} ...")
    idata = az.from_netcdf(idata_path)
    config = (
        json.loads(config_path.read_text()) if config_path.exists() else {}
    )

    print(f"Exporting dashboard data for variant '{variant}':")

    export_manifest(idata, config, export_dir)
    export_estimates(idata, export_dir)
    export_prior_posterior(idata, export_dir)
    export_skill_curves(idata, export_dir)
    export_effort_curves(idata, export_dir)
    export_success_curves(idata, export_dir)
    export_diagnostics(idata, variant_dir, export_dir)
    export_trace_plots(idata, export_dir)
    export_pareto_k(idata, export_dir)
    export_psens(idata, export_dir)
    export_predictive_checks(idata, export_dir)
    export_observed_data(idata, export_dir)
    export_random_effects(idata, export_dir)
    export_effort_skill_tradeoff(idata, export_dir)

    # Copy model graph if generated at fit time
    model_graph_src = variant_dir / "model_graph.png"
    if model_graph_src.exists():
        import shutil
        shutil.copy2(model_graph_src, export_dir / "model_graph.png")
        print("  copied model_graph.png")

    if not skip_shapley:
        export_shapley_values(idata, variant_dir, export_dir)
    else:
        print("  skipping Shapley values (--skip-shapley)")

    print(f"Done. Exports written to {export_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dashboard data from fitted models"
    )
    parser.add_argument("--variant", help="Model variant to export")
    parser.add_argument(
        "--all", action="store_true", help="Export all variants"
    )
    parser.add_argument(
        "--skip-shapley",
        action="store_true",
        help="Skip Shapley value export",
    )
    args = parser.parse_args()

    if args.all:
        variants = [
            d.name
            for d in RESULTS_DIR.iterdir()
            if d.is_dir() and (d / "idata.nc").exists()
        ]
        for v in sorted(variants):
            export_all(v, skip_shapley=args.skip_shapley)
    else:
        export_all(
            args.variant or "base",
            skip_shapley=args.skip_shapley,
        )


if __name__ == "__main__":
    main()
