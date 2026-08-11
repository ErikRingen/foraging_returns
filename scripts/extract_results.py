#!/usr/bin/env python3
"""
Extract key results from fitted model for manuscript.

Usage:
    pixi run python scripts/extract_results.py
    pixi run python scripts/extract_results.py --variant base

Requires:
    results/{variant}/idata.nc (from fit_model.py)

Outputs:
    results/manuscript_numbers.json - Key posterior summaries for manuscript
    (or results/{variant}/manuscript_numbers.json when --variant specified)
"""

import argparse
import json
import numpy as np
import arviz as az
from pathlib import Path
from pyprojroot import here


def compute_hdi(samples, prob=0.95):
    """Compute highest density interval."""
    hdi = az.hdi(samples, hdi_prob=prob)
    return float(hdi[0]), float(hdi[1])


def _skill_curve_estimands(m_pop, k_pop, b_pop, age_scale):
    """Posterior peak skill age (years) and age at 50% of peak (years).

    Inputs are sample-shaped 1-D arrays of population-level skill-curve
    parameters (e.g. m_pop = exp(m0 + m0_gender_offset)). Returns two
    posterior arrays in **years** (rescaled by ``age_scale``).
    """
    # Analytical peak: dS/dx = 0 ⇒ x_peak = (1/k) log(1 + bk/m)
    safe = (m_pop > 1e-8) & (b_pop > 1e-8) & (k_pop > 1e-8)
    x_peak_scaled = np.where(safe, (1 / k_pop) * np.log1p(b_pop * k_pop / m_pop), np.nan)
    x_peak_years = x_peak_scaled * age_scale

    # Age at 50% of peak — numerical (vectorized grid search).
    ages = np.linspace(0.01, 1.0, 1000)
    S_curves = (
        np.exp(-np.outer(m_pop, ages))
        * (1 - np.exp(-np.outer(k_pop, ages))) ** b_pop[:, None]
    )
    peak_S = S_curves.max(axis=1, keepdims=True)
    above_half = S_curves >= peak_S / 2
    # Use argmax-on-bool to find first True index (or set NaN if none).
    has_any = above_half.any(axis=1)
    first_idx = above_half.argmax(axis=1)
    age_50_years = np.where(has_any, ages[first_idx] * age_scale, np.nan)
    return x_peak_years, age_50_years


def _summarize(name, samples):
    return {
        name: {
            "mean": float(np.nanmean(samples)),
            "hdi_low": float(np.nanpercentile(samples, 2.5)),
            "hdi_high": float(np.nanpercentile(samples, 97.5)),
        }
    }


def extract_skill_curve_results(idata):
    """Extract population-level skill curve parameters and derived quantities.

    The skill curve depends on (m, k, b) which in the canonical model have
    a per-forager dimension because of gender offsets:
        m_i = exp(m0 + m0_gender[gender(i)])
    Reporting `posterior['m'].values.flatten()` would mix posterior
    uncertainty with between-forager (gender) variation. Here we instead
    work on `m0`, `k0`, `b0` plus the gender offsets to give clean
    population (`pooled`) and per-sex estimands.
    """
    posterior = idata.posterior
    kcal_scale = posterior.attrs.get('kcal_scale', 1.0)
    age_scale = posterior.attrs.get('age_scale', 1.0)

    # Log-scale population parameters → linear scale via exp
    m0 = posterior['m0'].values.flatten()
    k0 = posterior['k0'].values.flatten()
    b0 = posterior['b0'].values.flatten()

    out = {}
    out.update(_summarize("m0", m0))
    out.update(_summarize("k0", k0))
    out.update(_summarize("b0", b0))

    m_pop = np.exp(m0)
    k_pop = np.exp(k0)
    b_pop = np.exp(b0)
    out.update(_summarize("m", m_pop))
    out.update(_summarize("k", k_pop))
    out.update(_summarize("b", b_pop))

    pooled_peak, pooled_half = _skill_curve_estimands(m_pop, k_pop, b_pop, age_scale)
    out["peak_skill_age_years"] = _summarize("peak_skill_age_years", pooled_peak)["peak_skill_age_years"]
    out["age_50pct_skill_years"] = _summarize("age_50pct_skill_years", pooled_half)["age_50pct_skill_years"]

    # Post-peak retention: S(age)/S(peak), quantifying how shallow the
    # senescent decline is (cf. hunting-only curves, Koster et al. 2020).
    x_peak = (1 / k_pop) * np.log1p(b_pop * k_pop / m_pop)
    S_peak = np.exp(-m_pop * x_peak) * (1 - np.exp(-k_pop * x_peak)) ** b_pop
    for age_years in (60, 70):
        x = age_years / age_scale
        S_at = np.exp(-m_pop * x) * (1 - np.exp(-k_pop * x)) ** b_pop
        key = f"skill_at_{age_years}_vs_peak"
        out[key] = _summarize(key, S_at / S_peak)[key]

    # Per-sex estimands (gender coord order in posterior is [male, female])
    if "m0_gender" in posterior.data_vars:
        gender_coord = list(posterior["gender"].values)
        m0_g = posterior["m0_gender"].values.reshape(-1, len(gender_coord))
        k0_g = posterior["k0_gender"].values.reshape(-1, len(gender_coord))
        b0_g = posterior["b0_gender"].values.reshape(-1, len(gender_coord))
        per_sex = {}
        for g_idx, sex_name in enumerate(gender_coord):
            m_g = np.exp(m0 + m0_g[:, g_idx])
            k_g = np.exp(k0 + k0_g[:, g_idx])
            b_g = np.exp(b0 + b0_g[:, g_idx])
            peak, half = _skill_curve_estimands(m_g, k_g, b_g, age_scale)
            per_sex[str(sex_name)] = {
                "peak_skill_age_years": _summarize("x", peak)["x"],
                "age_50pct_skill_years": _summarize("x", half)["x"],
            }
        out["per_sex"] = per_sex

    out["age_scale"] = float(age_scale)
    out["kcal_scale"] = float(kcal_scale)
    return out


def extract_effort_results(idata):
    """Extract effort model parameters."""
    posterior = idata.posterior

    intercept = posterior['effort_intercept'].values.flatten()
    age_coef = posterior['effort_age'].values.flatten()
    age2_coef = posterior['effort_age2'].values.flatten()

    # Age at peak participation probability, in years. The effort linear
    # predictor is quadratic in z-scored age (z-constants from observed ages,
    # see model.py), so the vertex -b1/(2 b2) maps back through them.
    age_raw = idata.constant_data['age_raw'].values
    age_mean, age_std = float(np.mean(age_raw)), float(np.std(age_raw))
    peak_years = np.where(
        age2_coef < 0,
        age_mean + (-age_coef / (2 * age2_coef)) * age_std,
        np.nan,
    )

    return {
        **_summarize("effort_peak_age_years", peak_years),
        "effort_intercept": {
            "mean": float(np.mean(intercept)),
            "hdi_low": float(compute_hdi(intercept)[0]),
            "hdi_high": float(compute_hdi(intercept)[1]),
        },
        "effort_age": {
            "mean": float(np.mean(age_coef)),
            "hdi_low": float(compute_hdi(age_coef)[0]),
            "hdi_high": float(compute_hdi(age_coef)[1]),
        },
        "effort_age2": {
            "mean": float(np.mean(age2_coef)),
            "hdi_low": float(compute_hdi(age2_coef)[0]),
            "hdi_high": float(compute_hdi(age2_coef)[1]),
        },
    }


def extract_random_effect_correlation(idata):
    """Extract gender-stratified effort-skill correlation and RE scales."""
    posterior = idata.posterior

    def _summarise(corr_var, stds_var):
        c = posterior[corr_var].values[..., 0, 1].flatten()
        s = posterior[stds_var].values
        sigma_e = s[..., 0].flatten()
        sigma_s = s[..., 1].flatten()
        return {
            "effort_skill_correlation": {
                "mean": float(np.mean(c)),
                "hdi_low": float(compute_hdi(c)[0]),
                "hdi_high": float(compute_hdi(c)[1]),
                "prob_negative": float(np.mean(c < 0)),
            },
            "sigma_effort": {
                "mean": float(np.mean(sigma_e)),
                "hdi_low": float(compute_hdi(sigma_e)[0]),
                "hdi_high": float(compute_hdi(sigma_e)[1]),
            },
            "sigma_skill": {
                "mean": float(np.mean(sigma_s)),
                "hdi_low": float(compute_hdi(sigma_s)[0]),
                "hdi_high": float(compute_hdi(sigma_s)[1]),
            },
        }

    if "chol_cov_male_corr" in posterior:
        return {
            "male": _summarise("chol_cov_male_corr", "chol_cov_male_stds"),
            "female": _summarise("chol_cov_female_corr", "chol_cov_female_stds"),
        }
    if "chol_cov_corr" in posterior:  # legacy pooled fits
        return {"pooled": _summarise("chol_cov_corr", "chol_cov_stds")}
    return {}


def extract_returns_results(idata):
    """Extract LogNormal returns-component parameters.

    Note: the posterior variable ``shape`` is the LogNormal scale (sigma)
    parameter, kept under that name for idata.nc backward compatibility.
    Output JSON exposes it as ``kcal_log_sigma`` (the actual semantic).
    """
    posterior = idata.posterior

    intercept_mu = posterior['intercept_mu'].values.flatten()
    b_groupsize = posterior['b_groupsize_mu'].values.flatten()
    eta_mu = posterior['eta_mu'].values.flatten()
    sigma_kcal = posterior['shape'].values.flatten()

    return {
        "intercept_mu": {
            "mean": float(np.mean(intercept_mu)),
            "hdi_low": float(compute_hdi(intercept_mu)[0]),
            "hdi_high": float(compute_hdi(intercept_mu)[1]),
        },
        "b_groupsize_mu": {
            "mean": float(np.mean(b_groupsize)),
            "hdi_low": float(compute_hdi(b_groupsize)[0]),
            "hdi_high": float(compute_hdi(b_groupsize)[1]),
        },
        "eta_mu": {
            "mean": float(np.mean(eta_mu)),
            "hdi_low": float(compute_hdi(eta_mu)[0]),
            "hdi_high": float(compute_hdi(eta_mu)[1]),
        },
        "kcal_log_sigma": {
            "mean": float(np.mean(sigma_kcal)),
            "hdi_low": float(compute_hdi(sigma_kcal)[0]),
            "hdi_high": float(compute_hdi(sigma_kcal)[1]),
            "note": "LogNormal sigma; stored as 'shape' in idata for legacy reasons.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="ln_nogp_meage_long")
    parser.add_argument(
        "--output", default=None,
        help=("Output JSON path. Default: results/manuscript_numbers.json "
              "(canonical). Use e.g. results/<variant>/manuscript_numbers.json "
              "for sensitivity variants."),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Extracting Results for Manuscript")
    print("=" * 60)

    # Load idata
    idata_path = here("results") / args.variant / "idata.nc"
    if not idata_path.exists():
        print(f"Error: {idata_path} not found.")
        print("Run: pixi run python scripts/fit_model.py --name ln_nogp_meage_long --me-age --no-gp --tune 1000 --draws 5000 --chains 6 --target-accept 0.95")
        return

    print(f"Loading {idata_path}...")
    idata = az.from_netcdf(idata_path)
    
    # Extract results
    print("Extracting skill curve parameters...")
    skill_results = extract_skill_curve_results(idata)
    
    print("Extracting effort parameters...")
    effort_results = extract_effort_results(idata)
    
    print("Extracting random effect correlations...")
    re_results = extract_random_effect_correlation(idata)
    
    print("Extracting returns parameters...")
    returns_results = extract_returns_results(idata)
    
    # Combine all results
    results = {
        "skill_curve": skill_results,
        "effort": effort_results,
        "random_effects": re_results,
        "returns": returns_results,
    }
    
    # Save to JSON (default: results/manuscript_numbers.json)
    output_path = (
        here(args.output) if args.output else here("results/manuscript_numbers.json")
    )
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print key findings for manuscript
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR MANUSCRIPT")
    print("=" * 60)
    
    print("\n1. FORAGING SKILL INCREASES WITH AGE")
    sc = skill_results
    print(f"   Peak skill age: {sc['peak_skill_age_years']['mean']:.1f} years "
          f"(95% CI: {sc['peak_skill_age_years']['hdi_low']:.1f}-{sc['peak_skill_age_years']['hdi_high']:.1f})")
    print(f"   Age at 50% skill: {sc['age_50pct_skill_years']['mean']:.1f} years "
          f"(95% CI: {sc['age_50pct_skill_years']['hdi_low']:.1f}-{sc['age_50pct_skill_years']['hdi_high']:.1f})")
    
    print("\n2. FORAGING EFFORT INCREASES WITH AGE")
    ef = effort_results
    print(f"   Age effect (linear): {ef['effort_age']['mean']:.2f} "
          f"(95% CI: {ef['effort_age']['hdi_low']:.2f}-{ef['effort_age']['hdi_high']:.2f})")
    print(f"   Age effect (quadratic): {ef['effort_age2']['mean']:.2f} "
          f"(95% CI: {ef['effort_age2']['hdi_low']:.2f}-{ef['effort_age2']['hdi_high']:.2f})")
    
    print("\n3. EFFORT-SKILL CORRELATION (gender-stratified)")
    for g, gr in re_results.items():
        c = gr["effort_skill_correlation"]
        print(f"   {g}: psi = {c['mean']:.2f} "
              f"(95% CI: {c['hdi_low']:.2f}-{c['hdi_high']:.2f}), "
              f"P(<0) = {c['prob_negative']:.1%}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
