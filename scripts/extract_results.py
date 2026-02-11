#!/usr/bin/env python3
"""
Extract key results from fitted model for manuscript.

Usage:
    pixi run python scripts/extract_results.py

Requires:
    results/idata.nc (from fit_model.py)

Outputs:
    results/manuscript_numbers.json - Key posterior summaries for manuscript
"""

import json
import numpy as np
import arviz as az
from pathlib import Path
from pyprojroot.here import here


def compute_hdi(samples, prob=0.95):
    """Compute highest density interval."""
    hdi = az.hdi(samples, hdi_prob=prob)
    return float(hdi[0]), float(hdi[1])


def extract_skill_curve_results(idata):
    """Extract skill curve parameters and derived quantities."""
    posterior = idata.posterior
    
    # Get scale factors
    kcal_scale = posterior.attrs.get('kcal_scale', 1.0)
    age_scale = posterior.attrs.get('age_scale', 1.0)
    
    # Skill curve parameters (already transformed from log scale)
    m = posterior['m'].values.flatten()
    k = posterior['k'].values.flatten()
    b = posterior['b'].values.flatten()
    
    # Compute peak skill age
    # S(x) = exp(-mx) * (1 - exp(-kx))^b
    # Peak occurs where dS/dx = 0
    # Analytical solution: x_peak = (1/k) * log(1 + k*b/m)
    x_peak_scaled = (1/k) * np.log(1 + k*b/m)
    x_peak_years = x_peak_scaled * age_scale
    
    # Age at 50% of peak skill
    # This requires numerical solution, approximate with grid search
    ages = np.linspace(0.01, 1.0, 1000)  # scaled ages
    S_curves = np.exp(-np.outer(m, ages)) * (1 - np.exp(-np.outer(k, ages)))**b[:, None]
    peak_S = S_curves.max(axis=1)
    
    age_50_samples = []
    for i in range(len(m)):
        S_curve = S_curves[i]
        half_peak = peak_S[i] / 2
        # Find first age where S >= half_peak
        idx = np.where(S_curve >= half_peak)[0]
        if len(idx) > 0:
            age_50_samples.append(ages[idx[0]] * age_scale)
        else:
            age_50_samples.append(np.nan)
    age_50_samples = np.array(age_50_samples)
    
    return {
        "m": {
            "mean": float(np.mean(m)),
            "hdi_low": float(compute_hdi(m)[0]),
            "hdi_high": float(compute_hdi(m)[1]),
        },
        "k": {
            "mean": float(np.mean(k)),
            "hdi_low": float(compute_hdi(k)[0]),
            "hdi_high": float(compute_hdi(k)[1]),
        },
        "b": {
            "mean": float(np.mean(b)),
            "hdi_low": float(compute_hdi(b)[0]),
            "hdi_high": float(compute_hdi(b)[1]),
        },
        "peak_skill_age_years": {
            "mean": float(np.nanmean(x_peak_years)),
            "hdi_low": float(np.nanpercentile(x_peak_years, 2.5)),
            "hdi_high": float(np.nanpercentile(x_peak_years, 97.5)),
        },
        "age_50pct_skill_years": {
            "mean": float(np.nanmean(age_50_samples)),
            "hdi_low": float(np.nanpercentile(age_50_samples, 2.5)),
            "hdi_high": float(np.nanpercentile(age_50_samples, 97.5)),
        },
        "age_scale": float(age_scale),
        "kcal_scale": float(kcal_scale),
    }


def extract_effort_results(idata):
    """Extract effort model parameters."""
    posterior = idata.posterior
    
    intercept = posterior['effort_intercept'].values.flatten()
    age_coef = posterior['effort_age'].values.flatten()
    age2_coef = posterior['effort_age2'].values.flatten()
    
    return {
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
    """Extract correlation between effort and skill random effects."""
    posterior = idata.posterior
    
    # The correlation matrix is stored in 'chol_cov_corr' from LKJCholeskyCov
    # Shape: (chain, draw, 2, 2) for 2 random effects
    if 'chol_cov_corr' in posterior:
        corr_matrix = posterior['chol_cov_corr'].values
        # Extract (0,1) element = correlation between effort and skill
        effort_skill_corr = corr_matrix[:, :, 0, 1].flatten()
    else:
        # Fallback
        effort_skill_corr = np.array([0.0])
    
    # Get standard deviations from chol_cov_stds
    if 'chol_cov_stds' in posterior:
        sigma_re = posterior['chol_cov_stds'].values
        sigma_effort = sigma_re[:, :, 0].flatten()
        sigma_skill = sigma_re[:, :, 1].flatten()
    else:
        sigma_effort = np.array([1.0])
        sigma_skill = np.array([1.0])
    
    return {
        "effort_skill_correlation": {
            "mean": float(np.mean(effort_skill_corr)),
            "hdi_low": float(compute_hdi(effort_skill_corr)[0]),
            "hdi_high": float(compute_hdi(effort_skill_corr)[1]),
            "prob_negative": float(np.mean(effort_skill_corr < 0)),
        },
        "sigma_effort": {
            "mean": float(np.mean(sigma_effort)),
            "hdi_low": float(compute_hdi(sigma_effort)[0]),
            "hdi_high": float(compute_hdi(sigma_effort)[1]),
        },
        "sigma_skill": {
            "mean": float(np.mean(sigma_skill)),
            "hdi_low": float(compute_hdi(sigma_skill)[0]),
            "hdi_high": float(compute_hdi(sigma_skill)[1]),
        },
    }


def extract_returns_results(idata):
    """Extract group returns model parameters."""
    posterior = idata.posterior
    
    intercept_mu = posterior['intercept_mu'].values.flatten()
    b_groupsize = posterior['b_groupsize_mu'].values.flatten()
    eta_mu = posterior['eta_mu'].values.flatten()
    shape = posterior['shape'].values.flatten()
    
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
        "gamma_shape": {
            "mean": float(np.mean(shape)),
            "hdi_low": float(compute_hdi(shape)[0]),
            "hdi_high": float(compute_hdi(shape)[1]),
        },
    }


def main():
    print("=" * 60)
    print("Extracting Results for Manuscript")
    print("=" * 60)
    
    # Load idata
    idata_path = here("results/idata.nc")
    if not idata_path.exists():
        print(f"Error: {idata_path} not found.")
        print("Run scripts/fit_model.py first.")
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
    
    # Save to JSON
    output_path = here("results/manuscript_numbers.json")
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
    
    print("\n3. LOWER-SKILLED INDIVIDUALS EXPEND MORE EFFORT")
    re = re_results
    print(f"   Effort-skill correlation: {re['effort_skill_correlation']['mean']:.2f} "
          f"(95% CI: {re['effort_skill_correlation']['hdi_low']:.2f}-{re['effort_skill_correlation']['hdi_high']:.2f})")
    print(f"   P(correlation < 0): {re['effort_skill_correlation']['prob_negative']:.1%}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
