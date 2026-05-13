"""
Derive informative priors for the foraging skill curve and the
returns/success skill elasticities from the cross-cultural posteriors
in Koster et al. (2020, Science Advances).

Source: "The life history of human foraging: Cross-cultural and
individual variation." Pre-fit Stan model downloaded from:
https://osf.io/2kzb6/overview?view_only=682fcab2dd614dbdb015612b83044f49
File: model_fix_17092018.RData (590MB stanfit object)

The Koster model uses the same skill-curve functional form as ours:
    S(x) = exp(-m * x) * (1 - exp(-k * x))^b

where x = age / ref_age (ref_age = 80 in Koster, age_scale = 70 in ours).

Skill curve: Koster's `lifehistmeans[1:3]` are the cross-cultural means
of (log_k, log_m, log_b), and `sigma_societies[1:3]` are between-site
SDs. Rescale log_k, log_m by log(70/80). Prior sigma combines global
posterior SD and between-site SD (a new site like BaYaka could be
anywhere in the cross-cultural distribution).

Skill elasticities: Koster has per-society parameters
    sef[s] = log(exponent on S(x) in failure production)  ~ Normal(0, prior_scale)
    seh[s] = log(exponent on S(x) in harvest production)  ~ Normal(0, prior_scale)
with NO formal pooling across societies (each `sef[s]`, `seh[s]` has an
independent prior). We can still derive empirical cross-cultural priors
by pooling all (society x draw) posterior samples: the marginal mean
and SD of those pooled samples capture "what should we expect for the
elasticity in a new society." `seh` maps directly onto our `eta_mu0`
(same multiplicative form). `sef` maps onto our `eta_success0` by
interpretation (both control skill->success coupling), though the
link function differs (Koster uses 2*(1 - inv_logit(...)) for failure;
ours uses logit(p_success)).
"""
import numpy as np
import pandas as pd
import json

# --- Step 1: Load Koster posteriors (extracted from R) ---
df = pd.read_csv("/tmp/koster_posteriors.csv")
log_k_raw = df.log_k.values  # Koster posterior, ref_age=80
log_m_raw = df.log_m.values
log_b_raw = df.log_b.values

# Between-site SDs of skill curve params (from R extraction of sigma_societies)
between_site_sd = {"k": 0.251, "m": 0.162, "b": 0.300}

# Per-(society, draw) pooled draws of Koster's skill elasticities.
# These are the marginal cross-cultural distributions of `seh` (harvest
# elasticity) and `sef` (failure elasticity), both on the log scale.
# No global mean or between-site SD parameter exists in the Stan model
# for these — the empirical pooled mean and SD are what we want.
df_elast = pd.read_csv("/tmp/koster_elasticities.csv")
seh_raw = df_elast.seh.values  # log-scale exponent on S(x) in harvest production
sef_raw = df_elast.sef.values  # log-scale exponent on S(x) in failure production

# --- Step 2: Rescale from ref_age=80 to age_scale=70 ---
# k_new = k_old * (new_ref / old_ref) = k_old * (70/80)
# In log space: log_k_new = log_k_old + log(70/80) = log_k_old - log(80/70)
adj = np.log(70 / 80)  # ≈ -0.134
log_k = log_k_raw + adj
log_m = log_m_raw + adj
log_b = log_b_raw  # no adjustment needed

# --- Step 3: Global mean posterior summaries ---
global_mean = {
    "k0": float(np.mean(log_k)),
    "m0": float(np.mean(log_m)),
    "b0": float(np.mean(log_b)),
}
global_sd = {
    "k0": float(np.std(log_k)),
    "m0": float(np.std(log_m)),
    "b0": float(np.std(log_b)),
}

print("=" * 60)
print("Koster et al. (2020) global posteriors (log scale)")
print("Adjusted from ref_age=80 to age_scale=70")
print("=" * 60)
for p in ["k0", "m0", "b0"]:
    print(f"  {p}: mean={global_mean[p]:.3f}, sd={global_sd[p]:.3f}")

print(f"\nBetween-site SDs (log scale):")
for p, sd in between_site_sd.items():
    print(f"  {p}: {sd:.3f}")

# --- Step 4: Compute prior sigma ---
# sigma_prior² = sigma_global_mean² + sigma_between_site²
prior_sigma = {
    "k0": float(np.sqrt(global_sd["k0"]**2 + between_site_sd["k"]**2)),
    "m0": float(np.sqrt(global_sd["m0"]**2 + between_site_sd["m"]**2)),
    "b0": float(np.sqrt(global_sd["b0"]**2 + between_site_sd["b"]**2)),
}

print(f"\n{'=' * 60}")
print("PROPOSED INFORMATIVE PRIORS")
print("sigma = sqrt(global_mean_uncertainty² + between_site_SD²)")
print("=" * 60)
for p in ["m0", "k0", "b0"]:
    print(f"  {p}: Normal(mu={global_mean[p]:.2f}, sigma={prior_sigma[p]:.2f})")
    print(f"       components: global_sd={global_sd[p]:.3f}, "
          f"between_site_sd={between_site_sd[p[0]]:.3f}")

# --- Verify: implied peak age ---
m = np.exp(log_m)
k = np.exp(log_k)
b = np.exp(log_b)
x_peak = (1 / k) * np.log(1 + k * b / m)
peak_years = x_peak * 70

print(f"\nImplied peak age from global mean:")
print(f"  mean={np.mean(peak_years):.1f}y, "
      f"95% CI: {np.quantile(peak_years, 0.025):.1f}-"
      f"{np.quantile(peak_years, 0.975):.1f}")

# --- Also simulate from the proposed prior to verify peak age coverage ---
rng = np.random.default_rng(42)
n_sim = 10000
m0_sim = rng.normal(global_mean["m0"], prior_sigma["m0"], n_sim)
k0_sim = rng.normal(global_mean["k0"], prior_sigma["k0"], n_sim)
b0_sim = rng.normal(global_mean["b0"], prior_sigma["b0"], n_sim)
m_sim = np.exp(m0_sim)
k_sim = np.exp(k0_sim)
b_sim = np.exp(b0_sim)
# Clamp to avoid numerical issues
valid = (k_sim * b_sim / m_sim > 0)
x_peak_sim = np.where(valid, (1 / k_sim) * np.log(1 + k_sim * b_sim / m_sim), np.nan)
peak_sim = x_peak_sim * 70

print(f"\nPrior predictive peak age (sampling from proposed prior):")
print(f"  mean={np.nanmean(peak_sim):.1f}y, "
      f"median={np.nanmedian(peak_sim):.1f}y")
print(f"  95% CI: {np.nanquantile(peak_sim, 0.025):.1f}-"
      f"{np.nanquantile(peak_sim, 0.975):.1f}")
print(f"  (should encompass ~20-60 years to allow BaYaka-specific estimation)")

# --- Compare with current priors ---
print(f"\n{'=' * 60}")
print("COMPARISON WITH CURRENT PRIORS")
print("=" * 60)
current = {"m0": (-1.0, 1.0), "k0": (1.0, 1.0), "b0": (0.0, 1.0)}
for p in ["m0", "k0", "b0"]:
    cm, cs = current[p]
    print(f"  {p}: current Normal({cm:.1f}, {cs:.1f}) -> "
          f"proposed Normal({global_mean[p]:.2f}, {prior_sigma[p]:.2f})")

# --- Save as JSON for documentation ---
result = {
    "source": "Koster et al. 2020, Science Advances",
    "osf_url": "https://osf.io/2kzb6/",
    "model_file": "model_fix_17092018.RData",
    "ref_age_koster": 80,
    "age_scale_ours": 70,
    "adjustment": "log(80/70) added to log_k and log_m",
    "n_posterior_samples": len(df),
    "n_sites": 40,
    "n_individuals": 1821,
    "priors": {
        "m0": {
            "dist": "normal",
            "mu": round(global_mean["m0"], 2),
            "sigma": round(prior_sigma["m0"], 2),
            "description": "Senescence rate (log scale). "
                           "Center from cross-cultural global mean, "
                           "sigma from sqrt(global_posterior_sd² + between_site_sd²)",
        },
        "k0": {
            "dist": "normal",
            "mu": round(global_mean["k0"], 2),
            "sigma": round(prior_sigma["k0"], 2),
            "description": "Learning rate (log scale)",
        },
        "b0": {
            "dist": "normal",
            "mu": round(global_mean["b0"], 2),
            "sigma": round(prior_sigma["b0"], 2),
            "description": "Skill curve elasticity / power (log scale)",
        },
        "eta_mu0": {
            "dist": "normal",
            "mu": round(float(np.mean(seh_raw)), 2),
            "sigma": round(float(np.std(seh_raw)), 2),
            "description": "Returns skill elasticity (log scale). "
                           "From Koster's `seh` posterior (per-society, no "
                           "formal pooling), pooled across (society x draw). "
                           "Direct map: Koster's mu_harvest = lm_h * S(x)^exp(seh) "
                           "matches our mu_returns ∝ S(x)^exp(eta_mu0).",
        },
        "eta_success0": {
            "dist": "normal",
            "mu": round(float(np.mean(sef_raw)), 2),
            "sigma": round(float(np.std(sef_raw)), 2),
            "description": "Success skill elasticity (log scale). "
                           "From Koster's `sef` posterior, pooled across "
                           "(society x draw). Maps to our eta_success0 by "
                           "interpretation (skill->success/failure coupling); "
                           "link function differs (Koster: 2*(1-inv_logit(...)) "
                           "for failure; ours: logit(p_success)), but log-scale "
                           "exponent on S(x) and sign convention match.",
        },
    },
}
print(f"\n{'=' * 60}")
print("KOSTER-DERIVED ELASTICITY PRIORS")
print("=" * 60)
print(f"  eta_mu0      (returns):  Normal(mu={np.mean(seh_raw):+.2f}, sigma={np.std(seh_raw):.2f})")
print(f"  eta_success0 (success):  Normal(mu={np.mean(sef_raw):+.2f}, sigma={np.std(sef_raw):.2f})")

with open("foraging_model/cross_cultural_priors.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to foraging_model/cross_cultural_priors.json")
