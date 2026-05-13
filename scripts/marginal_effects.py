"""
Marginal effects of skill and effort on expected daily production.

For each forager, sweep re_skill or re_effort from -2 to +2 SD
while holding everything else at actual posterior values.
"""
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.special import expit

idata = az.from_netcdf("results/ln_nogp_meage_long/idata.nc")
post = idata.posterior
cd = idata.constant_data

# --- Extract and flatten posterior samples ---
def flat(x):
    return x.values.reshape(-1, *x.shape[2:])

# Subsample for speed
rng = np.random.default_rng(42)
n_sub = 300
idx = rng.choice(flat(post["eta_mu"]).shape[0], n_sub, replace=False)

re = flat(post["re"])[idx]              # (S, forager, 2)
S = flat(post["S"])[idx]                # (S, forager)
sigma = flat(post["chol_cov_stds"])[idx]  # (S, 2)
eta_mu = flat(post["eta_mu"])[idx]      # (S,)
intercept_mu = flat(post["intercept_mu"])[idx]
shape_param = flat(post["shape"])[idx]
eta_success = flat(post["eta_success"])[idx]    # (S, forager)
alpha_success = flat(post["alpha_success"])[idx]

# Effort model
eff_int = flat(post["effort_intercept"])[idx]
eff_b1 = flat(post["effort_age"])[idx]
eff_b2 = flat(post["effort_age2"])[idx]
eff_int_g = flat(post["effort_intercept_gender"])[idx]
eff_b1_g = flat(post["effort_age_gender"])[idx]
eff_b2_g = flat(post["effort_age2_gender"])[idx]
gp_eff = flat(post["gp_effort"])[idx]
gp_suc = flat(post["gp_success"])[idx]

age_raw = cd["age_raw"].values
gender_idx = cd["gender_idx"].values
n_foragers = len(age_raw)
age_mean, age_sd = np.mean(age_raw), np.std(age_raw)
age_z = (age_raw - age_mean) / age_sd

sigma_effort = sigma[:, 0]  # (S,)
sigma_skill = sigma[:, 1]
kcal_scale = float(post.attrs.get("kcal_scale", 2706.17))

# Percentage grid: 50% to 150% of actual
pct_grid = np.linspace(0.5, 1.5, 21)


def compute_annual_kcal(forager_i, skill_mult, effort_mult):
    """Compute expected annual kcal with skill and effort scaled by multipliers.

    skill_mult=1.0 means actual skill, 1.2 means 120% of actual.
    effort_mult=1.0 means actual effort probability, 1.2 means 120%.
    """
    g = int(gender_idx[forager_i])
    z = age_z[forager_i]

    # Actual effort logit
    re_eff_actual = re[:, forager_i, 0]
    base_logit = (
        eff_int + eff_int_g[:, g]
        + (eff_b1 + eff_b1_g[:, g]) * z
        + (eff_b2 + eff_b2_g[:, g]) * z**2
        + np.mean(gp_eff, axis=1)
    )
    p_effort_actual = expit(base_logit + re_eff_actual)
    # Scale effort probability directly, clamp to [0, 1]
    p_effort = np.clip(p_effort_actual * effort_mult, 0, 1)

    # Scale skill directly: S_new = S_actual * skill_mult
    S_actual = S[:, forager_i]
    S_new = np.clip(S_actual * skill_mult, 1e-10, 1)

    # Success
    gp_suc_mean = np.mean(gp_suc, axis=1)
    theta = expit(
        np.log(np.maximum(S_new**eta_success[:, forager_i] * alpha_success[:, forager_i], 1e-10))
        + gp_suc_mean
    )

    # Returns (solo)
    mu = np.exp(intercept_mu) * S_new**eta_mu
    e_kcal = mu * np.exp(shape_param**2 / 2) * kcal_scale

    annual = p_effort * 365 * theta * e_kcal
    return annual  # (S,)


# --- Sweep skill (hold effort at 100%) ---
skill_sweep = np.zeros((n_foragers, len(pct_grid)))
for j, pct in enumerate(pct_grid):
    for i in range(n_foragers):
        skill_sweep[i, j] = np.mean(compute_annual_kcal(i, skill_mult=pct, effort_mult=1.0))

# --- Sweep effort (hold skill at 100%) ---
effort_sweep = np.zeros((n_foragers, len(pct_grid)))
for j, pct in enumerate(pct_grid):
    for i in range(n_foragers):
        effort_sweep[i, j] = np.mean(compute_annual_kcal(i, skill_mult=1.0, effort_mult=pct))

# --- Normalize to % change vs actual (100%) ---
baseline_idx = np.argmin(np.abs(pct_grid - 1.0))
skill_baseline = skill_sweep[:, baseline_idx]
effort_baseline = effort_sweep[:, baseline_idx]

skill_pct_change = (skill_sweep / skill_baseline[:, None] - 1) * 100
effort_pct_change = (effort_sweep / effort_baseline[:, None] - 1) * 100

x_pct = (pct_grid - 1) * 100  # -50% to +50%

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Spaghetti
for i in range(n_foragers):
    ax.plot(x_pct, skill_pct_change[i], color="#3b82f6", alpha=0.12, lw=0.7)
    ax.plot(x_pct, effort_pct_change[i], color="#f97316", alpha=0.12, lw=0.7)

# Population means
skill_mean = np.mean(skill_pct_change, axis=0)
effort_mean = np.mean(effort_pct_change, axis=0)
ax.plot(x_pct, skill_mean, color="#1d4ed8", lw=2.5, label="Skill")
ax.plot(x_pct, effort_mean, color="#c2410c", lw=2.5, label="Effort")

ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("% change in skill or effort")
ax.set_ylabel("% change in annual kcal production")
ax.set_title("Marginal Returns to Skill vs. Effort")
ax.legend(fontsize=12)

fig.tight_layout()
fig.savefig("results/ln_nogp_meage_long/exports/marginal_effects.png", dpi=150)
plt.close(fig)
print("Saved marginal_effects.png")

# Print elasticities at 0% (finite difference around actual)
d = x_pct[1] - x_pct[0]
mid = baseline_idx
skill_elast = (skill_mean[mid+1] - skill_mean[mid-1]) / (2*d)
effort_elast = (effort_mean[mid+1] - effort_mean[mid-1]) / (2*d)
print(f"\nLocal elasticity (% change in annual kcal per 1% change in input):")
print(f"  Skill:  {skill_elast:.2f}")
print(f"  Effort: {effort_elast:.2f}")
print(f"  Ratio:  {skill_elast/effort_elast:.1f}x")
