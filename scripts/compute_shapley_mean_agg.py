#!/usr/bin/env python3
"""Compute Shapley contributions for the avg-aggregation fit using the
TRUE mean-aggregation value function (not best-top-k as the engine
in foraging_model/counterfactuals.py uses).

Saves results to results/ln_nogp_meage_avg/shapley_mean_agg.nc.
"""
import numpy as np, xarray as xr, arviz as az
from itertools import combinations
from math import factorial
from pathlib import Path
from pyprojroot.here import here

avg = az.from_netcdf(here("results/ln_nogp_meage_avg/idata.nc"))
post = avg.posterior

# ---- Posterior subsample (mean Shapley is what we want) ---------
N_SUB = 100  # representative posterior subsample
rng = np.random.default_rng(0)
S_x = post["S"].stack(sample=("chain","draw")).transpose("sample","forager").values
imu = post["intercept_mu"].stack(sample=("chain","draw")).values
bgs = post["b_groupsize_mu"].stack(sample=("chain","draw")).values
emu = post["eta_mu"].stack(sample=("chain","draw")).values
shape_post = post["shape"].stack(sample=("chain","draw")).values
gp_returns = post["gp_returns"].stack(sample=("chain","draw")).transpose("sample","date").values
sub_idx = rng.choice(S_x.shape[0], N_SUB, replace=False)
S_x, imu, bgs, emu, shape_post, gp_returns = (
    S_x[sub_idx], imu[sub_idx], bgs[sub_idx], emu[sub_idx],
    shape_post[sub_idx], gp_returns[sub_idx],
)
print(f"Using {N_SUB} posterior draws.")

kcal_scale = float(post.attrs.get("kcal_scale", 1.0))
fids_padded = avg.constant_data["forager_ids"].values  # (group, slot)
group_date_idx = avg.constant_data["group_date_idx"].values  # (group,)
y_obs = np.asarray(avg.observed_data["kcal"].values, dtype=float)
n_groups, n_slots = fids_padded.shape
n_foragers = S_x.shape[1]

# ---- Mean-aggregation value function ----------------------------
def v_mean(skills_in_subset, k, imu, bgs, emu):
    """skills_in_subset: (n_samples, k); k: scalar; imu/bgs/emu: (n_samples,)
    returns (n_samples,)"""
    if k == 0:
        return np.zeros(imu.shape[0])
    mean_skill = skills_in_subset.mean(axis=1)
    g_k = np.exp(imu + bgs * np.log(k))
    return g_k * (mean_skill + 1e-12) ** emu

def shapley_exact_mean(members, S_x, imu, bgs, emu):
    """members: tuple of forager indices. Returns (n_samples, n_members)."""
    n = len(members)
    if n == 0:
        return np.zeros((imu.shape[0], 0))
    fact = factorial(n)
    # Cache v(S) for each subset
    cache = {}
    skills_g = S_x[:, list(members)]   # (n_samples, n)
    for size in range(n + 1):
        for sub in combinations(range(n), size):
            if not sub:
                cache[frozenset(sub)] = np.zeros(imu.shape[0])
                continue
            cache[frozenset(sub)] = v_mean(skills_g[:, list(sub)], len(sub),
                                            imu, bgs, emu)
    sh = np.zeros((imu.shape[0], n))
    # Marginal contributions
    for i in range(n):
        others = [x for x in range(n) if x != i]
        for size in range(len(others) + 1):
            w = factorial(size) * factorial(n - size - 1) / fact
            for sub in combinations(others, size):
                T = frozenset(sub); T1 = T | {i}
                sh[:, i] += w * (cache[T1] - cache[T])
    return sh

# ---- Loop over groups -------------------------------------------
shapley_array = np.full((N_SUB, n_groups, n_foragers), np.nan)

for g in range(n_groups):
    if y_obs[g] <= 0:                # only attribute positive groups
        continue
    members = tuple(int(x) for x in fids_padded[g] if x >= 0)
    if len(members) == 0: continue
    if len(members) > 14:
        # Too big for exact; skip with warning (very few)
        print(f"  group {g}: |members| = {len(members)}, skipping")
        continue

    sh = shapley_exact_mean(members, S_x, imu, bgs, emu)  # (S, k)

    # Apply per-group rescaling matching the existing engine
    # (1) kcal_scale, (2) date RE multiplier, (3) proportional residual
    sh = sh * kcal_scale
    sh = sh * np.exp(gp_returns[:, group_date_idx[g]])[:, None]
    sums = sh.sum(axis=1, keepdims=True)
    nonzero = (sums.squeeze() != 0)
    rescale = np.where(nonzero, y_obs[g] / sums.squeeze(), 0.0)
    sh = sh * rescale[:, None]

    for slot, fid in enumerate(members):
        shapley_array[:, g, fid] = sh[:, slot]
    if g % 200 == 0:
        print(f"  group {g}/{n_groups}")

ds_out = xr.Dataset(
    {"shapley_contribution": (("sample", "group", "forager"), shapley_array)},
    coords={
        "sample":  np.arange(N_SUB),
        "group":   avg.observed_data["kcal"].coords.get("group",
                       np.arange(n_groups)).values if "group" in avg.observed_data["kcal"].coords else np.arange(n_groups),
        "forager": np.arange(n_foragers),
    },
)

out = Path(here("results/ln_nogp_meage_avg/shapley_mean_agg.nc"))
ds_out.to_netcdf(out)
print(f"\nSaved {out}")

# Summary: per-forager total attribution
total = np.nansum(shapley_array.mean(axis=0), axis=0)
neg_share = (np.nansum(np.where(shapley_array.mean(axis=0) < 0,
                                  shapley_array.mean(axis=0), 0)) /
             np.nansum(np.abs(shapley_array.mean(axis=0))))
print(f"Total positive Shapley sum (mean-agg): {np.nansum(np.where(shapley_array.mean(axis=0)>0, shapley_array.mean(axis=0), 0)):,.0f}")
print(f"Total negative Shapley sum (mean-agg): {np.nansum(np.where(shapley_array.mean(axis=0)<0, shapley_array.mean(axis=0), 0)):,.0f}")
print(f"Total net (matches Σ y_obs):            {np.nansum(shapley_array.mean(axis=0)):,.0f}")
print(f"Σ y_obs:                                {y_obs.sum():,.0f}")
