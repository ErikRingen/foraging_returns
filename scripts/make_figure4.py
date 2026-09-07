#!/usr/bin/env python3
"""
Figure 4 (main paper): residual effort-skill association psi, gender-stratified.

Two-panel:
  (A) Scatter of posterior mean per-forager random effects on the
      observable scale: difference in P(trip) from the age x gender
      expectation (x) vs ratio of skill to expected (y); per-forager 50%
      credible intervals as cross-bars.
  (B) Posterior densities of psi (effort-skill correlation) by gender,
      annotated with Pr(psi < 0).

Reads from the canonical model variant `ln_nogp_meage_long`. Random-effect
covariance is gender-stratified, so the model exposes
`chol_cov_male_corr` and `chol_cov_female_corr` rather than a pooled
`chol_cov_corr`.

Usage:
    pixi run python scripts/make_figure4.py
Output:
    results/figures/figure4_psi.png
"""
from __future__ import annotations

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.stats import gaussian_kde
from pyprojroot.here import here

CANONICAL = "ln_nogp_meage_long"
GENDER_LABELS = ["male", "female"]
DISPLAY = {"male": "Boys/men", "female": "Girls/women"}
COL_M = "#2E86AB"
COL_F = "#E94F37"


def _flat(da):
    """Flatten (chain, draw, ...) to (chain*draw, ...)."""
    return da.values.reshape(-1, *da.shape[2:])


def main():
    idata = az.from_netcdf(here(f"results/{CANONICAL}/idata.nc"))
    post = idata.posterior  # type: ignore[attr-defined]
    cd = idata.constant_data  # type: ignore[attr-defined]

    gender_idx = np.asarray(cd["gender_idx"].values, dtype=int)
    sex_per = np.array([GENDER_LABELS[g] for g in gender_idx])

    # --- Panel A inputs: per-forager observable-scale deviations ----------
    age_z = _flat(post["age_z"])
    S_base = _flat(post["S_base"])
    S_full = _flat(post["S"])
    ei = _flat(post["effort_intercept"])
    eig = _flat(post["effort_intercept_gender"])
    ea = _flat(post["effort_age"])
    eag = _flat(post["effort_age_gender"])
    ea2 = _flat(post["effort_age2"])
    ea2g = _flat(post["effort_age2_gender"])
    re_arr = _flat(post["re"])

    g = gender_idx[None, :]
    lp_pop = (
        ei[:, None]
        + np.take_along_axis(eig, g.repeat(eig.shape[0], axis=0), axis=1)
        + (ea[:, None]
           + np.take_along_axis(eag, g.repeat(eag.shape[0], axis=0), axis=1)) * age_z
        + (ea2[:, None]
           + np.take_along_axis(ea2g, g.repeat(ea2g.shape[0], axis=0), axis=1)) * age_z ** 2
    )
    lp_for = lp_pop + re_arr[:, :, 0]
    dP = expit(lp_for) - expit(lp_pop)        # (sample, forager) probability points
    skill_ratio = S_full / S_base              # (sample, forager) multiplicative

    dP_m = dP.mean(axis=0)
    dP_lo, dP_hi = np.percentile(dP, [25, 75], axis=0)
    sk_m = skill_ratio.mean(axis=0)
    sk_lo, sk_hi = np.percentile(skill_ratio, [25, 75], axis=0)

    # --- Panel B inputs: gender-stratified psi ----------------------------
    psi_male = post["chol_cov_male_corr"].values[..., 0, 1].flatten()
    psi_female = post["chol_cov_female_corr"].values[..., 0, 1].flatten()
    p_neg_m = float((psi_male < 0).mean())
    p_neg_f = float((psi_female < 0).mean())

    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # --- Panel A ---
    ax = axes[0]
    for sex, color in [("male", COL_M), ("female", COL_F)]:
        m = sex_per == sex
        xerr = np.vstack([dP_m[m] - dP_lo[m], dP_hi[m] - dP_m[m]]) * 100
        yerr = np.vstack([sk_m[m] - sk_lo[m], sk_hi[m] - sk_m[m]])
        ax.errorbar(dP_m[m] * 100, sk_m[m],
                    xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=color, alpha=0.4, lw=0.9, capsize=0, zorder=2)
        ax.scatter(dP_m[m] * 100, sk_m[m],
                   s=46, c=color, edgecolors="white", linewidth=0.6,
                   alpha=0.95, zorder=3)

    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Probability of foraging effort (difference from expected, pp)")
    ax.set_ylabel("Forager skill (ratio to expected)")
    ax.set_title("(A) Per-forager deviations from age × gender expectation\n"
                 "(error bars: posterior 50% CIs)",
                 fontsize=10, loc="left")
    ax.legend(handles=[
        Line2D([0], [0], color=COL_M, marker="o", lw=0, markersize=7,
               markeredgecolor="white", label=DISPLAY["male"]),
        Line2D([0], [0], color=COL_F, marker="o", lw=0, markersize=7,
               markeredgecolor="white", label=DISPLAY["female"]),
    ], loc="upper right", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # --- Panel B ---
    ax = axes[1]
    grid = np.linspace(-1, 1, 400)
    for color, psi_, lbl, p_neg in [
        (COL_M, psi_male, DISPLAY["male"], p_neg_m),
        (COL_F, psi_female, DISPLAY["female"], p_neg_f),
    ]:
        kde = gaussian_kde(psi_)
        dens = kde(grid)
        ax.fill_between(grid, dens, color=color, alpha=0.22)
        ax.plot(grid, dens, color=color, lw=2.0)
        ax.axvline(psi_.mean(), color=color, lw=1.8)

    for color, psi_, lbl, y_frac in [
        (COL_M, psi_male, DISPLAY["male"], 0.93),
        (COL_F, psi_female, DISPLAY["female"], 0.81),
    ]:
        p_neg = (psi_ < 0).mean()
        ax.text(0.97, y_frac,
                f"{lbl}: $\\Pr(\\psi<0) = {p_neg:.2f}$",
                transform=ax.transAxes, ha="right", va="top",
                color=color, fontsize=10, fontweight="bold")

    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"Effort–skill correlation $\psi$")
    ax.set_ylabel("Posterior density")
    ax.set_title("(B) Posterior of $\\psi$ by gender", fontsize=10, loc="left")
    ax.set_xlim(-1.08, 1.08)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    out = here("results/figures/figure4_psi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
