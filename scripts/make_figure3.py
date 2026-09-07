#!/usr/bin/env python3
"""
Figure 3 (main paper): subsistence skill and effort across the lifespan.

Produces a dual-axis figure showing:
  - left y-axis: subsistence skill S(x) by age, by gender
  - right y-axis: P(subsistence trip | in camp) by age, by gender

Reads from the canonical model variant `ln_nogp_meage_long`.

Usage:
    pixi run python scripts/make_figure3.py
Outputs:
    results/figures/figure3_skill_effort.png
"""
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pyprojroot.here import here
from scipy.special import expit

sys.path.insert(0, str(here()))
from preprocessing import preprocess_data

CANONICAL = "ln_nogp_meage_long"
COLORS = {"male": "#2E86AB", "female": "#E94F37"}
LABELS = {"male": "boys/men", "female": "girls/women"}


def main():
    idata = az.from_netcdf(here(f"results/{CANONICAL}/idata.nc"))
    post = idata.posterior  # type: ignore[attr-defined]
    age_scale = float(post.attrs.get("age_scale", 70.0))

    df_foragers, _, _, _ = preprocess_data(
        returns_file=here("raw_data/returns.csv"),
        recall_file=here("raw_data/recall.csv"),
        kcal_file=here("raw_data/kcal.csv"),
        group_file=here("raw_data/groups.csv"),
        camp_members_file=here("raw_data/camp_members.csv"),
        days_in_camp_file=here("raw_data/daysincamp.csv"),
        combine_returns_recall=True,
    )

    raw_ages = np.asarray(df_foragers["age"].values, dtype=float)
    age_mean, age_sd = float(raw_ages.mean()), float(raw_ages.std())
    # Start at 2: the youngest forager in the model. Below that the curves are
    # extrapolation — under-2s were excluded as too young to participate.
    ages_plot = np.linspace(2, 70, 200)
    ages_scaled = ages_plot / age_scale
    ages_z = (ages_plot - age_mean) / age_sd

    # --- skill curves by gender ---
    m0 = post["m0"].values.reshape(-1)
    k0 = post["k0"].values.reshape(-1)
    b0 = post["b0"].values.reshape(-1)
    m0_g = post["m0_gender"].values.reshape(-1, 2)
    k0_g = post["k0_gender"].values.reshape(-1, 2)
    b0_g = post["b0_gender"].values.reshape(-1, 2)

    # --- effort curves by gender ---
    ei = post["effort_intercept"].values.reshape(-1)
    ea = post["effort_age"].values.reshape(-1)
    ea2 = post["effort_age2"].values.reshape(-1)
    ei_g = post["effort_intercept_gender"].values.reshape(-1, 2)
    ea_g = post["effort_age_gender"].values.reshape(-1, 2)
    ea2_g = post["effort_age2_gender"].values.reshape(-1, 2)

    skill_by = {}
    effort_by = {}
    for g_idx, sex in enumerate(["male", "female"]):
        m_g = np.exp(m0 + m0_g[:, g_idx])
        k_g = np.exp(k0 + k0_g[:, g_idx])
        b_g = np.exp(b0 + b0_g[:, g_idx])
        S_g = (
            np.exp(-np.outer(m_g, ages_scaled))
            * (1 - np.exp(-np.outer(k_g, ages_scaled))) ** b_g[:, None]
        )
        skill_by[sex] = S_g

        logit_p = (
            (ei + ei_g[:, g_idx])[:, None]
            + (ea + ea_g[:, g_idx])[:, None] * ages_z[None, :]
            + (ea2 + ea2_g[:, g_idx])[:, None] * (ages_z[None, :] ** 2)
        )
        effort_by[sex] = expit(logit_p)

    # --- observed scatter overlays ---
    # --- plot ---
    fig, ax_skill = plt.subplots(figsize=(8, 5))
    ax_eff = ax_skill.twinx()

    for sex in ["male", "female"]:
        c = COLORS[sex]
        S_g = skill_by[sex]
        ax_skill.plot(
            ages_plot, S_g.mean(axis=0), color=c, lw=2.2, label=f"Skill ({LABELS[sex]})"
        )
        ax_skill.fill_between(
            ages_plot,
            np.percentile(S_g, 2.5, axis=0),
            np.percentile(S_g, 97.5, axis=0),
            color=c, alpha=0.12,
        )

        p_g = effort_by[sex]
        ax_eff.plot(
            ages_plot, p_g.mean(axis=0), color=c, lw=2.2, ls="--",
            label=f"Effort ({LABELS[sex]})",
        )
        ax_eff.fill_between(
            ages_plot,
            np.percentile(p_g, 2.5, axis=0),
            np.percentile(p_g, 97.5, axis=0),
            color=c, alpha=0.08,
        )

    ax_skill.set_xlim(2, 70)
    ax_skill.set_xlabel("Age (years)")
    ax_skill.set_ylabel("Subsistence skill $S(x)$ (solid)")
    ax_eff.set_ylabel("P(subsistence trip | in camp) (dashed)")
    ax_eff.set_ylim(0, 1)

    handles_s, labels_s = ax_skill.get_legend_handles_labels()
    handles_e, labels_e = ax_eff.get_legend_handles_labels()
    ax_skill.legend(handles_s + handles_e, labels_s + labels_e, loc="lower right", fontsize=8)

    plt.tight_layout()
    out_dir = here("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure3_skill_effort.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
