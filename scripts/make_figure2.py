#!/usr/bin/env python3
"""
Figure 2 (main paper): per-forager daily Shapley-attributed kcal across
foraging days (effort = 1; failed trips count as zero kcal).

Strip plot of every (forager, day) observation, with a horizontal median
marker per forager. X-position = actual age (years), coloured by sex.
Reads the canonical forager-day dataset built by ``foraging_model.analytics``.

Usage:
    pixi run python scripts/make_figure2.py
Output:
    results/figures/figure2_shapley_by_forager.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from foraging_model.analytics import load_or_build, per_forager_day_long

COLORS = {"male": "#2E86AB", "female": "#E94F37"}
RNG = np.random.default_rng(0)


def jittered_age_positions(ages: np.ndarray, spread: float = 0.7) -> np.ndarray:
    """Place foragers at their integer age, jittering same-age foragers."""
    pos = np.empty_like(ages, dtype=float)
    for a in np.unique(ages):
        same = np.where(ages == a)[0]
        k = len(same)
        if k == 1:
            pos[same[0]] = a
        else:
            offsets = (np.arange(k) - (k - 1) / 2.0) * spread
            for o, idx in zip(offsets, same):
                pos[idx] = a + o
    return pos


def main():
    ds = load_or_build()
    df = per_forager_day_long(ds, denominator="effort")

    forager_ids = list(ds["forager"].values)
    ages = np.asarray(ds["age"].values, dtype=float)
    sex = np.asarray(ds["sex"].values)
    positions = jittered_age_positions(ages)

    fig, ax = plt.subplots(figsize=(13, 6))

    bar_half_width = 0.55
    point_jitter = 0.22

    for f_idx, fid in enumerate(forager_ids):
        vals = np.asarray(df.loc[df["forager"] == fid, "kcal"], dtype=float)
        if len(vals) == 0:
            continue
        c = COLORS.get(str(sex[f_idx]), "gray")
        x_centre = float(positions[f_idx])

        # Strip-plot points
        x_pts = x_centre + RNG.uniform(-point_jitter, point_jitter, size=len(vals))
        ax.scatter(x_pts, vals, s=14, c=c, alpha=0.45,
                   edgecolors="none", linewidths=0, zorder=2)

        # Mean marker: thick black bar with coloured fill for visibility
        # against both the dot cloud and any background gridlines.
        m = float(np.mean(vals))
        ax.plot([x_centre - bar_half_width, x_centre + bar_half_width],
                [m, m], color="black", lw=3.5,
                solid_capstyle="butt", zorder=4)
        ax.plot([x_centre - bar_half_width, x_centre + bar_half_width],
                [m, m], color=c, lw=2.0,
                solid_capstyle="butt", zorder=5)

    # Square-root y-axis compresses the single ~99k palm-nut event without
    # losing the zero baseline, so the bulk of the distribution is readable.
    ax.set_yscale("function", functions=(np.sqrt, lambda y: y ** 2))
    yticks = [0, 500, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:,}" for y in yticks])
    ax.set_ylim(0, 110_000)

    age_max = int(np.ceil(ages.max()))
    ax.set_xlim(-2, age_max + 2)
    ticks = list(range(0, age_max + 5, 5))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Forager age (years)")
    ax.set_ylabel("Daily Shapley-attributed production (kcal, $\\sqrt{\\cdot}$ axis)")

    legend = [
        Patch(facecolor=COLORS["male"], alpha=0.65, label="Male"),
        Patch(facecolor=COLORS["female"], alpha=0.65, label="Female"),
        Line2D([0], [0], color="gray", lw=2.0, label="Per-forager mean"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    out = here("results/figures/figure2_shapley_by_forager.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
