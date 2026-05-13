#!/usr/bin/env python3
"""Parallel-coordinates: per-forager kcal share by resource, stratified by
within-gender skill tercile. Two panels (male, female)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyprojroot.here import here

# --- Build (forager × article) Shapley share matrix ---
shap = xr.load_dataset(here("results/ln_nogp_meage_long/shapley.nc"))
forager_ids = [str(x) for x in shap["forager"].values]
group_ids = [str(g) for g in shap["group"].values]
shap_mean = shap["shapley_contribution"].mean(dim="sample").values

returns = pd.read_csv(here("raw_data/returns.csv"))
returns["Date"] = pd.to_datetime(returns["Date"], format="%d.%m.%y", errors="coerce")
returns = returns.dropna(subset=["Date"])
returns["date_str"] = returns["Date"].dt.strftime("%Y-%m-%d")
package = (returns
    .groupby(["date_str", "pooled_group", "index"])
    .agg(article=("article", "first"),
         ids=("ID", lambda s: tuple(sorted(set(int(x) for x in s)))))
    .reset_index())
package["group_id"] = package.apply(
    lambda r: "_".join([r["date_str"]] + [str(i) for i in r["ids"]]), axis=1)
# Relabel "palm nut" -> "palm oil" for display (Yaka palm products are
# processed from the same Elaeis guineensis fruit; "palm oil" is the more
# accurate ethnographic label).
RENAME = {"palm nut": "palm oil"}
package["article"] = package["article"].replace(RENAME)
gid2article = package.groupby("group_id")["article"].first().to_dict()

articles_seen = set()
totals = {fid: {} for fid in forager_ids}
for g_idx, gid in enumerate(group_ids):
    a = gid2article.get(gid)
    if a is None: continue
    articles_seen.add(a)
    for f_idx, fid in enumerate(forager_ids):
        v = shap_mean[g_idx, f_idx]
        if np.isfinite(v) and v > 0:
            totals[fid][a] = totals[fid].get(a, 0.0) + float(v)

df_M = pd.DataFrame(0.0, index=forager_ids, columns=sorted(articles_seen))
for fid, d in totals.items():
    for a, v in d.items():
        df_M.loc[fid, a] = v

top_articles = list(df_M.sum(axis=0).sort_values(ascending=False).head(12).index)
df_top = df_M[top_articles].copy()
df_top["other"] = df_M.drop(columns=top_articles).sum(axis=1)
shares = df_top.div(df_top.sum(axis=1), axis=0).fillna(0)
cols = top_articles + ["other"]

# --- Per-forager skill (posterior mean) ---
idata = az.from_netcdf(here("results/ln_nogp_meage_long/idata.nc"))
S_full = idata.posterior["S"].mean(dim=("chain", "draw")).values  # (forager,)
gender_idx = idata.constant_data["gender_idx"].values             # 0=male, 1=female
sex = np.where(gender_idx == 0, "male", "female")
# Within-gender log-skill ranks for tercile assignment
log_S = np.log(S_full)
tercile = np.full(len(log_S), -1, dtype=int)
for g in (0, 1):
    m = gender_idx == g
    qs = np.quantile(log_S[m], [1/3, 2/3])
    tercile[m] = np.digitize(log_S[m], qs)  # 0 low, 1 mid, 2 high

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
x = np.arange(len(cols))

cmap = cm.get_cmap("viridis")
tercile_colors = [cmap(0.15), cmap(0.5), cmap(0.85)]
tercile_labels = ["Low skill", "Mid skill", "High skill"]

for ax, sex_label, title in [(axes[0], "male", "Male"), (axes[1], "female", "Female")]:
    sel = sex == sex_label
    fids = np.array(forager_ids)[sel]
    terc = tercile[sel]
    log_S_sel = log_S[sel]
    # normalise log-S within gender for colour
    norm = (log_S_sel - log_S_sel.min()) / (log_S_sel.max() - log_S_sel.min() + 1e-9)

    for fid, n in zip(fids, norm):
        ax.plot(x, shares.loc[fid, cols].values * 100,
                color=cmap(0.1 + 0.8 * n), alpha=0.25, lw=0.9)

    for t, color, lbl in zip([0, 1, 2], tercile_colors, tercile_labels):
        m = terc == t
        if not m.any(): continue
        mean_share = shares.loc[fids[m], cols].mean(axis=0).values * 100
        ax.plot(x, mean_share, color=color, lw=3.0,
                marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.0,
                label=lbl, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_xlabel("Resource (top-12 + other)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              title="Within-gender", title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Share of forager's Shapley-attributed kcal (%)")
fig.suptitle("Per-forager resource composition by within-gender skill tercile",
             fontsize=11, x=0.02, ha="left")

plt.tight_layout()
out = here("results/figures/eda_parallel_resource_skill.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")

print("\nMean kcal share by within-gender skill tercile:")
for sex_label in ("male", "female"):
    print(f"\n{sex_label}:")
    sel = sex == sex_label
    fids = np.array(forager_ids)[sel]
    terc = tercile[sel]
    for t, lbl in enumerate(tercile_labels):
        m = terc == t
        if not m.any(): continue
        means = shares.loc[fids[m], cols].mean(axis=0) * 100
        print(f"  {lbl} (n={m.sum()}):")
        for c, v in means.items():
            print(f"    {c:<18} {v:5.1f}%")
