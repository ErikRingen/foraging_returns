#!/usr/bin/env python3
"""Canonical script for per-forager kcal summary statistics.

Single source of truth for headline numbers reported in the manuscript and
supplement (kcal/foraging day, kcal/in-camp day, by gender, with and
without in-field consumption).

All numbers use:
- Shapley-attributed kcal from results/ln_nogp_meage_long/shapley.nc
  (per-forager, per-group attribution with the four Shapley axioms held
  within each group)
- Foraging-day denominators count any day with positive Shapley
  contribution (i.e., the forager appeared in at least one positive-return
  group that day)
- In-camp-day denominators come from raw_data/daysincamp.csv via
  preprocessing.preprocess_data
- Returns-only variants scale each per-group Shapley by
  returns_kcal / (returns + recall) for that group

Usage:
    pixi run python scripts/compute_kcal_summary.py

Outputs:
    results/kcal_summary.csv  -- the table in long format
    Console: pretty-printed summary tables
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import preprocess_data  # noqa: E402
from pyprojroot.here import here  # noqa: E402


def _drop_outright_gifts(df):
    """Drop rows recorded as outright gifts (gift == 1) from non-camp members.

    Partly gifted rows (gift == 0.5) are retained as recorded. Canonical for
    every analysis; the public dataset is built with the same rule.
    """
    gcol = next((c for c in df.columns if c.lower() == "gift"), None)
    if gcol is None:
        return df
    g = pd.to_numeric(df[gcol], errors="coerce").fillna(0)
    return df[g != 1].copy()


def _stream_packages(df, weight_col, kcal_lookup):
    """Compute deduplicated per-package kcal rows, per the data guide."""
    df = df.merge(kcal_lookup, on="index", how="left").copy()
    df["kcal_g"] = pd.to_numeric(df["kcal_g"], errors="coerce")
    df["kcal"] = df["kcal_g"] * df[weight_col]
    df["ID_str"] = df["ID"].astype(str)
    df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    is_solo = df["pooled_group"].isna() | (df["pooled_group"] == 0)
    solo = df[is_solo].copy()
    coop = df[~is_solo].drop_duplicates(["Date", "pooled_group"]).copy()
    return solo, coop


def _coop_gid_map(df):
    """Map (date, pooled_group) -> canonical group_id 'YYYY-MM-DD_<sorted_ids>'."""
    df = df.copy()
    df["ID_str"] = df["ID"].astype(str)
    df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
    coop = df[df["pooled_group"].notna() & (df["pooled_group"] > 0)]
    g = (
        coop.groupby(["date_str", "pooled_group"])["ID_str"]
        .agg(lambda x: "_".join(sorted(x.unique())))
        .reset_index()
        .rename(columns={"ID_str": "sorted_ids"})
    )
    g["gid"] = g["date_str"] + "_" + g["sorted_ids"]
    return g[["date_str", "pooled_group", "gid"]]


def _per_group_returns_share(group_str_ids, returns_df, recall_df, kcal_lookup):
    """For each group_id in the Shapley dataset, returns_kcal / total_kcal.

    Falls back to 1.0 when total is zero (zero-return groups don't appear
    in the Shapley file anyway, but be defensive).
    """
    ret_solo, ret_coop = _stream_packages(returns_df, "net_food_weight_gram", kcal_lookup)
    rec_solo, rec_coop = _stream_packages(recall_df, "total_weight_grams", kcal_lookup)
    ret_coop = ret_coop.merge(_coop_gid_map(returns_df), on=["date_str", "pooled_group"], how="left")
    rec_coop = rec_coop.merge(_coop_gid_map(recall_df), on=["date_str", "pooled_group"], how="left")

    # Solo group_ids: 'date_id'
    ret_solo["gid"] = ret_solo["date_str"] + "_" + ret_solo["ID_str"]
    rec_solo["gid"] = rec_solo["date_str"] + "_" + rec_solo["ID_str"]

    ret_per_gid = (
        pd.concat([ret_solo[["gid", "kcal"]], ret_coop[["gid", "kcal"]]])
        .groupby("gid")["kcal"].sum().to_dict()
    )
    rec_per_gid = (
        pd.concat([rec_solo[["gid", "kcal"]], rec_coop[["gid", "kcal"]]])
        .groupby("gid")["kcal"].sum().to_dict()
    )
    share = np.empty(len(group_str_ids))
    for i, gid in enumerate(group_str_ids):
        r = ret_per_gid.get(gid, 0.0)
        c = rec_per_gid.get(gid, 0.0)
        share[i] = r / (r + c) if (r + c) > 0 else 1.0
    return share


def _per_forager_totals(attr, forager_ids, dates_per_group, effort_days_by_for):
    """attr: (group, forager) -> per-forager (total_kcal, foraging_days).

    foraging_days uses the *effort* denominator: any day the forager went
    out (effort = 1), so failed trips are included (as 0 kcal). This
    matches the per-in-camp-day denominator's logic of counting zero-kcal
    days, applied to the foraging-day subset.
    """
    df = (
        pd.DataFrame(attr, columns=forager_ids)
        .assign(date=dates_per_group)
        .melt(id_vars="date", var_name="forager", value_name="kcal")
        .groupby(["forager", "date"], as_index=False)["kcal"].sum()
    )
    total = df.groupby("forager")["kcal"].sum().to_dict()
    return total, dict(effort_days_by_for)


def main() -> pd.DataFrame:
    shap = xr.load_dataset(here("results/ln_nogp_meage_long/shapley.nc"))
    forager_ids = [str(x) for x in shap["forager"].values]
    group_str_ids = [str(g) for g in shap["group"].values]
    attr = shap["shapley_contribution"].mean(dim="sample").values  # (group, forager)
    dates_per_group = np.array([g.split("_")[0] for g in group_str_ids])

    returns = _drop_outright_gifts(pd.read_csv(here("raw_data/returns.csv")))
    recall = _drop_outright_gifts(pd.read_csv(here("raw_data/recall.csv")))
    kcal = pd.read_csv(here("raw_data/kcal.csv")).rename(columns={"Index": "index", "kcal.g": "kcal_g"})
    returns["Date"] = pd.to_datetime(returns["Date"], format="mixed", dayfirst=True)
    recall["Date"] = pd.to_datetime(recall["Date"], format="mixed", dayfirst=True)
    recall = recall.rename(columns={"id": "ID"})

    share = _per_group_returns_share(group_str_ids, returns, recall, kcal[["index", "kcal_g"]])
    attr_returns_only = attr * share[:, None]

    df_for, df_time, _, df_days = preprocess_data(
        returns_file=here("raw_data/returns.csv"),
        recall_file=here("raw_data/recall.csv"),
        kcal_file=here("raw_data/kcal.csv"),
        group_file=here("raw_data/groups.csv"),
        camp_members_file=here("raw_data/camp_members.csv"),
        days_in_camp_file=here("raw_data/daysincamp.csv"),
        combine_returns_recall=True,
    )
    df_for["id"] = df_for["id"].astype(str)
    df_time["id"] = df_time["id"].astype(str)
    df_time["went_out"] = df_time["total.minutes"] > 0
    effort_days = (
        df_time[df_time["went_out"]]
        .groupby("id")["date"].nunique()
        .reindex(forager_ids).fillna(0).astype(int).to_dict()
    )

    tot_c, fd_c = _per_forager_totals(attr, forager_ids, dates_per_group, effort_days)
    tot_r, fd_r = _per_forager_totals(attr_returns_only, forager_ids, dates_per_group, effort_days)
    sex_map = dict(zip(df_for["id"], df_for["sex"]))
    df_days["forager_id"] = df_days["forager_id"].astype(str)
    df_days["date"] = pd.to_datetime(df_days["date"]).dt.strftime("%Y-%m-%d")
    camp = (
        df_days[df_days["in_camp"] == 1]
        .groupby("forager_id")["date"]
        .nunique()
        .to_dict()
    )

    per_for = pd.DataFrame([
        {
            "id": fid,
            "sex": sex_map.get(fid, "?"),
            "tot_combined": tot_c.get(fid, 0.0),
            "tot_returns": tot_r.get(fid, 0.0),
            "fd_combined": fd_c.get(fid, 0),
            "fd_returns": fd_r.get(fid, 0),
            "camp_days": camp.get(fid, 0),
        }
        for fid in forager_ids
    ])
    # The forager dimension carries one slot for a forager with no
    # daily-presence data and no positive food package; he does not enter
    # the likelihood and figures use n = 48, so drop him here too.
    per_for = per_for[
        (per_for["camp_days"] > 0) | (per_for["tot_combined"] > 0)
    ].reset_index(drop=True)

    rows = []
    for grp_lbl, mask in [
        ("All", np.ones(len(per_for), dtype=bool)),
        ("Male", per_for["sex"] == "male"),
        ("Female", per_for["sex"] == "female"),
    ]:
        sub = per_for[mask]
        rows.append({
            "group": grp_lbl,
            "n_foragers": int(len(sub)),
            "kcal_per_foraging_day_with_recall": (
                sub["tot_combined"].sum() / max(sub["fd_combined"].sum(), 1)
            ),
            "kcal_per_foraging_day_returns_only": (
                sub["tot_returns"].sum() / max(sub["fd_returns"].sum(), 1)
            ),
            "kcal_per_camp_day_with_recall": (
                sub["tot_combined"].sum() / max(sub["camp_days"].sum(), 1)
            ),
            "kcal_per_camp_day_returns_only": (
                sub["tot_returns"].sum() / max(sub["camp_days"].sum(), 1)
            ),
            "total_kcal_with_recall": float(sub["tot_combined"].sum()),
            "total_kcal_returns_only": float(sub["tot_returns"].sum()),
            "total_foraging_days_with_recall": int(sub["fd_combined"].sum()),
            "total_foraging_days_returns_only": int(sub["fd_returns"].sum()),
            "total_camp_days": int(sub["camp_days"].sum()),
        })
    summary = pd.DataFrame(rows)

    print("=== Per-forager kcal summary (Shapley basis) ===")
    print(
        summary[[
            "group", "n_foragers",
            "kcal_per_foraging_day_with_recall",
            "kcal_per_foraging_day_returns_only",
            "kcal_per_camp_day_with_recall",
            "kcal_per_camp_day_returns_only",
        ]].round(0).to_string(index=False)
    )

    out = here("results/kcal_summary.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved {out}")

    return summary


if __name__ == "__main__":
    main()
