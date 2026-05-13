"""
Canonical forager-day dataset for downstream analysis and figure generation.

Single source of truth for "what counts as an effort day", "in-camp day", and
how Shapley contributions are aggregated to per-forager-per-day kcal. Every
script and supplement chunk should consume this dataset rather than rebuild
its own attribution and effort logic.

Returned dataset has dims (forager, date) (and optionally a `sample` dim for
posterior-aware kcal). Variables:

  - in_camp           : bool (forager, date)
  - effort            : bool (forager, date) — went out on a subsistence trip
  - success           : bool (forager, date) — effort AND positive Shapley
  - kcal_attributed   : float (forager, date) — sum of positive Shapley
                        contributions across packages on that date,
                        averaged across posterior samples
  - kcal_attributed_samples : float (sample, forager, date) — same as above
                        but with posterior dim preserved (only present if
                        with_samples=True)
  - n_packages        : int (forager, date) — number of positive packages
                        the forager contributed to on that date

Forager-level coordinates: age, sex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from preprocessing import preprocess_data

DEFAULT_VARIANT = "ln_nogp_meage_long"


def _load_raw(variant: str):
    shap = xr.open_dataset(here(f"results/{variant}/shapley.nc"))
    df_foragers, _, df_prod, df_days_long = preprocess_data(
        returns_file=here("raw_data/returns.csv"),
        recall_file=here("raw_data/recall.csv"),
        kcal_file=here("raw_data/kcal.csv"),
        group_file=here("raw_data/groups.csv"),
        camp_members_file=here("raw_data/camp_members.csv"),
        days_in_camp_file=here("raw_data/daysincamp.csv"),
        combine_returns_recall=True,
    )
    return shap, df_foragers, df_prod, df_days_long


def build_forager_day_dataset(
    variant: str = DEFAULT_VARIANT,
    with_samples: bool = False,
) -> xr.Dataset:
    """Build the canonical (forager, date) dataset for the given model variant.

    Parameters
    ----------
    variant : str
        Model variant subdirectory under ``results/``.
    with_samples : bool
        If True, include a ``kcal_attributed_samples`` variable with the
        ``sample`` dim preserved (memory-heavier; only needed for
        posterior-uncertainty figures).
    """
    shap, df_foragers, df_prod, df_days_long = _load_raw(variant)

    forager_ids = [str(x) for x in shap["forager"].values]
    fid_to_idx = {fid: i for i, fid in enumerate(forager_ids)}
    shap_group_ids = [str(g) for g in shap["group"].values]
    gid_to_date = dict(zip(
        df_prod["group_id"].astype(str), df_prod["date"].astype(str)
    ))

    # All in-camp dates (the master date axis)
    df_days = df_days_long.copy()
    df_days["forager_id"] = df_days["forager_id"].astype(str)
    df_days["date"] = df_days["date"].astype(str)
    dates = sorted(df_days["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    n_for, n_dat = len(forager_ids), len(dates)

    # in_camp (forager, date)
    in_camp = np.zeros((n_for, n_dat), dtype=bool)
    for fid, d, ic in zip(df_days["forager_id"], df_days["date"], df_days["in_camp"]):
        if ic == 1 and fid in fid_to_idx and d in date_to_idx:
            in_camp[fid_to_idx[fid], date_to_idx[d]] = True

    # effort: every (forager, date) where the forager was a member of any
    # food-package group AND was in camp that day. df_prod['forager_ids']
    # is a set of int IDs per group row (positive-kcal groups + zero-kcal
    # failed-trip rows appended by preprocessing). The in-camp restriction
    # mirrors the model's effort definition (data.py: effort = in_camp ∧
    # went_foraging) so that the analytics dataset and the fitted idata
    # see the same set of effort observations.
    df_prod = df_prod.copy()
    df_prod["date"] = df_prod["date"].astype(str)
    effort = np.zeros((n_for, n_dat), dtype=bool)
    for fids, d in zip(df_prod["forager_ids"], df_prod["date"]):
        if d not in date_to_idx:
            continue
        d_i = date_to_idx[d]
        for fid in fids:
            f_i = fid_to_idx.get(str(fid))
            if f_i is not None:
                effort[f_i, d_i] = True
    # Restrict to in-camp days (drops the small number of forager-days
    # where membership in a group is recorded but in_camp=False — e.g.
    # short-stay non-residents or off-by-one date rows).
    effort = effort & in_camp

    # Posterior-mean Shapley attribution per (forager, date), summed across
    # packages. NaN slots in shap mean "not a member of that group".
    sample_mean = shap["shapley_contribution"].mean(dim="sample").values
    kcal_mean = np.zeros((n_for, n_dat), dtype=float)
    n_packages = np.zeros((n_for, n_dat), dtype=int)
    for g_idx, gid in enumerate(shap_group_ids):
        d = gid_to_date.get(gid)
        if d is None or d not in date_to_idx:
            continue
        d_i = date_to_idx[d]
        col = sample_mean[g_idx]
        for f_i in range(n_for):
            v = col[f_i]
            if np.isfinite(v) and v > 0:
                kcal_mean[f_i, d_i] += float(v)
                n_packages[f_i, d_i] += 1

    success = (effort & (kcal_mean > 0))

    # Forager metadata
    df_foragers = df_foragers.copy()
    df_foragers["id"] = df_foragers["id"].astype(str)
    age_lookup = dict(zip(df_foragers["id"], df_foragers["age"]))
    sex_lookup = dict(zip(df_foragers["id"], df_foragers["sex"]))
    age = np.array([
        float(age_lookup[fid]) if fid in age_lookup else np.nan
        for fid in forager_ids
    ])
    sex = np.array([
        str(sex_lookup[fid]) if fid in sex_lookup else ""
        for fid in forager_ids
    ])

    coords = {
        "forager": forager_ids,
        "date": dates,
        "age": ("forager", age),
        "sex": ("forager", sex),
    }
    data_vars = {
        "in_camp": (("forager", "date"), in_camp),
        "effort": (("forager", "date"), effort),
        "success": (("forager", "date"), success),
        "kcal_attributed": (("forager", "date"), kcal_mean),
        "n_packages": (("forager", "date"), n_packages),
    }

    if with_samples:
        sample_full = shap["shapley_contribution"].values  # (sample, group, forager)
        n_sam = sample_full.shape[0]
        kcal_samples = np.zeros((n_sam, n_for, n_dat), dtype=float)
        for g_idx, gid in enumerate(shap_group_ids):
            d = gid_to_date.get(gid)
            if d is None or d not in date_to_idx:
                continue
            d_i = date_to_idx[d]
            block = sample_full[:, g_idx, :]  # (sample, forager)
            mask = np.isfinite(block) & (block > 0)
            kcal_samples[:, :, d_i] += np.where(mask, block, 0.0)
        coords["sample"] = np.arange(n_sam)
        data_vars["kcal_attributed_samples"] = (
            ("sample", "forager", "date"), kcal_samples
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["variant"] = variant
    ds.attrs["description"] = (
        "Canonical forager-day dataset. effort=True iff forager was a member "
        "of any food package on that date (failed trips included). "
        "success=True iff effort AND positive Shapley kcal. "
        "kcal_attributed is the sum of positive posterior-mean Shapley "
        "contributions across packages on that date."
    )
    return ds


# ---------------------------------------------------------------------------
# Convenience views for figures
# ---------------------------------------------------------------------------

def per_forager_day_long(
    ds: xr.Dataset, denominator: str = "effort"
) -> pd.DataFrame:
    """Long-format DataFrame of (forager, date, kcal) restricted to a denominator.

    denominator
        - 'effort' : every effort-day (failed trips → 0 kcal)
        - 'in_camp': every in-camp day (effort=0 days also → 0 kcal)
        - 'success': only days with positive returns (rare; usually you don't
                     want this — but provided for completeness)
    """
    if denominator == "effort":
        mask = ds["effort"]
    elif denominator == "in_camp":
        mask = ds["in_camp"]
    elif denominator == "success":
        mask = ds["success"]
    else:
        raise ValueError(f"Unknown denominator: {denominator}")

    # Keep kcal only where mask is True; flatten to long form
    kcal = ds["kcal_attributed"].where(mask)
    df = kcal.to_dataframe(name="kcal").reset_index().dropna(subset=["kcal"])
    age_lookup = dict(zip(ds["forager"].values, ds["age"].values))
    sex_lookup = dict(zip(ds["forager"].values, ds["sex"].values))
    df["age"] = [float(age_lookup[f]) for f in df["forager"]]
    df["sex"] = [str(sex_lookup[f]) for f in df["forager"]]
    return df


def per_forager_summary(ds: xr.Dataset, denominator: str = "effort") -> pd.DataFrame:
    """Per-forager mean / SD / CV / n of daily kcal under a chosen denominator."""
    df = per_forager_day_long(ds, denominator=denominator)
    rows = []
    for fid, sub in df.groupby("forager"):
        vals = np.asarray(sub["kcal"], dtype=float)
        mean = float(vals.mean()) if len(vals) else np.nan
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        cv = (sd / mean) if (mean and mean > 0) else np.nan
        rows.append({
            "forager": str(fid),
            "age": float(sub["age"].iloc[0]),
            "sex": str(sub["sex"].iloc[0]),
            "n_days": int(len(vals)),
            "mean_kcal": mean,
            "sd_kcal": sd,
            "cv": cv,
        })
    return pd.DataFrame(rows).sort_values(by=["sex", "age"]).reset_index(drop=True)


def cache_path(variant: str = DEFAULT_VARIANT) -> Path:
    return Path(here(f"results/{variant}/forager_day.nc"))


def load_or_build(
    variant: str = DEFAULT_VARIANT,
    with_samples: bool = False,
    rebuild: bool = False,
) -> xr.Dataset:
    """Load cached dataset if present (and not stale), otherwise build & save."""
    path = cache_path(variant)
    has_samples_required = with_samples
    if path.exists() and not rebuild:
        ds = xr.open_dataset(path)
        if (not has_samples_required) or "kcal_attributed_samples" in ds.data_vars:
            return ds
        ds.close()
    ds = build_forager_day_dataset(variant=variant, with_samples=with_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds
