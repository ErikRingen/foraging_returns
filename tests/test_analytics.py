"""
Smoke tests for foraging_model.analytics — the canonical (forager, date)
dataset used by every figure script.

These tests exercise the full pipeline (preprocessing → model → Shapley
→ analytics) against the fitted canonical variant on real data, so they
require:
  - results/ln_nogp_meage_long/idata.nc
  - results/ln_nogp_meage_long/shapley.nc
  - raw_data/ accessible
If those aren't present, tests are skipped.

The goal is to catch silent regressions in how effort, success, and
kcal_attributed are derived (the failure mode addressed in commit
74a3303 was that `effort` was being inferred from positive Shapley
contributions only — these tests would have flagged it).
"""
from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr
from pyprojroot.here import here

sys.path.insert(0, str(here()))
from foraging_model.analytics import (
    build_forager_day_dataset,
    per_forager_day_long,
    per_forager_summary,
)

CANONICAL = "ln_nogp_meage_long"


def _have_canonical() -> bool:
    return (
        (Path(here(f"results/{CANONICAL}/idata.nc"))).exists()
        and (Path(here(f"results/{CANONICAL}/shapley.nc"))).exists()
        and Path(here("raw_data/returns.csv")).exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_canonical(),
    reason="canonical fit + raw data required",
)


@pytest.fixture(scope="module")
def ds() -> xr.Dataset:
    return build_forager_day_dataset(variant=CANONICAL)


@pytest.fixture(scope="module")
def idata() -> az.InferenceData:
    return az.from_netcdf(here(f"results/{CANONICAL}/idata.nc"))


def test_dimensions_and_metadata(ds):
    assert {"forager", "date"} <= set(ds.dims)
    assert {"in_camp", "effort", "success", "kcal_attributed", "n_packages"} <= set(ds.data_vars)
    assert ds["age"].dims == ("forager",)
    assert ds["sex"].dims == ("forager",)
    assert ds.attrs.get("variant") == CANONICAL
    assert (np.asarray(ds["sex"].values) != "").all(), "every forager has a sex"


def test_effort_is_strict_superset_of_success(ds):
    """A success day must also be an effort day."""
    success_implies_effort = (~ds["success"].values | ds["effort"].values).all()
    assert success_implies_effort


def test_effort_is_subset_of_in_camp(ds):
    """A forager can only have effort=True on a day they were in camp."""
    effort_implies_in_camp = (~ds["effort"].values | ds["in_camp"].values).all()
    assert effort_implies_in_camp


def test_success_implies_positive_kcal(ds):
    """If success=True then kcal_attributed > 0 (definition: success = effort ∧ kcal>0)."""
    kcal = ds["kcal_attributed"].values
    success = ds["success"].values
    assert (kcal[success] > 0).all()


def test_effort_count_matches_idata_within_tolerance(ds, idata):
    """Effort total agrees with idata up to a small acceptable discrepancy.

    Analytics derives effort from preprocessing's group-membership records;
    the model derives it from the time-allocation file. The two definitions
    match by construction except for a small number of edge cases (e.g.
    short-stay non-residents, in-camp-without-logged-time foragers in
    cooperative zero-kcal groups). We accept a ≤ 10-day tolerance and flag
    larger drifts as a regression.
    """
    total_effort = int(ds["effort"].sum().values)
    n_effort_in_idata = int(idata.observed_data["effort"].sum().values)
    assert abs(total_effort - n_effort_in_idata) <= 10, (
        f"analytics effort count ({total_effort}) drifted significantly "
        f"from idata effort count ({n_effort_in_idata})"
    )


def test_success_count_matches_idata_within_tolerance(ds, idata):
    total_success = int(ds["success"].sum().values)
    n_success_in_idata = int(idata.observed_data["non_zero_prod"].sum().values)
    assert abs(total_success - n_success_in_idata) <= 15, (
        f"analytics success count ({total_success}) drifted from idata "
        f"success count ({n_success_in_idata}) by more than tolerance"
    )


def test_per_forager_day_long_denominators(ds):
    """The three denominators are nested: success ⊆ effort ⊆ in_camp."""
    n_succ = len(per_forager_day_long(ds, "success"))
    n_eff = len(per_forager_day_long(ds, "effort"))
    n_inc = len(per_forager_day_long(ds, "in_camp"))
    assert n_succ <= n_eff <= n_inc


def test_per_forager_summary_columns_and_means(ds):
    """Summary table has expected columns and valid means for active foragers."""
    s = per_forager_summary(ds, "effort")
    assert {"forager", "age", "sex", "n_days", "mean_kcal", "sd_kcal", "cv"} <= set(s.columns)
    # Every forager with >1 effort day has a finite mean and SD
    multi = s[s["n_days"] > 1]
    assert np.isfinite(np.asarray(multi["mean_kcal"], dtype=float)).all()
    assert np.isfinite(np.asarray(multi["sd_kcal"], dtype=float)).all()
    # Some foragers may have zero effort days (e.g. non-resident with
    # in_camp=0 throughout); they are simply omitted from the summary.
    assert len(s) <= len(ds["forager"])


def test_per_sex_counts_balance(ds):
    """Sanity: gender counts match expected (25 male, 24 female in canonical)."""
    by_sex = {
        s: int((ds["sex"].values == s).sum())
        for s in ("male", "female")
    }
    assert by_sex == {"male": 25, "female": 24}
