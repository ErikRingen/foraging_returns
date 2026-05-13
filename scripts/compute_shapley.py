#!/usr/bin/env python3
"""
Compute Shapley values from a fitted model's idata.nc and save shapley.nc.

Thin wrapper around ``ForagingModel.compute_shapley`` so the same code
path runs at fit time (in ``fit_model.py``) and on cached idata.

Usage:
    pixi run python scripts/compute_shapley.py --variant ln_nogp_meage_long
"""
import argparse
import json
import sys

import arviz as az
from pyprojroot import here

sys.path.insert(0, str(here()))
from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel
from preprocessing import preprocess_data

RESULTS_DIR = here("results")
DEFAULT_VARIANT = "ln_nogp_meage_long"


def _load_config(variant_dir):
    cfg_path = variant_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    # Fall back to canonical config
    return {
        "aggregation_method": "best_top_k",
        "measurement_error_age": True,
        "use_gp": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    variant_dir = RESULTS_DIR / args.variant
    idata_path = variant_dir / "idata.nc"
    shapley_path = variant_dir / "shapley.nc"

    print(f"Loading {idata_path}...")
    idata = az.from_netcdf(idata_path)

    print("Building model from preprocessed data + config...")
    raw = "raw_data/"
    df_foragers, df_time_agg, df_production, df_days_long = preprocess_data(
        returns_file=here(raw + "returns.csv"),
        recall_file=here(raw + "recall.csv"),
        kcal_file=here(raw + "kcal.csv"),
        group_file=here(raw + "groups.csv"),
        camp_members_file=here(raw + "camp_members.csv"),
        days_in_camp_file=here(raw + "daysincamp.csv"),
        combine_returns_recall=True,
    )
    dataset = ForagingData(
        foragers_df=df_foragers,
        time_allocation_df=df_time_agg,
        production_df=df_production,
        days_in_camp_df=df_days_long,
        target_column="kcal",
        group_id_col="group_id",
        forager_id_col="id",
    ).to_dataset()

    cfg = _load_config(variant_dir)
    model = ForagingModel(
        data=dataset,
        aggregation_method=cfg.get("aggregation_method", "best_top_k"),
        target_scaling="mean",
        age_scaling="max",
        measurement_error_age=cfg.get("measurement_error_age", True),
        use_gp=cfg.get("use_gp", False),
    )

    print(f"Computing Shapley values ({args.n_samples} posterior samples)...")
    shapley_ds = model.compute_shapley(
        idata,
        n_posterior_samples=args.n_samples,
        random_seed=args.seed,
    )

    print(f"Saving to {shapley_path}...")
    shapley_ds.to_netcdf(shapley_path)
    print("Done.")


if __name__ == "__main__":
    main()
