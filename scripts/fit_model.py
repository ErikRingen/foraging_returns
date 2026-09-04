#!/usr/bin/env python3
"""
Fit the foraging model and save results.

Usage:
    pixi run python scripts/fit_model.py
    pixi run python scripts/fit_model.py --name base
    pixi run python scripts/fit_model.py --name me_age --me-age

Outputs:
    results/{name}/idata.nc - Full InferenceData (posterior + predictive + log_likelihood)
    results/{name}/config.json - Configuration used for this fit
    results/{name}/model_summary.csv - ArviZ summary table
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymc as pm
import arviz as az
from pyprojroot.here import here

from preprocessing import preprocess_data
from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel
from foraging_model.priors import scale_priors, KOSTER_DERIVED

RANDOM_SEED = 42

# Oil-palm resource indices in raw_data/kcal.csv: Mbila palm nut under all
# processing conditions (5-8) plus Madi ma mbila red palm oil (9). Palm wine
# is a distinct resource and is not excluded.
PALM_INDICES = [5, 6, 7, 8, 9]
TUNE = 1000
DRAWS = 1000
CHAINS = 4
TARGET_ACCEPT = 0.95


def fit_model(
    *,
    name: str,
    aggregation_method: str,
    measurement_error_age: bool,
    use_gp: bool = True,
    force: bool = False,
    data_dir: str = "raw_data",
    foraging_only: bool = True,
    include_recall: bool = True,
    exclude_gifts: bool = False,
    exclude_palm: bool = False,
    exclude_top_palm_harvest: bool = False,
    prior_scale: float = 1.0,
    prior_scale_scope: str = "all",
) -> tuple:
    """Fit a single model configuration and save to results/{name}/."""
    results_dir = here("results") / name
    results_dir.mkdir(parents=True, exist_ok=True)
    idata_path = results_dir / "idata.nc"
    config_path = results_dir / "config.json"

    if idata_path.exists() and not force:
        print(f"[{name}] Cached idata found, skipping fit. Use --force to refit.")
        return az.from_netcdf(idata_path), None

    print("=" * 60)
    print(f"Foraging Model: '{name}'")
    print("=" * 60)

    config = {
        "name": name,
        "aggregation_method": aggregation_method,
        "measurement_error_age": measurement_error_age,
        "use_gp": use_gp,
        "foraging_only": foraging_only,
        "include_recall": include_recall,
        "exclude_gifts": exclude_gifts,
        "exclude_palm": exclude_palm,
        "exclude_top_palm_harvest": exclude_top_palm_harvest,
        "prior_scale": prior_scale,
        "prior_scale_scope": prior_scale_scope,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    print(f"\n[1/6] Loading and preprocessing data from {data_dir}/ ...")
    data_root = data_dir.rstrip("/") + "/"
    # `kcal.csv` is the resource→kcal lookup. It exists in raw_data/ but
    # not in public_data/ (resource names are stripped). Pass `None` when
    # absent; preprocess_data uses precomputed `kcal` columns instead.
    kcal_path = here(data_root + "kcal.csv")
    df_foragers, df_time_agg, df_production, df_days_long = preprocess_data(
        returns_file=here(data_root + "returns.csv"),
        recall_file=here(data_root + "recall.csv"),
        kcal_file=kcal_path if kcal_path.exists() else None,
        group_file=here(data_root + "groups.csv"),
        camp_members_file=here(data_root + "camp_members.csv"),
        days_in_camp_file=here(data_root + "daysincamp.csv"),
        combine_returns_recall=True,
        foraging_only=foraging_only,
        include_recall=include_recall,
        exclude_gifts=exclude_gifts,
        exclude_resource_indices=PALM_INDICES if exclude_palm else None,
        exclude_top_harvest_of=PALM_INDICES if exclude_top_palm_harvest else None,
    )

    print("[2/6] Building model dataset...")
    data = ForagingData(
        foragers_df=df_foragers,
        time_allocation_df=df_time_agg,
        production_df=df_production,
        days_in_camp_df=df_days_long,
        target_column="kcal",
        group_id_col="group_id",
        forager_id_col="id",
    )
    dataset = data.to_dataset()

    print("[3/6] Building PyMC model...")
    foraging_model = ForagingModel(
        data=dataset,
        aggregation_method=aggregation_method,
        target_scaling="mean",
        age_scaling="max",
        measurement_error_age=measurement_error_age,
        use_gp=use_gp,
        priors=(
            scale_priors(
                prior_scale,
                only=KOSTER_DERIVED if prior_scale_scope == "koster" else None,
            )
            if prior_scale != 1.0 else None
        ),
    )

    # Save model graph
    try:
        from pymc import model_to_graphviz
        graph = model_to_graphviz(foraging_model.model)
        graph.render(results_dir / "model_graph", format="png", cleanup=True)
        print("  Model graph saved to model_graph.png")
    except Exception as e:
        print(f"  Model graph export failed: {e}")

    print(f"  - {len(dataset.forager)} foragers")
    print(f"  - {len(dataset.group)} groups")
    print(f"  - {len(dataset.effort_obs)} effort observations")
    print(f"  - {len(dataset.forager_date)} success observations")

    print(f"\n[4/6] Fitting model (tune={TUNE}, draws={DRAWS}, chains={CHAINS})...")
    idata = foraging_model.fit(
        tune=TUNE,
        draws=DRAWS,
        chains=CHAINS,
        random_seed=RANDOM_SEED,
        target_accept=TARGET_ACCEPT,
        low_rank_modified_mass_matrix=True,
    )

    print("[5/8] Computing log-likelihood for LOO...")
    with foraging_model.model:
        pm.compute_log_likelihood(idata, extend_inferencedata=True)

    print("[6/8] Computing log-prior for psens...")
    with foraging_model.model:
        log_prior = pm.compute_log_prior(idata, extend_inferencedata=True)

    print("[7/8] Generating predictive samples...")
    with foraging_model.model:
        prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
        posterior_predictive = pm.sample_posterior_predictive(
            idata, random_seed=RANDOM_SEED
        )
        idata.extend(prior)
        idata.extend(posterior_predictive)

    foraging_model.rescale_predictive()

    print("Computing deterministics (S, mu)...")
    with foraging_model.model:
        det = pm.compute_deterministics(
            idata.posterior,
            var_names=["S", "S_base", "M", "K", "mu", "theta", "p_effort"],
        )
    for var in det.data_vars:
        idata.posterior[var] = det[var]

    idata.posterior.attrs["kcal_scale"] = foraging_model.kcal_scale
    idata.posterior.attrs["age_scale"] = foraging_model.age_scale

    print("[8/8] Computing Shapley values...")
    shapley_path = results_dir / "shapley.nc"
    shapley_ds = foraging_model.compute_shapley(
        idata, n_posterior_samples=100, random_seed=RANDOM_SEED,
    )
    shapley_ds.to_netcdf(shapley_path)
    print(f"Shapley values saved to {shapley_path}")

    print(f"\nSaving InferenceData to {idata_path}...")
    idata.to_netcdf(idata_path)

    print("Generating model summary...")
    summary = az.summary(
        idata,
        var_names=[var.name for var in foraging_model.model.free_RVs],
    )
    summary_path = results_dir / "model_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    divergences = idata.sample_stats.diverging.sum().values
    print(f"Divergences: {divergences}")
    rhat_max = summary["r_hat"].max()
    print(f"Max r-hat: {rhat_max:.3f}")
    ess_min = summary["ess_bulk"].min()
    print(f"Min ESS (bulk): {ess_min:.0f}")
    print("=" * 60)

    return idata, foraging_model


def main():
    parser = argparse.ArgumentParser(description="Fit foraging model.")
    parser.add_argument(
        "--name", default="base",
        help="Run name (used for results subdirectory). Default: 'base'.",
    )
    parser.add_argument(
        "--aggregation", default="best_top_k",
        choices=["best_top_k", "mean"],
        help="Skill aggregation method. Default: 'best_top_k'.",
    )
    parser.add_argument(
        "--me-age", action="store_true",
        help="Enable measurement error on age.",
    )
    parser.add_argument(
        "--no-gp", action="store_true",
        help="Use independent date RE instead of GP.",
    )
    parser.add_argument(
        "--tune", type=int, default=None, help="Tuning steps (default: script constant)",
    )
    parser.add_argument(
        "--draws", type=int, default=None, help="Draws per chain (default: script constant)",
    )
    parser.add_argument(
        "--target-accept", type=float, default=None, help="Target acceptance rate",
    )
    parser.add_argument(
        "--chains", type=int, default=None, help="Number of chains (default: script constant)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Refit even if cached idata.nc exists.",
    )
    parser.add_argument(
        "--data-dir", default="raw_data",
        help=("Directory holding the input CSVs. Default: 'raw_data' (full "
              "data with resource attribution). Use 'public_data' to refit "
              "from the anonymised public dataset (precomputed kcal column, "
              "no resource attribution)."),
    )
    parser.add_argument(
        "--all-outings", action="store_true",
        help=("Count every out-of-camp trip as effort, not just trips with "
              "activity type 'Foraging' (sensitivity variant)."),
    )
    parser.add_argument(
        "--no-recall", action="store_true",
        help="Exclude the recall (field-consumption) stream from production.",
    )
    parser.add_argument(
        "--exclude-palm", action="store_true",
        help="Exclude all oil-palm resources (raw data only).",
    )
    parser.add_argument(
        "--exclude-gifts", action="store_true",
        help=("Drop returns/recall rows with any gift component (gift > 0): "
              "food given by non-camp members."),
    )
    parser.add_argument(
        "--exclude-top-palm-harvest", action="store_true",
        help=("Exclude the single largest oil-palm harvest -- one date x "
              "contributor set, summed across packages (raw data only)."),
    )
    parser.add_argument(
        "--prior-scale", type=float, default=1.0,
        help="Widen all priors by this factor (e.g. 2.0 doubles prior SDs).",
    )
    parser.add_argument(
        "--prior-scale-scope", default="all", choices=["all", "koster"],
        help=("Which priors --prior-scale widens: 'all' (default) or 'koster' "
              "(only the Koster et al.-derived priors: m0, k0, b0, eta_mu0, "
              "eta_success0)."),
    )
    args = parser.parse_args()

    # Override globals if CLI args provided
    global TUNE, DRAWS, TARGET_ACCEPT, CHAINS
    if args.tune is not None:
        TUNE = args.tune
    if args.draws is not None:
        DRAWS = args.draws
    if args.target_accept is not None:
        TARGET_ACCEPT = args.target_accept
    if args.chains is not None:
        CHAINS = args.chains

    fit_model(
        name=args.name,
        aggregation_method=args.aggregation,
        measurement_error_age=args.me_age,
        use_gp=not args.no_gp,
        force=args.force,
        data_dir=args.data_dir,
        foraging_only=not args.all_outings,
        include_recall=not args.no_recall,
        exclude_gifts=args.exclude_gifts,
        exclude_palm=args.exclude_palm,
        exclude_top_palm_harvest=args.exclude_top_palm_harvest,
        prior_scale=args.prior_scale,
        prior_scale_scope=args.prior_scale_scope,
    )
    print("\nDone! Results saved to results/")


if __name__ == "__main__":
    main()
