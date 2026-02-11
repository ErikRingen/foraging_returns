#!/usr/bin/env python3
"""
Fit the foraging model and save results.

Usage:
    pixi run python scripts/fit_model.py

Outputs:
    results/idata.nc - Full InferenceData (posterior + predictive samples)
    results/model_summary.csv - ArviZ summary table
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymc as pm
import arviz as az
from pyprojroot.here import here

from preprocessing import preprocess_data
from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel

# Configuration
RANDOM_SEED = 42
TUNE = 1000
DRAWS = 1000
CHAINS = 4
TARGET_ACCEPT = 0.95

def main():
    print("=" * 60)
    print("Foraging Model Fitting Script")
    print("=" * 60)
    
    # Create results directory
    results_dir = here("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    raw_data_dir = "raw_data/"
    
    df_foragers, df_time_agg, df_production, df_days_long = preprocess_data(
        returns_file=here(raw_data_dir + 'returns.csv'),
        recall_file=here(raw_data_dir + 'recall.csv'),
        kcal_file=here(raw_data_dir + 'kcal.csv'),
        group_file=here(raw_data_dir + 'groups.csv'),
        camp_members_file=here(raw_data_dir + 'camp_members.csv'),
        days_in_camp_file=here(raw_data_dir + 'daysincamp.csv'),
        combine_returns_recall=True,
        output_dir=here('data')
    )
    
    # Build ForagingData and dataset
    print("[2/5] Building model dataset...")
    data = ForagingData(
        foragers_df=df_foragers,
        time_allocation_df=df_time_agg,
        production_df=df_production,
        days_in_camp_df=df_days_long,
        target_column='kcal',
        group_id_col='group_id',
        forager_id_col='id',
    )
    dataset = data.to_dataset()
    
    # Build model
    print("[3/5] Building PyMC model...")
    foraging_model = ForagingModel(
        data=dataset,
        target_scaling='mean',
        age_scaling='max',
    )
    
    # Print model info
    print(f"  - {len(dataset.forager)} foragers")
    print(f"  - {len(dataset.group)} groups")
    print(f"  - {len(dataset.effort_obs)} effort observations")
    print(f"  - {len(dataset.forager_date)} success observations")
    
    # Fit model
    print(f"\n[4/5] Fitting model (tune={TUNE}, draws={DRAWS}, chains={CHAINS})...")
    idata = foraging_model.fit(
        tune=TUNE,
        draws=DRAWS,
        chains=CHAINS,
        random_seed=RANDOM_SEED,
        target_accept=TARGET_ACCEPT,
        low_rank_modified_mass_matrix=True,
    )
    
    # Generate predictive samples
    print("[5/5] Generating predictive samples...")
    with foraging_model.model:
        prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
        posterior_predictive = pm.sample_posterior_predictive(
            idata, random_seed=RANDOM_SEED
        )
        idata.extend(prior)
        idata.extend(posterior_predictive)
    
    # Rescale predictive samples
    foraging_model.rescale_predictive()
    
    # Compute deterministics (S, mu, etc.) for all posterior samples
    print("Computing deterministics (S, mu)...")
    with foraging_model.model:
        det = pm.compute_deterministics(
            idata.posterior, 
            var_names=['S', 'S_base', 'M', 'K', 'mu', 'theta', 'p_effort']
        )
    # Merge deterministics into posterior
    for var in det.data_vars:
        idata.posterior[var] = det[var]
    
    # Store scale factors as attributes
    idata.posterior.attrs['kcal_scale'] = foraging_model.kcal_scale
    idata.posterior.attrs['age_scale'] = foraging_model.age_scale
    
    # Save idata
    idata_path = results_dir / "idata.nc"
    print(f"\nSaving InferenceData to {idata_path}...")
    idata.to_netcdf(idata_path)
    
    # Generate and save summary
    print("Generating model summary...")
    summary = az.summary(
        idata, 
        var_names=[var.name for var in foraging_model.model.free_RVs]
    )
    summary_path = results_dir / "model_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    
    # Print diagnostics
    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    
    # Check for divergences
    divergences = idata.sample_stats.diverging.sum().values
    print(f"Divergences: {divergences}")
    
    # Check r-hat
    rhat_max = summary['r_hat'].max()
    print(f"Max r-hat: {rhat_max:.3f}")
    
    # Check ESS
    ess_min = summary['ess_bulk'].min()
    print(f"Min ESS (bulk): {ess_min:.0f}")
    
    print("\n" + "=" * 60)
    print("Done! Results saved to results/")
    print("=" * 60)
    
    return idata, foraging_model


if __name__ == "__main__":
    main()
