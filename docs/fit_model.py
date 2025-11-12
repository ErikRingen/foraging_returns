# %%
import pandas as pd
import xarray as xr
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pyprojroot.here import here
import arviz as az
import plotly.graph_objects as go

from preprocessing import preprocess_data

from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel
from foraging_model.plotting import (
    plot_curve, 
    plot_forager_contributions,
    plot_shapley_contributions,
    plot_shapley_summary,
    plot_shapley_by_forager
)
from foraging_model.counterfactuals import (
    marginal_contributions,
    shapley_contributions,
    shapley_contributions_all_groups
)

# %load_ext autoreload
# %autoreload 2

# %%
raw_data_dir = "raw_data/"

# %%
df_foragers, df_time_agg, df_production, df_days_long = preprocess_data(
    returns_file=here(raw_data_dir + 'returns.csv'),
    recall_file=here(raw_data_dir + 'recall.csv'),
    kcal_file=here(raw_data_dir + 'kcal.csv'),
    group_file=here(raw_data_dir + 'groups.csv'),
    camp_members_file=here(raw_data_dir + 'camp_members.csv'),
    days_in_camp_file=here(raw_data_dir + 'daysincamp.csv'),
    # for a given group/day, sum returns and recall
    combine_returns_recall=True,
    output_dir=here('data')
)

# %%
data = ForagingData(
    foragers_df=df_foragers,
    time_allocation_df=df_time_agg,
    production_df=df_production,
    days_in_camp_df=df_days_long,
    target_column='kcal',
    group_id_col='group_id',
    forager_id_col='id',
)

# %%
dataset = data.to_dataset()

# %%
foraging_model = ForagingModel(
    data=dataset,
    target_scaling='mean',  # kcal scaled by mean of non-zero values
    age_scaling='max',
)

# %%
# gv = pm.model_to_graphviz(foraging_model.model)
# gv.format = "png"
# gv.render(filename="img/model_graph")

# %%
# Fit the model
idata = foraging_model.fit(
    tune=1000,
    draws=1000,
    chains=4,
    random_seed=42,
    target_accept=0.95,
    low_rank_modified_mass_matrix=True,
)

# %%
# Generate prior and posterior predictive samples
# Make sure idata is the InferenceData object from fit()
assert hasattr(idata, 'posterior'), "idata must be an InferenceData object from foraging_model.fit()"

with foraging_model.model:
    prior = pm.sample_prior_predictive()
    posterior_predictive = pm.sample_posterior_predictive(idata)
    idata.extend(prior)
    idata.extend(posterior_predictive)

# %%
az.summary(idata, var_names=[var.name for var in foraging_model.model.free_RVs])

# %%
# compare prior and posterior
az.plot_forest(
    [idata.prior, idata.posterior],
    var_names=[var.name for var in foraging_model.model.free_RVs],
    combined=True,
    model_names=['prior', 'posterior'],
    figsize=(10, 10),
    textsize=12,
    hdi_prob=0.9,
)

# %%
# Rescale predictive samples
foraging_model.rescale_predictive()  # Modifies idata in place

# %%
az.plot_ppc(idata, group='prior', kind='cumulative', var_names=['kcal']);
az.plot_ppc(idata, kind='cumulative', group='posterior', var_names=['kcal']);

# %%
# Plot skill curves
plot_curve(idata, age_scale=foraging_model.age_scale, prior=False, curve_type="S")
plot_curve(idata, age_scale=foraging_model.age_scale, prior=False, curve_type="M")
plot_curve(idata, age_scale=foraging_model.age_scale, prior=False, curve_type="K")
plot_curve(idata, age_scale=foraging_model.age_scale, prior=False, curve_type="harvest")
plot_curve(idata, age_scale=foraging_model.age_scale, prior=False, curve_type="success")

# %%
# Plot forager contributions
marginal_contrib = marginal_contributions(
    model=foraging_model.model,
    idata=idata,
    data=foraging_model.data,
    kcal_scale=foraging_model.kcal_scale,
    group="posterior"
)
plot_forager_contributions(marginal_contrib, foraging_model.data)

# %%
marginal_contrib_prior = marginal_contributions(
    foraging_model.model,
    idata,
    foraging_model.data,
    kcal_scale=foraging_model.kcal_scale,
    group="prior"
)
plot_forager_contributions(marginal_contrib_prior, foraging_model.data)

# %%
# Pre-compute skills and mu using pm.compute_deterministics()
# This computes skills for all MCMC samples, then we take the mean
# (Don't take mean of parameters, take mean after computing skills)
with foraging_model.model:
    # Compute deterministic variables (S, mu) for all MCMC samples
    deterministics = pm.compute_deterministics(
        dataset=idata.posterior,
        var_names=['S', 'mu']
    )

# Get full posterior distributions (not means)
# S has dimensions (chain, draw, forager)
S_x = deterministics['S']

# mu has dimensions (chain, draw, group) - keep in scaled space for now
mu = deterministics['mu']

# Get full posterior parameter distributions
intercept_mu = idata.posterior["intercept_mu"]
b_groupsize_mu = idata.posterior["b_groupsize_mu"]
eta_mu = idata.posterior["eta_mu"]

# %%
# Compute Shapley values for contribution attribution
# This provides fair, additive, order-invariant attribution of group returns
# Only compute for groups with reasonable size (exponential complexity)
# Uses full MCMC samples with vectorized operations
shapley_ds = shapley_contributions_all_groups(
    S_x=S_x,
    mu=mu,
    data=foraging_model.data,
    intercept_mu=intercept_mu,
    b_groupsize_mu=b_groupsize_mu,
    eta_mu=eta_mu,
    kcal_scale=foraging_model.kcal_scale,
    method="monte_carlo",  # Use "monte_carlo" for groups > 10 members
    max_group_size=20,  # Skip groups larger than 20 (too computationally expensive)
    n_samples=50,
)


# %%
# Plot summary of Shapley contributions across all groups
# This shows total contributions summed across all groups for each forager
if len(computed_groups) > 0:
    plot_shapley_summary(
        shapley_results=shapley_ds,
        data=foraging_model.data,
        figsize=(12, 8)
    )

# %%
# Plot Shapley contributions by forager (similar to marginal contributions plot)
# Each subplot shows one forager's contributions across all groups they participated in
if len(computed_groups) > 0:
    plot_shapley_by_forager(
        shapley_results=shapley_ds,
        data=foraging_model.data,
        sort_by_age=True,
        figsize=(15, 20),
        nrows=10,
        ncols=5
    )

# %%
