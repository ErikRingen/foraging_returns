# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: foraging
#     language: python
#     name: foraging
# ---

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

from preprocessing import preprocess_data

from foraging_model.foraging_data import ForagingData
from foraging_model.foraging_model import ForagingModel

import pytensor
pytensor.config.cxx = '/usr/bin/clang++' 

# %load_ext autoreload
# %autoreload 2

# %%
params = {
    "sample_params": {
        "tune": 150,
        "draws": 250,
        "chains": 4,
        "nuts_sampler": "nutpie",
    },
    "seed": sum(map(ord, "outlet model")),
    "rng": np.random.default_rng(sum(map(ord, "outlet model"))),
    "HDI_PROB": 0.9,
}

# %%
raw_data_dir = "raw_data/"

# %%
df_foragers, df_time_agg, df_returns = preprocess_data(
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
    production_df=df_returns,
    target_column='kcal',
    # kcal scaled by mean of non-zero values
    target_scaling='mean',
    age_scaling='max',
    group_id_col='group_id',
    forager_id_col='id',
)

# %%
dataset = data.to_dataset()

# %%
foraging_model = ForagingModel(data=dataset)
foraging_model.build_model()

# %%
gv = pm.model_to_graphviz(foraging_model.model)
gv.format = "png"
gv.render(filename="img/model_graph")

# %%
foraging_model.fit(**params["sample_params"])

# %%
az.summary(foraging_model.idata, var_names=['shape', 'eta_mu', 'eta_success', 'b_groupsize_mu', 'b_groupsize_success'])

# %%
az.plot_dist_comparison(foraging_model.idata, var_names=['shape', 'eta_mu', 'eta_success', 'b_groupsize_mu', 'b_groupsize_success', 'intercept_mu', 'intercept_success', 'm', 'k', 'b'])


# %%
if not foraging_model.already_rescaled:
    foraging_model.rescale_predictive()

# %%
#az.plot_ppc(foraging_model.idata);

az.plot_ppc(foraging_model.idata, group='prior', kind='cumulative', var_names=['kcal']);
az.plot_ppc(foraging_model.idata, kind='cumulative', group='posterior')

# %%
foraging_model.plot_curve(prior=True, type="S")
foraging_model.plot_curve(prior=True, type="M")
foraging_model.plot_curve(prior=True, type="K")

# %%
foraging_model.plot_curve(prior=False, type="S")
foraging_model.plot_curve(prior=False, type="M")
foraging_model.plot_curve(prior=False, type="K")
foraging_model.plot_curve(prior=False, type="harvest")
foraging_model.plot_curve(prior=False, type="success")
# %%

# %%
forager_ids = foraging_model.idata["constant_data"]["forager_ids"].copy()

# replace forager_ids 0 with -1
forager_ids = np.where(forager_ids == 0, -1, forager_ids)
intervention = {
    "forager_ids": forager_ids,
}

intervened_model = pm.do(foraging_model.model, intervention)

# %%
with intervened_model:
    idata_intervened = pm.sample_posterior_predictive(
        foraging_model.idata,
        var_names=["expected"],
    )

counterfactual_mu = idata_intervened

# %%
diff = foraging_model.idata.posterior["expected"] - counterfactual_mu.posterior_predictive["expected"]

# %%
# add the date coordinate to the diff
diff["date"] = foraging_model.data.date

# %%
# sum over groups for date and plot
diff.groupby('date').sum().mean(dim=["chain", "draw"]).plot(x="date", y="expected")

# %%
expected = foraging_model.idata.posterior["expected"]

# Get forager ages from the data
forager_ages = foraging_model.data.age.values

# Create a list of (forager_idx, age) tuples and sort by age
forager_info = [(idx, forager_ages[idx]) for idx in range(len(foraging_model.data.coords["forager"]))]
forager_info.sort(key=lambda x: x[1])  # Sort by age (second element in tuple)

# Create a single figure with subplots
fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 20), sharex=True, sharey=True)
axes_flat = axes.flatten()

# For legend
all_lines = []
all_labels = []

# Loop over foragers in age-sorted order
for plot_idx, (forager_idx, age) in enumerate(forager_info):
    if plot_idx >= len(axes_flat):  # Safety check
        break
        
    forager_id = foraging_model.data.coords["forager"].values[forager_idx]  # Get actual ID for label
    forager_ids = foraging_model.idata["constant_data"]["forager_ids"].copy()
    
    # Set this forager's ID to -1 (remove from groups)
    forager_ids = np.where(forager_ids == forager_idx, -1, forager_ids)
    intervention = {"forager_ids": forager_ids}
    
    intervened_model = pm.do(foraging_model.model, intervention)
    
    with intervened_model:
        idata_intervened = pm.sample_posterior_predictive(
            foraging_model.idata,
            var_names=["expected"]
        )
    
    # Calculate the difference
    diff = foraging_model.idata.posterior["expected"] - idata_intervened.posterior_predictive["expected"]

    diff *= foraging_model.data.kcal_scale
    
    # Add the date coordinate
    diff = diff.assign_coords(date=("group", foraging_model.data.date.values))
    
    # Group by date
    diff_by_date = diff.groupby('date').sum()
    
    # Calculate mean and credible intervals
    mean_diff = diff_by_date.mean(dim=["chain", "draw"])
    lower_ci = diff_by_date.quantile(0.05, dim=["chain", "draw"])
    upper_ci = diff_by_date.quantile(0.95, dim=["chain", "draw"])
    
    # Get dates and convert to datetime for better plotting
    dates = mean_diff.coords['date'].values
    
    # Plot mean line
    line, = axes_flat[plot_idx].plot(dates, mean_diff.values, 
                                   label=f"Forager {forager_id}")
    
    # Add shaded credible interval
    axes_flat[plot_idx].fill_between(dates, 
                                   lower_ci.values, 
                                   upper_ci.values, 
                                   alpha=0.3)
    
    # Add to legend collection
    all_lines.append(line)
    all_labels.append(f"Forager {forager_id} (Age: {age:.1f})")
    
    # Set the title for each subplot including age
    axes_flat[plot_idx].set_title(f"Forager {forager_id} (Age: {age:.1f})")
    
    # Format date ticks if dates are datetime objects
    if hasattr(dates[0], 'strftime'):
        axes_flat[plot_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes_flat[plot_idx].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for idx in range(len(forager_info), len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Add common labels using figure-level commands
fig.suptitle("Marginal Contribution by Forager with 90% Credible Intervals (Sorted by Age)", fontsize=16)

# Add common x and y labels
fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Contribution (expected kcal)', va='center', rotation='vertical', fontsize=14)

# Add a legend at the bottom of the figure with multiple columns
# fig.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0), 
#            ncol=min(5, len(all_lines)), fontsize='small')

# Adjust layout
plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.95])  # Leave space for labels and legend
plt.subplots_adjust(top=0.92, bottom=0.12)  # Adjust top and bottom margins

plt.show()

# Add a legend at the bottom o
# %%
