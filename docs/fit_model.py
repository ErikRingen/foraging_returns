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
    forager_id_col='id'
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
az.summary(foraging_model.idata, var_names=['shape', 'intercept', 'b_groupsize'])

# %%
if not foraging_model.already_rescaled:
    foraging_model.rescale_predictive()

# %%
#az.plot_ppc(foraging_model.idata);

az.plot_ppc(foraging_model.idata, kind='cumulative', group='prior');
az.plot_ppc(foraging_model.idata, kind='cumulative', group='posterior')



# %%
