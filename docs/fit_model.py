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
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pyprojroot.here import here

from preprocessing import preprocess_data

# %%
raw_data_dir = "raw_data/"

# %%
df_foragers, df_groups, df_returns = preprocess_data(
    returns_file=here(raw_data_dir + 'returns.csv'),
    kcal_file=here(raw_data_dir + 'kcal.csv'),
    group_file=here(raw_data_dir + 'groups.csv'),
    camp_members_file=here(raw_data_dir + 'camp_members.csv')
)

# %%
ds_groups = df_groups.set_index('group_id').to_xarray().rename({'group_id': 'group', 'id': 'foragers'}).set_coords('date')

ds_returns = df_returns.set_index('group_id').to_xarray().rename({'group_id': 'group'}).set_coords('date')

ds_foragers = df_foragers.set_index('id').to_xarray().rename({'id': 'forager'})

# %% Combine into a single dataset
combined_ds = xr.merge([ds_foragers, ds_groups, ds_returns])

# %%
