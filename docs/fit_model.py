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

# %%
raw_data_dir = "raw_data/"

# %%
df_foragers, df_time_agg, df_returns = preprocess_data(
    returns_file=here(raw_data_dir + 'returns.csv'),
    recall_file=here(raw_data_dir + 'recall.csv'),
    kcal_file=here(raw_data_dir + 'kcal.csv'),
    group_file=here(raw_data_dir + 'groups.csv'),
    camp_members_file=here(raw_data_dir + 'camp_members.csv'),
    days_in_camp_file=here(raw_data_dir + 'daysincamp.csv')
)

# %%
data = ForagingData(
    foragers_df=df_foragers,
    time_allocation_df=df_time_agg,
    production_df=df_returns,
    target_column='kcal',
    target_scaling='mean',
    age_scaling='max',
    group_id_col='group_id',
    forager_id_col='id'
)

# %%
dataset = data.to_dataset()
# %%
