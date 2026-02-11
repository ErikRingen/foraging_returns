# Foraging Returns Model

A Bayesian model for analyzing foraging returns data, examining how individual forager skills combine to predict group-level hunting/gathering outcomes.

## Overview

This project models foraging returns using a two-component hierarchical Bayesian approach:

1. **Bernoulli Component**: Models daily forager success probability as a function of individual skill
2. **Gamma Component**: Models group-level return magnitude based on aggregated forager skills

The model implements age-dependent skill curves and supports multiple methods for aggregating individual skills to group-level predictions.

## Installation

This project uses [pixi](https://pixi.sh/) for environment management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
git clone <repository-url>
cd foraging_returns
pixi install
```

## Quick Start

```python
import pandas as pd
from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel

# Load data
foragers_df = pd.read_csv("data/foragers.csv")
production_df = pd.read_csv("data/production.csv")
time_allocation_df = pd.read_csv("data/time_allocation.csv")
days_in_camp_df = pd.read_csv("data/groups.csv")

# Prepare data
data = ForagingData(
    foragers_df=foragers_df,
    time_allocation_df=time_allocation_df,
    production_df=production_df,
    days_in_camp_df=days_in_camp_df,
    target_column="kcal",
    group_id_col="group_id",
    forager_id_col="id"
)
dataset = data.to_dataset()

# Create and fit model (model is built automatically)
model = ForagingModel(
    data=dataset,
    aggregation_method="best_top_k",  # or "mean" for simpler baseline
    target_scaling="mean"
)
idata = model.fit(tune=500, draws=1000, chains=4)
```

## Model Comparison

The model supports two aggregation methods, enabling model comparison via LOO-CV:

```python
# Fit both models
model_topk = ForagingModel(dataset, aggregation_method="best_top_k")
model_mean = ForagingModel(dataset, aggregation_method="mean")

idata_topk = model_topk.fit()
idata_mean = model_mean.fit()

# Compare using LOO-CV
import arviz as az
az.compare({"best_top_k": idata_topk, "mean": idata_mean})
```

## Project Structure

```
foraging_returns/
├── foraging_model/          # Main package
│   ├── model.py             # ForagingModel class
│   ├── data.py              # ForagingData class  
│   ├── counterfactuals.py   # Shapley value computations
│   ├── plotting.py          # Visualization utilities
│   └── utils.py             # Helper functions
├── data/                    # Processed data files
├── raw_data/                # Original data files
├── docs/                    # Documentation
│   ├── MODEL.md             # Mathematical specification
│   └── SHAPLEY_VALUES.md    # Contribution attribution methods
├── tests/                   # Test suite
├── pixi.toml                # Environment specification
└── pyproject.toml           # Package configuration
```

## Data Description

The model expects four input DataFrames:

| DataFrame | Description | Key columns |
|-----------|-------------|-------------|
| `foragers_df` | Forager demographics | `id`, `age`, `sex` |
| `production_df` | Group production records | `group_id`, `kcal`, `date`, `forager_ids` |
| `time_allocation_df` | Daily time allocation | `id`, `date`, `total.minutes` |
| `days_in_camp_df` | Camp presence records | `forager_id`, `date`, `in_camp` |

## Documentation

- **[Model Specification](docs/MODEL.md)**: Mathematical details of the model components
- **[Shapley Values](docs/SHAPLEY_VALUES.md)**: Methods for attributing group returns to individuals

## Running Tests

```bash
pixi run pytest tests/
```

## Dependencies

Key dependencies managed via pixi:
- PyMC 5.x
- ArviZ
- nutpie (for fast sampling)
- xarray
- pandas
- numpy

## Citation

If you use this code, please cite:

```
[Citation to be added]
```

## License

[License to be added]
