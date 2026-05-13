"""
Test fixtures and shared utilities for foraging returns model tests.

This module provides fixtures that can be used across multiple test modules.
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


@pytest.fixture
def sample_foragers_df():
    """Create a minimal foragers dataframe for testing."""
    return pd.DataFrame({
        'id': ['1', '2', '3'],
        'sex': ['male', 'female', 'male'],
        'age': [25.0, 30.0, 35.0]
    })


@pytest.fixture
def sample_production_df():
    """Create a minimal production dataframe for testing."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'group_id': [f'{d.strftime("%Y-%m-%d")}_1' for d in dates],
        'kcal': [100.0, 200.0, 0.0, 150.0, 250.0],
        'type': ['foraging', 'foraging', 'zero_inferred', 'foraging', 'foraging'],
        'date': dates,
        'group_size': [1, 1, 1, 1, 1],
        'forager_ids': [{'1'}, {'1'}, {'1'}, {'1'}, {'1'}]
    })


@pytest.fixture
def sample_time_allocation_df():
    """Create a minimal time allocation dataframe for testing."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'id': ['1'] * 5,
        'date': dates,
        'total.minutes': [120.0, 180.0, 90.0, 150.0, 200.0]
    })


@pytest.fixture
def sample_days_in_camp_df():
    """Create a minimal days in camp dataframe for testing."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'forager_id': ['1'] * 5,
        'date': dates,
        'in_camp': [1, 1, 1, 1, 1]
    })


@pytest.fixture
def sample_dataset():
    """Create a minimal xarray Dataset matching what ForagingData.to_dataset() produces.

    Dimensions: forager (3), group (4), forager_in_group (3), date (5),
    effort_obs (15 = 3 foragers × 5 dates), forager_date (8).
    Unscaled values only — scaling is handled in ForagingModel.
    """
    n_dates = 5
    n_foragers = 3
    n_groups = 4
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    date_numeric = np.arange(n_dates, dtype=float)

    # Effort component: every (forager, date) combination is an observation.
    n_effort_obs = n_foragers * n_dates
    effort_forager_idx = np.repeat(np.arange(n_foragers), n_dates)
    effort_date_idx = np.tile(np.arange(n_dates), n_foragers)
    forager_effort = np.array(
        [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=int
    )

    # Success component: only "successful" forager-day pairs (n=8).
    n_success_obs = 8
    forager_idx = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    success_date_idx = np.array([0, 1, 0, 1, 2, 0, 1, 3], dtype=int)
    forager_success = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)

    # Returns component: 4 groups, with forager_ids matrix mapping group -> forager indices.
    forager_ids = np.array(
        [[0, 1, -1], [0, -1, -1], [1, -1, -1], [2, -1, -1]], dtype=int
    )
    group_date_idx = np.array([0, 1, 2, 3], dtype=int)
    group_size = np.array([2, 1, 1, 1], dtype=int)
    foraging_proportion = np.ones(n_groups, dtype=float)

    coords = {
        "forager": ["1", "2", "3"],
        "group": [f"group_{i}" for i in range(n_groups)],
        "forager_in_group": range(forager_ids.shape[1]),
        "date": dates,
        "effort_date": np.arange(n_dates),
        "effort_obs": np.arange(n_effort_obs),
        "forager_date": np.arange(n_success_obs),
        "success_date": np.arange(n_success_obs),
        "group_date": np.arange(n_groups),
    }

    rng = np.random.default_rng(0)
    data_vars = {
        # Forager-level
        "age": (["forager"], [25.0, 30.0, 35.0]),
        "sex": (["forager"], np.array(["male", "female", "male"], dtype=object)),
        "gender_idx": (["forager"], np.array([0, 1, 0], dtype=np.int32)),
        # Date
        "date_numeric": (["date"], date_numeric),
        # Effort component (Bernoulli)
        "forager_effort": (["effort_obs"], forager_effort),
        "effort_forager_idx": (["effort_obs"], effort_forager_idx),
        "effort_date_idx": (["effort_obs"], effort_date_idx),
        # Success component (Bernoulli, only eligible obs)
        "forager_success": (["forager_date"], forager_success),
        "forager_idx": (["forager_date"], forager_idx),
        "success_date_idx": (["forager_date"], success_date_idx),
        # Returns component (Gamma / LogNormal)
        "kcal": (["group"], [100.0, 200.0, 150.0, 250.0]),
        "group_date_idx": (["group"], group_date_idx),
        "group_size": (["group"], group_size),
        "foraging_proportion": (["group"], foraging_proportion),
        "forager_ids": (["group", "forager_in_group"], forager_ids),
        # Auxiliary
        "total.minutes": (["forager", "date"], rng.uniform(0, 300, (n_foragers, n_dates))),
        "in_camp": (["forager", "date"], np.ones((n_foragers, n_dates), dtype=bool)),
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def synthetic_forager_daily_data():
    """Create synthetic forager-daily level data for testing."""
    n_foragers = 10
    n_dates = 20
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    
    return pd.DataFrame({
        'forager_id': np.repeat(range(n_foragers), n_dates),
        'date': np.tile(dates, n_foragers),
        'success': np.random.binomial(1, 0.3, n_foragers * n_dates),
        'age': np.repeat(np.random.uniform(20, 50, n_foragers), n_dates),
        'time_allocation': np.random.uniform(0, 300, n_foragers * n_dates)
    })


@pytest.fixture
def synthetic_group_trip_data():
    """Create synthetic group-trip level data for testing."""
    n_groups = 50
    dates = pd.date_range('2023-01-01', periods=n_groups, freq='D')
    
    return pd.DataFrame({
        'group_id': [f'group_{i}' for i in range(n_groups)],
        'date': dates,
        'returns': np.random.gamma(2, 100, n_groups),
        'group_size': np.random.randint(1, 5, n_groups),
        'has_success': np.random.binomial(1, 0.7, n_groups)
    })

