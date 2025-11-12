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
    """Create a minimal xarray Dataset for testing.
    
    Note: Dataset should have unscaled data only (scaling happens in model).
    group_date uses numeric values to avoid datetime64 issues in PyTensor.
    Includes forager-daily level data for Bernoulli component.
    """
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    coords = {
        'forager': ['1', '2', '3'],
        'group': [f'group_{i}' for i in range(4)],  # Only successful trips (no zeros)
        'forager_in_group': range(3),
        'date': dates  # For forager-daily level
    }
    
    # Create long-format forager_success data (only eligible observations)
    # Simulate some eligible (forager, date) pairs
    n_observations = 8  # Some eligible observations
    forager_indices = [0, 0, 1, 1, 1, 2, 2, 2]  # Which foragers
    success_values = [1, 0, 1, 1, 0, 1, 0, 1]  # Success values
    # Create date array matching n_observations
    observation_dates = pd.date_range('2023-01-01', periods=n_observations, freq='D')
    
    data_vars = {
        'age': (['forager'], [25.0, 30.0, 35.0]),
        'kcal': (['group'], [100.0, 200.0, 150.0, 250.0]),  # Unscaled, all > 0 (only successful trips)
        'group_date': (['group'], np.arange(4)),  # Use numeric instead of datetime
        'forager_ids': (['group', 'forager_in_group'], 
                       np.array([[0, 1, -1], [0, -1, -1], 
                                [1, -1, -1], [2, -1, -1]])),
        # Forager-daily level data (Bernoulli component) - long format
        'forager_success': (['forager_date'], np.array(success_values, dtype=int)),
        'forager_idx': (['forager_date'], np.array(forager_indices, dtype=int)),
        'forager_date_date': (['forager_date'], observation_dates.values),  # Renamed to avoid conflict
        'total.minutes': (['forager', 'date'], np.random.uniform(0, 300, (3, 5))),
        'in_camp': (['forager', 'date'], np.ones((3, 5), dtype=bool))
    }
    
    # Add forager_date coordinate
    coords['forager_date'] = np.arange(n_observations)
    
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

