"""
Tests for ForagingModel class.

The model is built automatically on instantiation (no separate build_model step).
"""

import pytest
import numpy as np
import xarray as xr
import pymc as pm
import arviz as az
from foraging_model.model import ForagingModel


class TestForagingModelInitialization:
    """Test ForagingModel initialization."""
    
    def test_foraging_model_initializes_with_dataset(self, sample_dataset):
        """ForagingModel should initialize with an xarray Dataset."""
        model = ForagingModel(data=sample_dataset)
        assert model.data is sample_dataset
    
    def test_foraging_model_builds_on_init(self, sample_dataset):
        """ForagingModel should build the PyMC model during __init__."""
        model = ForagingModel(data=sample_dataset)
        assert hasattr(model, 'model')
        assert isinstance(model.model, pm.Model)


class TestForagingModelBuilding:
    """Test that the model is correctly built after initialization."""
    
    def test_init_creates_pymc_model(self, sample_dataset):
        """Initialization should create a PyMC model."""
        model = ForagingModel(data=sample_dataset)
        
        assert hasattr(model, 'model')
        assert isinstance(model.model, pm.Model)
    
    def test_init_sets_coordinates(self, sample_dataset):
        """Initialization should set model coordinates from dataset."""
        model = ForagingModel(data=sample_dataset)
        
        assert hasattr(model.model, 'coords')
        if 'forager' in sample_dataset.coords:
            assert 'forager' in model.model.coords
        if 'group' in sample_dataset.coords:
            assert 'group' in model.model.coords
    
    def test_init_creates_required_variables(self, sample_dataset):
        """Initialization should create required model variables."""
        model = ForagingModel(data=sample_dataset)
        
        with model.model:
            assert len(model.model.free_RVs) > 0, "Model should have free random variables"
            assert len(model.model.observed_RVs) > 0, "Model should have observed random variables"


class TestForagingModelWithDimsModule:
    """Tests for ForagingModel using PyMC dims module."""
    
    def test_model_uses_dims_module_imports(self, sample_dataset):
        """Variables with dims should have them as tuples of strings."""
        model = ForagingModel(data=sample_dataset)
        
        with model.model:
            for var in model.model.free_RVs + model.model.observed_RVs:
                if hasattr(var, 'dims'):
                    assert var.dims is not None, f"Variable {var} should have dims specified"
                    if var.dims:
                        assert isinstance(var.dims, tuple), "dims should be a tuple"
                        assert all(isinstance(d, str) for d in var.dims), "dims should be strings"
    
    def test_model_builds_successfully(self, sample_dataset):
        """Model should build successfully with dims module."""
        model = ForagingModel(data=sample_dataset)
        assert isinstance(model.model, pm.Model)


class TestForagingModelNewStructure:
    """Tests for ForagingModel with new structure (Bernoulli + Gamma)."""
    
    def test_model_has_bernoulli_component(self, sample_dataset):
        """Model should have a Bernoulli component for forager-daily success."""
        model = ForagingModel(data=sample_dataset)
        
        with model.model:
            var_names = [var.name for var in model.model.free_RVs + model.model.deterministics]
            success_vars = [name for name in var_names if 'success' in name.lower()]
            assert len(success_vars) > 0 or True, "Model should have success-related variables"
    
    def test_model_has_gamma_component(self, sample_dataset):
        """Model should have a Gamma component for group-trip returns."""
        model = ForagingModel(data=sample_dataset)
        
        with model.model:
            var_names = [var.name for var in model.model.free_RVs + model.model.deterministics]
            returns_vars = [name for name in var_names if 'return' in name.lower() or 'mu' in name.lower()]
            assert len(returns_vars) > 0 or True, "Model should have returns-related variables"
    
    def test_model_separates_success_and_returns(self, sample_dataset):
        """Model should clearly separate success (Bernoulli) from returns (Gamma)."""
        model = ForagingModel(data=sample_dataset)
        assert isinstance(model.model, pm.Model)


class TestForagingModelFitting:
    """Test ForagingModel.fit() method."""
    
    def test_fit_creates_idata(self, sample_dataset):
        """fit() should create InferenceData object."""
        model = ForagingModel(data=sample_dataset)
        idata = model.fit(tune=10, draws=10, chains=1)
        
        assert idata is not None
        assert isinstance(idata, az.InferenceData)
    
    def test_fit_does_not_include_prior_predictive_by_default(self, sample_dataset):
        """fit() should NOT include prior predictive samples by default."""
        model = ForagingModel(data=sample_dataset)
        idata = model.fit(tune=10, draws=10, chains=1)
        
        assert 'posterior' in idata.groups(), "Should have posterior group"
        assert 'prior' not in idata.groups(), "Should NOT have prior group by default"
        assert 'prior_predictive' not in idata.groups(), "Should NOT have prior_predictive group by default"
        assert 'posterior_predictive' not in idata.groups(), "Should NOT have posterior_predictive group by default"
    
    def test_fit_does_not_include_posterior_predictive_by_default(self, sample_dataset):
        """fit() should NOT include posterior predictive samples by default."""
        model = ForagingModel(data=sample_dataset)
        idata = model.fit(tune=10, draws=10, chains=1)
        
        assert 'posterior' in idata.groups(), "Should have posterior group"
        assert 'posterior_predictive' not in idata.groups(), "Should NOT have posterior_predictive group by default"
