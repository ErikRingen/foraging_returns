"""Tests for Shapley value computation for group contribution attribution."""
import pytest
import numpy as np
import xarray as xr
from foraging_model.counterfactuals import shapley_values


class TestShapleyValuesBasic:
    """Test basic Shapley value properties."""
    
    def test_shapley_values_sum_to_total(self):
        """Shapley values should sum to the total value (efficiency property)."""
        # Simple test case: group with 2 members
        group_members = [0, 1]
        mu_values = {
            frozenset(): 0.0,  # Empty group
            frozenset([0]): 1.0,
            frozenset([1]): 2.0,
            frozenset([0, 1]): 4.0,  # Full group
        }
        
        shapley = shapley_values(group_members, mu_values)
        
        # Efficiency: sum of Shapley values equals total value
        assert abs(sum(shapley.values()) - mu_values[frozenset(group_members)]) < 1e-10
        
    def test_shapley_values_symmetric(self):
        """Shapley values should be symmetric (order-invariant)."""
        group_members = [0, 1]
        mu_values = {
            frozenset(): 0.0,
            frozenset([0]): 1.0,
            frozenset([1]): 1.0,  # Same skill
            frozenset([0, 1]): 3.0,
        }
        
        shapley = shapley_values(group_members, mu_values)
        
        # If members contribute equally, Shapley values should be equal
        assert abs(shapley[0] - shapley[1]) < 1e-10
        
    def test_shapley_values_dummy_player(self):
        """Players who don't contribute should get zero Shapley value."""
        group_members = [0, 1]
        # True dummy player: never adds value to any subset
        mu_values = {
            frozenset(): 0.0,
            frozenset([0]): 1.0,
            frozenset([1]): 0.0,  # Member 1 alone contributes nothing
            frozenset([0, 1]): 1.0,  # Same as just member 0 (member 1 adds nothing)
        }
        
        shapley = shapley_values(group_members, mu_values)
        
        # Member 1 adds no value to any subset, so should get zero
        assert abs(shapley[1]) < 1e-10
        
    def test_shapley_values_three_members(self):
        """Test Shapley values with 3 members."""
        group_members = [0, 1, 2]
        mu_values = {
            frozenset(): 0.0,
            frozenset([0]): 1.0,
            frozenset([1]): 2.0,
            frozenset([2]): 3.0,
            frozenset([0, 1]): 4.0,
            frozenset([0, 2]): 5.0,
            frozenset([1, 2]): 6.0,
            frozenset([0, 1, 2]): 10.0,
        }
        
        shapley = shapley_values(group_members, mu_values)
        
        # Efficiency property
        assert abs(sum(shapley.values()) - 10.0) < 1e-10
        
        # All members should have positive contributions
        assert all(v >= 0 for v in shapley.values())


    def test_compute_mu_for_subset(self):
        """Test the helper function that computes mu for a subset."""
        from foraging_model.counterfactuals import _compute_mu_for_subset
        import numpy as np
        
        # Create test skills
        S_x = np.array([0.8, 0.6, 0.4, 0.2])
        
        # Test parameters
        intercept_mu = 0.0
        b_groupsize_mu = 0.2
        eta_mu = 1.0
        
        # Test empty subset
        mu_empty = _compute_mu_for_subset([], S_x, intercept_mu, b_groupsize_mu, eta_mu)
        assert mu_empty == 0.0
        
        # Test single member
        mu_single = _compute_mu_for_subset([0], S_x, intercept_mu, b_groupsize_mu, eta_mu)
        # Should be exp(0) * 0.8^1 = 0.8
        assert abs(mu_single - 0.8) < 1e-10
        
        # Test two members
        mu_two = _compute_mu_for_subset([0, 1], S_x, intercept_mu, b_groupsize_mu, eta_mu)
        # Should compute max over k=1, k=2
        # k=1: exp(0) * 0.8^1 = 0.8
        # k=2: exp(0.2 * log(2)) * 0.7^1 = exp(0.2 * 0.693) * 0.7 ≈ 1.15 * 0.7 ≈ 0.805
        assert mu_two > 0.7  # Should be positive


class TestShapleyValuesApproximation:
    """Test approximation methods for Shapley values."""
    
    def test_monte_carlo_shapley(self):
        """Test Monte Carlo approximation of Shapley values."""
        group_members = [0, 1, 2]
        mu_values = {
            frozenset(): 0.0,
            frozenset([0]): 1.0,
            frozenset([1]): 2.0,
            frozenset([2]): 3.0,
            frozenset([0, 1]): 4.0,
            frozenset([0, 2]): 5.0,
            frozenset([1, 2]): 6.0,
            frozenset([0, 1, 2]): 10.0,
        }
        
        # Exact Shapley values
        exact_shapley = shapley_values(group_members, mu_values)
        
        # Monte Carlo approximation
        mc_shapley = shapley_values(
            group_members, 
            mu_values, 
            method='monte_carlo',
            n_samples=1000
        )
        
        # Should be close to exact values
        for member in group_members:
            assert abs(exact_shapley[member] - mc_shapley[member]) < 0.5  # Allow some error

