"""Tests for Shapley value computation for group contribution attribution."""
import pytest
import numpy as np
import xarray as xr
from foraging_model.counterfactuals import (
    compute_mu_for_subset,
    shapley_exact,
    shapley_monte_carlo,
    shapley_contributions,
    shapley_contributions_all_groups,
    EXACT_THRESHOLD,
    VALID_METHODS,
)


class TestComputeMuForSubset:
    """Test the helper function that computes mu for a subset."""

    def test_empty_subset_returns_zero(self):
        """Empty subset should return zero mu."""
        S_x = np.array([0.8, 0.6, 0.4, 0.2])
        mu = compute_mu_for_subset([], S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0)
        assert mu == 0.0

    def test_single_member_subset(self):
        """Single member subset should return skill value scaled by g(1)."""
        S_x = np.array([0.8, 0.6, 0.4, 0.2])
        mu = compute_mu_for_subset(
            [0], S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        # g(1) = exp(0 + 0.2 * log(1)) = exp(0) = 1.0
        # mu = 1.0 * 0.8^1 = 0.8
        assert abs(mu - 0.8) < 1e-10

    def test_two_member_subset(self):
        """Two member subset should compute max over k=1,2."""
        S_x = np.array([0.8, 0.6, 0.4, 0.2])
        mu = compute_mu_for_subset(
            [0, 1], S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        # Should be positive and reasonable
        assert mu > 0.5
        assert mu < 2.0


class TestShapleyExact:
    """Test exact Shapley value computation."""

    def test_efficiency_property(self):
        """Shapley values should sum to total coalition value."""
        S_x = np.array([0.8, 0.6])
        group_members = [0, 1]
        
        shapley = shapley_exact(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        total_mu = compute_mu_for_subset(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        assert abs(sum(shapley.values()) - total_mu) < 1e-10

    def test_symmetry_property(self):
        """Players with same skills should have same Shapley values."""
        S_x = np.array([0.5, 0.5])
        group_members = [0, 1]
        
        shapley = shapley_exact(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        assert abs(shapley[0] - shapley[1]) < 1e-10

    def test_higher_skill_gets_higher_value(self):
        """Player with higher skill should get higher Shapley value."""
        S_x = np.array([0.8, 0.4])
        group_members = [0, 1]
        
        shapley = shapley_exact(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        assert shapley[0] > shapley[1]

    def test_three_members(self):
        """Test with three members."""
        S_x = np.array([0.8, 0.6, 0.4])
        group_members = [0, 1, 2]
        
        shapley = shapley_exact(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        total_mu = compute_mu_for_subset(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        # Efficiency
        assert abs(sum(shapley.values()) - total_mu) < 1e-10
        # All positive
        assert all(v >= 0 for v in shapley.values())

    def test_empty_group(self):
        """Empty group should return empty dict."""
        S_x = np.array([0.8, 0.6])
        shapley = shapley_exact(
            [], S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        assert shapley == {}


class TestShapleyMonteCarlo:
    """Test Monte Carlo Shapley value computation."""

    def test_approximates_exact(self):
        """Monte Carlo should approximate exact values."""
        S_x = np.array([0.8, 0.6, 0.4])
        group_members = [0, 1, 2]
        
        exact = shapley_exact(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        mc = shapley_monte_carlo(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0,
            n_samples=5000, random_seed=42
        )
        
        for member in group_members:
            # Allow 10% relative error
            assert abs(exact[member] - mc[member]) < 0.1 * exact[member] + 0.01

    def test_efficiency_property(self):
        """Monte Carlo Shapley values should approximately sum to total."""
        S_x = np.array([0.8, 0.6, 0.4])
        group_members = [0, 1, 2]
        
        mc = shapley_monte_carlo(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0,
            n_samples=5000, random_seed=42
        )
        
        total_mu = compute_mu_for_subset(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0
        )
        
        # Allow some approximation error
        assert abs(sum(mc.values()) - total_mu) < 0.1

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        S_x = np.array([0.8, 0.6, 0.4])
        group_members = [0, 1, 2]
        
        mc1 = shapley_monte_carlo(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0,
            n_samples=100, random_seed=42
        )
        
        mc2 = shapley_monte_carlo(
            group_members, S_x, intercept_mu=0.0, b_groupsize_mu=0.2, eta_mu=1.0,
            n_samples=100, random_seed=42
        )
        
        for member in group_members:
            assert mc1[member] == mc2[member]


class TestShapleyMethodParameter:
    """Test the method parameter for shapley_contributions functions."""

    def test_valid_methods(self):
        """Check valid method constants."""
        assert "auto" in VALID_METHODS
        assert "exact" in VALID_METHODS
        assert "monte_carlo" in VALID_METHODS

    def test_exact_threshold_value(self):
        """Exact threshold should be reasonable (around 15)."""
        assert 10 <= EXACT_THRESHOLD <= 20


class TestVectorizedFunctions:
    """Test that vectorized functions match scalar versions."""

    def test_vectorized_mu_matches_scalar(self):
        """Vectorized mu computation should match scalar version."""
        from foraging_model.counterfactuals import (
            compute_mu_for_subset_vectorized,
        )
        
        # Create test data: 5 samples, 4 foragers
        n_samples = 5
        S_x_2d = np.random.RandomState(42).rand(n_samples, 4) * 0.8 + 0.1
        intercept = np.random.RandomState(43).randn(n_samples) * 0.1
        b_groupsize = np.random.RandomState(44).rand(n_samples) * 0.3
        eta = np.random.RandomState(45).rand(n_samples) * 0.5 + 0.5
        
        subset = [0, 2]
        
        # Vectorized
        mu_vec = compute_mu_for_subset_vectorized(
            subset, S_x_2d, intercept, b_groupsize, eta
        )
        
        # Scalar loop
        mu_scalar = np.array([
            compute_mu_for_subset(subset, S_x_2d[i], intercept[i], b_groupsize[i], eta[i])
            for i in range(n_samples)
        ])
        
        np.testing.assert_allclose(mu_vec, mu_scalar, rtol=1e-10)

    def test_vectorized_exact_matches_scalar(self):
        """Vectorized exact Shapley should match scalar version."""
        from foraging_model.counterfactuals import shapley_exact_vectorized
        
        n_samples = 3
        S_x_2d = np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3],
            [0.9, 0.7, 0.5],
        ])
        intercept = np.zeros(n_samples)
        b_groupsize = np.full(n_samples, 0.2)
        eta = np.ones(n_samples)
        
        group_members = [0, 1, 2]
        
        # Vectorized
        shapley_vec = shapley_exact_vectorized(
            group_members, S_x_2d, intercept, b_groupsize, eta
        )
        
        # Scalar loop
        for i in range(n_samples):
            scalar_result = shapley_exact(
                group_members, S_x_2d[i], intercept[i], b_groupsize[i], eta[i]
            )
            for j, member in enumerate(group_members):
                assert abs(shapley_vec[i, j] - scalar_result[member]) < 1e-10

    def test_vectorized_monte_carlo_reasonable(self):
        """Vectorized MC Shapley should give reasonable results."""
        from foraging_model.counterfactuals import shapley_monte_carlo_vectorized
        
        n_samples = 3
        S_x_2d = np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3],
            [0.9, 0.7, 0.5],
        ])
        intercept = np.zeros(n_samples)
        b_groupsize = np.full(n_samples, 0.2)
        eta = np.ones(n_samples)
        
        group_members = [0, 1, 2]
        
        shapley_mc = shapley_monte_carlo_vectorized(
            group_members, S_x_2d, intercept, b_groupsize, eta,
            n_permutations=1000, random_seed=42
        )
        
        # All values should be positive (skill contributions)
        assert np.all(shapley_mc >= 0)
        
        # Higher skill foragers should have higher contributions
        for i in range(n_samples):
            assert shapley_mc[i, 0] > shapley_mc[i, 1]
            assert shapley_mc[i, 1] > shapley_mc[i, 2]
