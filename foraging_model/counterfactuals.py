"""Counterfactual analysis utilities for foraging model."""
import numpy as np
import math
import xarray as xr
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from itertools import combinations, permutations
from typing import Dict, List, Optional, Callable, Union
from collections import defaultdict

def marginal_contributions(
    model: pm.Model,
    idata: az.InferenceData,
    data: xr.Dataset,
    kcal_scale: float = 1.0,
    group: str = "posterior",
) -> xr.DataArray:
    """
    Calculate the marginal contributions of each forager to the expected returns (mu) on each day.
    
    Parameters
    ----------
    model : pm.Model
        The PyMC model
    idata : az.InferenceData
        InferenceData object from fitted model
    data : xr.Dataset
        Original dataset with forager data
    kcal_scale : float, optional
        Scaling factor for kcal (to convert back to original scale), by default 1.0
    group : str, optional
        Which inference group to use: "posterior" or "prior", by default "posterior"
        
    Returns
    -------
    xr.DataArray
        Marginal contributions by forager and date
    """
    if group not in ["posterior", "prior"]:
        raise ValueError("Invalid group, must be one of posterior or prior")

    if group == "posterior":
        idata_group = idata.posterior
    else:
        idata_group = idata.prior

    mu = idata_group["mu"]
    forager_indices = range(len(data.coords["forager"]))
    forager_ids = data.coords["forager"].values

    forager_contributions = []

    # Get the original forager_ids variable from the model to check dtype
    # PyMC typically converts int64 to int32, so we need to match the model's dtype
    original_forager_ids_var = model["forager_ids"]
    # Access dtype through the tensor type
    try:
        original_dtype = original_forager_ids_var.type.dtype
    except (AttributeError, TypeError):
        # Fallback: PyMC typically uses int32 for integer data
        # Based on error message, model expects int32
        original_dtype = np.int32

    # Store original value to restore later
    original_forager_ids_value = data.forager_ids.values.copy().astype(original_dtype)

    for forager_idx in forager_indices:
        # Set this forager's ID to -1 (remove from groups)
        # Cast to match the model's expected dtype (int32, not int64)
        forager_ids_copy = data.forager_ids.values.copy().astype(original_dtype)
        forager_ids_copy = np.where(forager_ids_copy == forager_idx, -1, forager_ids_copy)
        
        # Use set_data() within model context to update the data variable directly
        # This avoids pm.do() issues with deterministic variables
        # set_data() updates pm.Data variables without rebuilding the model
        with model:
            pm.set_data({"forager_ids": forager_ids_copy})
        
        # Sample posterior predictive with the updated data
        # No need for pm.do() - set_data() modifies the model in place
        if group == "posterior":
            idata_intervened = pm.sample_posterior_predictive(
                idata,
                var_names=["mu"],
                model=model
            )
        else:
            idata_intervened = pm.sample_prior_predictive(
                var_names=["mu"],
                model=model
            )
        
        # Calculate the difference
        if group == "posterior":
            diff = mu - idata_intervened.posterior_predictive["mu"]
        else:
            diff = mu - idata_intervened.prior["mu"]
        
        # Restore original value for next iteration
        with model:
            pm.set_data({"forager_ids": original_forager_ids_value})
        
        # Rescale to original units
        diff *= kcal_scale
        
        # Add the date coordinate
        diff = diff.assign_coords(date=("group", data.group_date.values))
        
        # Group by date while preserving chains and draws
        diff_by_date = diff.groupby('date').sum()
        
        # Keep only the dimensions we need
        reduced_dims = [dim for dim in diff_by_date.dims if dim not in ['date', 'chain', 'draw']]
        if reduced_dims:
            diff_by_date = diff_by_date.sum(dim=reduced_dims)
        
        # Store this forager's contribution
        forager_contributions.append(diff_by_date)
    
    # Ensure original value is restored at the end
    with model:
        pm.set_data({"forager_ids": original_forager_ids_value})
    
    # Combine all forager contributions into a single DataArray
    # by stacking them along a new 'forager' dimension
    combined = xr.concat(forager_contributions, dim=pd.Index(forager_ids, name='forager'))
    
    return combined.rename('marginal_contribution')


def shapley_values(
    group_members: List[int],
    mu_values: Dict[frozenset, float],
    method: str = "exact",
    n_samples: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute Shapley values for group members.
    
    Parameters
    ----------
    group_members : List[int]
        List of member indices in the group
    mu_values : Dict[frozenset, float]
        Dictionary mapping subsets (as frozensets) to their mu values.
        Must include all subsets of group_members, including empty set.
    method : str, optional
        Computation method: "exact" or "monte_carlo", by default "exact"
    n_samples : int, optional
        Number of Monte Carlo samples (only used if method="monte_carlo")
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping member indices to their Shapley values
    """
    n = len(group_members)
    if n == 0:
        return {}
    
    if method == "exact":
        # Reference function from module globals to handle autoreload issues
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, '_compute_exact_shapley'):
            raise NameError(
                "_compute_exact_shapley not found. "
                "This may be due to a module reload issue. "
                "Try restarting the kernel or reloading the module."
            )
        shapley = current_module._compute_exact_shapley(group_members, mu_values)
    elif method == "monte_carlo":
        if n_samples is None:
            n_samples = 1000
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, '_compute_monte_carlo_shapley'):
            raise NameError(
                "_compute_monte_carlo_shapley not found. "
                "This may be due to a module reload issue. "
                "Try restarting the kernel or reloading the module."
            )
        shapley = current_module._compute_monte_carlo_shapley(group_members, mu_values, n_samples)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'exact' or 'monte_carlo'")
    
    return shapley


def _compute_exact_shapley(
    group_members: List[int],
    mu_values: Dict[frozenset, float],
) -> Dict[int, float]:
    """
    Compute exact Shapley values using the formula.
    
    Optimized with cached factorials to avoid repeated computations.
    """
    n = len(group_members)
    if n == 0:
        return {}
    
    shapley = {member: 0.0 for member in group_members}
    
    # Ensure empty set is in mu_values
    if frozenset() not in mu_values:
        mu_values[frozenset()] = 0.0
    
    # Precompute factorials once (more efficient than computing repeatedly)
    n_factorial = math.factorial(n)
    # Cache factorials for subset sizes
    factorial_cache = {s: math.factorial(s) for s in range(n + 1)}
    
    # For each member, compute their Shapley value
    for member in group_members:
        # Sum over all subsets T not containing member
        other_members = [m for m in group_members if m != member]
        
        for subset_size in range(n):
            # Weight: |T|! * (n - |T| - 1)! / n!
            # Use cached factorials
            weight = (
                factorial_cache[subset_size] 
                * factorial_cache[n - subset_size - 1]
                / n_factorial
            )
            
            for subset in combinations(other_members, subset_size):
                T = frozenset(subset)
                T_with_member = T | {member}
                
                # Marginal contribution
                marginal = mu_values.get(T_with_member, 0.0) - mu_values.get(T, 0.0)
                
                shapley[member] += weight * marginal
    
    return shapley


def _compute_monte_carlo_shapley(
    group_members: List[int],
    mu_values: Dict[frozenset, float],
    n_samples: int,
) -> Dict[int, float]:
    """
    Compute Shapley values using Monte Carlo sampling of permutations.
    
    This version uses precomputed mu_values. For on-the-fly computation,
    use _compute_monte_carlo_shapley_lazy instead.
    """
    n = len(group_members)
    shapley = {member: 0.0 for member in group_members}
    
    # Ensure empty set is in mu_values
    if frozenset() not in mu_values:
        mu_values[frozenset()] = 0.0
    
    # Sample random permutations
    rng = np.random.RandomState(42)  # For reproducibility
    
    for _ in range(n_samples):
        # Random permutation of group members
        perm = rng.permutation(group_members).tolist()
        
        # For each member, compute their marginal contribution in this permutation
        for i, member in enumerate(perm):
            # Members before this one in the permutation
            predecessors = frozenset(perm[:i])
            # Members before plus this member
            predecessors_with_member = predecessors | {member}
            
            # Marginal contribution
            marginal = (
                mu_values.get(predecessors_with_member, 0.0) 
                - mu_values.get(predecessors, 0.0)
            )
            
            shapley[member] += marginal
    
    # Average over samples
    for member in group_members:
        shapley[member] /= n_samples
    
    return shapley


def _compute_monte_carlo_shapley_lazy(
    group_members: List[int],
    mu_func: Callable[[List[int]], float],
    n_samples: int,
    random_seed: int = 42,
) -> Dict[int, float]:
    """
    Compute Shapley values using Monte Carlo sampling with lazy mu computation.
    
    This is more efficient than precomputing all 2^n mu values when using
    Monte Carlo sampling, as it only computes mu for subsets encountered
    during permutation sampling.
    
    Parameters
    ----------
    group_members : List[int]
        List of member indices in the group
    mu_func : Callable[[List[int]], float]
        Function that computes mu for a given subset of members
    n_samples : int
        Number of Monte Carlo samples
    random_seed : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping member indices to their Shapley values
    """
    n = len(group_members)
    if n == 0:
        return {}
    
    shapley = {member: 0.0 for member in group_members}
    
    # Cache for mu values to avoid recomputation
    mu_cache: Dict[frozenset, float] = {frozenset(): mu_func([])}
    
    # Sample random permutations
    rng = np.random.RandomState(random_seed)
    
    for _ in range(n_samples):
        # Random permutation of group members
        perm = rng.permutation(group_members).tolist()
        
        # For each member, compute their marginal contribution in this permutation
        for i, member in enumerate(perm):
            # Members before this one in the permutation
            predecessors = frozenset(perm[:i])
            # Members before plus this member
            predecessors_with_member = predecessors | {member}
            
            # Compute mu values (use cache if available)
            if predecessors not in mu_cache:
                mu_cache[predecessors] = mu_func(list(predecessors))
            if predecessors_with_member not in mu_cache:
                mu_cache[predecessors_with_member] = mu_func(list(predecessors_with_member))
            
            # Marginal contribution
            marginal = mu_cache[predecessors_with_member] - mu_cache[predecessors]
            
            shapley[member] += marginal
    
    # Average over samples
    for member in group_members:
        shapley[member] /= n_samples
    
    return shapley


def _generate_all_subsets(members: List[int]) -> List[frozenset]:
    """Generate all subsets of members."""
    subsets = []
    n = len(members)
    for subset_size in range(n + 1):
        for subset in combinations(members, subset_size):
            subsets.append(frozenset(subset))
    return subsets


def _compute_mu_for_subset(
    subset_members: List[int],
    S_x: np.ndarray,
    intercept_mu: float,
    b_groupsize_mu: float,
    eta_mu: float,
) -> float:
    """
    Compute mu for a subset using the best-top-k formula (non-vectorized, single sample).
    
    Parameters
    ----------
    subset_members : List[int]
        List of forager indices in the subset
    S_x : np.ndarray
        Array of skills for all foragers (indexed by forager), shape (n_foragers,)
    intercept_mu : float
        Intercept parameter for mu
    b_groupsize_mu : float
        Group size coefficient for mu
    eta_mu : float
        Skill elasticity for mu
        
    Returns
    -------
    float
        Expected returns mu for this subset
    """
    if len(subset_members) == 0:
        return 0.0
    
    # Get skills for subset members
    subset_skills = S_x[subset_members]
    
    # Sort descending
    sorted_skills = np.sort(subset_skills)[::-1]
    
    n = len(subset_members)
    
    # Compute values for all k
    values_by_k = np.zeros(n)
    
    for k in range(1, n + 1):
        # Top-k average
        top_k_avg = np.mean(sorted_skills[:k])
        
        # g(k) = exp(intercept_mu + b_groupsize_mu * log(k))
        g_k = np.exp(intercept_mu + b_groupsize_mu * np.log(k))
        
        # Value for this k
        values_by_k[k - 1] = g_k * (top_k_avg ** eta_mu)
    
    # Maximum over k
    return np.max(values_by_k)


def _compute_mu_for_subset_vectorized(
    subset_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Compute mu for a subset using the best-top-k formula, vectorized across MCMC samples.
    
    Parameters
    ----------
    subset_members : List[int]
        List of forager indices in the subset
    S_x : np.ndarray
        Array of skills for all foragers, shape (chain, draw, forager) or (n_samples, forager)
    intercept_mu : np.ndarray
        Intercept parameter, shape (chain, draw) or (n_samples,)
    b_groupsize_mu : np.ndarray
        Group size coefficient, shape (chain, draw) or (n_samples,)
    eta_mu : np.ndarray
        Skill elasticity, shape (chain, draw) or (n_samples,)
        
    Returns
    -------
    np.ndarray
        Expected returns mu for this subset, shape (chain, draw) or (n_samples,)
    """
    if len(subset_members) == 0:
        # Return zeros with same shape as intercept_mu
        return np.zeros_like(intercept_mu)
    
    # Get skills for subset members
    # S_x has shape (chain, draw, forager) or (n_samples, forager)
    # subset_skills will have shape (chain, draw, n_members) or (n_samples, n_members)
    subset_skills = S_x[..., subset_members]
    
    # Sort descending along the forager dimension (last dimension)
    # Use np.sort with axis=-1 and reverse
    sorted_skills = np.sort(subset_skills, axis=-1)[..., ::-1]
    
    n = len(subset_members)
    
    # Compute values for all k, vectorized across MCMC samples
    # values_by_k will have shape (chain, draw, n) or (n_samples, n)
    values_by_k = np.zeros(subset_skills.shape[:-1] + (n,))
    
    for k in range(1, n + 1):
        # Top-k average: mean of first k skills
        # sorted_skills[..., :k] has shape (chain, draw, k) or (n_samples, k)
        top_k_avg = np.mean(sorted_skills[..., :k], axis=-1)  # Shape: (chain, draw) or (n_samples,)
        
        # g(k) = exp(intercept_mu + b_groupsize_mu * log(k))
        # Broadcasting: intercept_mu and b_groupsize_mu have shape (chain, draw) or (n_samples,)
        log_k = np.log(k)
        g_k = np.exp(intercept_mu + b_groupsize_mu * log_k)  # Shape: (chain, draw) or (n_samples,)
        
        # Value for this k: g_k * (top_k_avg ** eta_mu)
        # Broadcasting: top_k_avg and eta_mu have shape (chain, draw) or (n_samples,)
        values_by_k[..., k - 1] = g_k * (top_k_avg ** eta_mu)
    
    # Maximum over k (along last dimension)
    return np.max(values_by_k, axis=-1)  # Shape: (chain, draw) or (n_samples,)


def shapley_contributions(
    S_x: xr.DataArray,
    mu: xr.DataArray,
    data: xr.Dataset,
    group_idx: int,
    intercept_mu: xr.DataArray,
    b_groupsize_mu: xr.DataArray,
    eta_mu: xr.DataArray,
    kcal_scale: float = 1.0,
    method: str = "exact",
    n_samples: Optional[int] = None,
) -> xr.DataArray:
    """
    Compute Shapley values for a specific group using full MCMC posterior distributions.
    
    This function computes Shapley values for each MCMC sample, preserving full posterior uncertainty.
    Uses vectorized operations to efficiently compute mu for all subsets across all MCMC samples.
    
    Parameters
    ----------
    S_x : xr.DataArray
        Skills for all foragers, shape (chain, draw, forager)
        Should be computed using pm.compute_deterministics() with var_names=['S']
    mu : xr.DataArray
        Expected returns for all groups, shape (chain, draw, group)
        Should be computed using pm.compute_deterministics() with var_names=['mu']
    data : xr.Dataset
        Original dataset with forager data
    group_idx : int
        Index of the group to compute Shapley values for
    intercept_mu : xr.DataArray
        Intercept parameter, shape (chain, draw)
    b_groupsize_mu : xr.DataArray
        Group size coefficient, shape (chain, draw)
    eta_mu : xr.DataArray
        Skill elasticity, shape (chain, draw)
    kcal_scale : float, optional
        Scaling factor for kcal (to convert back to original scale), by default 1.0
    method : str, optional
        Computation method: "exact" or "monte_carlo", by default "exact"
    n_samples : int, optional
        Number of Monte Carlo samples (only used if method="monte_carlo")
        
    Returns
    -------
    xr.DataArray
        Shapley values by forager and MCMC sample, shape (chain, draw, forager)
        Uses full posterior distributions for computation
    """
    # Get group members for this group
    group_forager_ids = data.forager_ids.isel(group=group_idx).values
    group_members = [int(fid) for fid in group_forager_ids if fid >= 0]
    
    if len(group_members) == 0:
        # Empty group - return zeros with MCMC dimensions
        n_foragers = len(data.coords["forager"])
        return xr.DataArray(
            np.zeros((len(S_x.chain), len(S_x.draw), n_foragers)),
            dims=["chain", "draw", "forager"],
            coords={
                "chain": S_x.chain,
                "draw": S_x.draw,
                "forager": data.coords["forager"].values,
            },
            name="shapley_contribution"
        )
    
    # Convert to numpy arrays for efficient computation
    S_x_np = S_x.values  # Shape: (chain, draw, forager)
    intercept_mu_np = intercept_mu.values  # Shape: (chain, draw)
    b_groupsize_mu_np = b_groupsize_mu.values  # Shape: (chain, draw)
    eta_mu_np = eta_mu.values  # Shape: (chain, draw)
    
    n_chains, n_draws = intercept_mu_np.shape
    n_group_members = len(group_members)
    
    # Initialize output array: (chain, draw, forager)
    shapley_array = np.zeros((n_chains, n_draws, n_group_members))
    
    # Precompute all subsets only if using exact method
    if method == "exact":
        all_subsets = _generate_all_subsets(group_members)
        
        # Vectorized computation: compute mu for all subsets across all MCMC samples
        # mu_values will be a dict mapping subset -> array of shape (chain, draw)
        mu_values = {}
        for subset in all_subsets:
            mu_vals = _compute_mu_for_subset_vectorized(
                list(subset),
                S_x_np,
                intercept_mu_np,
                b_groupsize_mu_np,
                eta_mu_np,
            )
            mu_values[subset] = mu_vals
        
        # Compute Shapley values for each MCMC sample
        for chain_idx in range(n_chains):
            for draw_idx in range(n_draws):
                # Extract mu values for this sample
                mu_values_sample = {
                    subset: float(mu_vals[chain_idx, draw_idx])
                    for subset, mu_vals in mu_values.items()
                }
                
                # Compute exact Shapley values
                shapley_sample = shapley_values(group_members, mu_values_sample, method="exact")
                
                # Store results
                for i, member in enumerate(group_members):
                    shapley_array[chain_idx, draw_idx, i] = shapley_sample.get(member, 0.0)
    else:
        # Monte Carlo method: compute mu for all subsets across all MCMC samples (vectorized),
        # then take posterior mean and run Monte Carlo Shapley once
        all_subsets = _generate_all_subsets(group_members)
        
        # Vectorized computation: compute mu for all subsets across all MCMC samples
        # mu_values will be a dict mapping subset -> array of shape (chain, draw)
        mu_values_full = {}
        for subset in all_subsets:
            mu_vals = _compute_mu_for_subset_vectorized(
                list(subset),
                S_x_np,
                intercept_mu_np,
                b_groupsize_mu_np,
                eta_mu_np,
            )
            mu_values_full[subset] = mu_vals
        
        # Take posterior mean across (chain, draw) dimensions
        mu_values_mean = {
            subset: float(np.mean(mu_vals))
            for subset, mu_vals in mu_values_full.items()
        }
        
        # Run Monte Carlo Shapley once with mean mu values
        def mu_func(subset_members: List[int]) -> float:
            subset_frozen = frozenset(subset_members)
            return mu_values_mean.get(subset_frozen, 0.0)
        
        shapley_mean = _compute_monte_carlo_shapley_lazy(
            group_members,
            mu_func,
            n_samples=n_samples if n_samples is not None else 1000,
            random_seed=42,
        )
        
        # Broadcast the mean Shapley values to all MCMC samples
        for i, member in enumerate(group_members):
            shapley_array[:, :, i] = shapley_mean.get(member, 0.0)
    
    # Apply scaling
    shapley_array *= kcal_scale
    
    return xr.DataArray(
        shapley_array,
        dims=["chain", "draw", "forager"],
        coords={
            "chain": S_x.chain,
            "draw": S_x.draw,
            "forager": group_members,
        },
        name="shapley_contribution"
    )


def shapley_contributions_all_groups(
    S_x: xr.DataArray,
    mu: xr.DataArray,
    data: xr.Dataset,
    intercept_mu: xr.DataArray,
    b_groupsize_mu: xr.DataArray,
    eta_mu: xr.DataArray,
    kcal_scale: float = 1.0,
    method: str = "exact",
    n_samples: Optional[int] = None,
    max_group_size: Optional[int] = None,
    return_dict: bool = False,
) -> Union[xr.Dataset, Dict[int, xr.DataArray]]:
    """
    Compute Shapley values for all groups in the dataset using full MCMC posterior distributions.
    
    This function expects all pre-computation to be done outside:
    - S_x: Skills for all foragers, shape (chain, draw, forager)
    - mu: Expected returns for all groups, shape (chain, draw, group)
    - intercept_mu, b_groupsize_mu, eta_mu: Parameters with shape (chain, draw)
    
    Uses vectorized operations to efficiently compute Shapley values across all MCMC samples.
    
    Parameters
    ----------
    S_x : xr.DataArray
        Skills for all foragers, shape (chain, draw, forager)
        Should be computed using pm.compute_deterministics() with var_names=['S']
    mu : xr.DataArray
        Expected returns for all groups, shape (chain, draw, group)
        Should be computed using pm.compute_deterministics() with var_names=['mu']
    data : xr.Dataset
        Original dataset with forager data
    intercept_mu : xr.DataArray
        Intercept parameter, shape (chain, draw)
    b_groupsize_mu : xr.DataArray
        Group size coefficient, shape (chain, draw)
    eta_mu : xr.DataArray
        Skill elasticity, shape (chain, draw)
    kcal_scale : float, optional
        Scaling factor for kcal (to convert back to original scale), by default 1.0
    method : str, optional
        Computation method: "exact" or "monte_carlo", by default "exact"
    n_samples : int, optional
        Number of Monte Carlo samples (only used if method="monte_carlo")
    max_group_size : int, optional
        Maximum group size to compute Shapley values for (None = no limit).
        Groups larger than this will be skipped with a warning.
    return_dict : bool, optional
        If True, returns a dictionary mapping group indices to DataArrays (legacy format).
        If False (default), returns an xarray Dataset with dimensions (chain, draw, group, forager).
        
    Returns
    -------
    xr.Dataset or Dict[int, xr.DataArray]
        If return_dict=False: Dataset with 'shapley_contribution' variable,
        dimensions (chain, draw, group, forager), using same coordinates as model.
        Missing values (foragers not in group) are NaN.
        If return_dict=True: Dictionary mapping group indices to their Shapley value DataArrays
        with shape (chain, draw, forager)
    """
    n_groups = len(data.coords['group'])
    n_foragers = len(data.coords['forager'])
    n_chains = len(S_x.chain)
    n_draws = len(S_x.draw)
    
    # Initialize array for all groups and foragers: (chain, draw, group, forager)
    shapley_array = np.full((n_chains, n_draws, n_groups, n_foragers), np.nan, dtype=np.float64)
    
    # Track which groups were computed
    computed_groups = []
    
    for group_idx in range(n_groups):
        group_forager_ids = data.forager_ids.isel(group=group_idx).values
        group_members = [int(fid) for fid in group_forager_ids if fid >= 0]
        group_size = len(group_members)
        
        # Skip if group is too large (exponential complexity)
        if max_group_size is not None and group_size > max_group_size:
            import warnings
            warnings.warn(
                f"Skipping group {group_idx} with size {group_size} "
                f"(exceeds max_group_size={max_group_size})"
            )
            continue
        
        # Skip empty groups
        if group_size == 0:
            continue
        
        # Compute Shapley values for this group using full posterior distributions
        shapley_data = shapley_contributions(
            S_x=S_x,
            mu=mu,
            data=data,
            group_idx=group_idx,
            intercept_mu=intercept_mu,
            b_groupsize_mu=b_groupsize_mu,
            eta_mu=eta_mu,
            kcal_scale=kcal_scale,
            method=method,
            n_samples=n_samples,
        )
        
        # Extract contributions for foragers in this group
        # shapley_data has shape (chain, draw, forager) where forager coords are group_members
        for forager_idx in shapley_data.coords['forager'].values:
            # shapley_data.sel(forager=forager_idx) has shape (chain, draw)
            contrib_values = shapley_data.sel(forager=forager_idx).values
            shapley_array[:, :, group_idx, forager_idx] = contrib_values
        
        computed_groups.append(group_idx)
    
    # Create Dataset with same coordinates as model
    shapley_ds = xr.Dataset(
        {
            'shapley_contribution': (
                ['chain', 'draw', 'group', 'forager'],
                shapley_array,
            )
        },
        coords={
            'chain': S_x.chain,
            'draw': S_x.draw,
            'group': data.coords['group'],
            'forager': data.coords['forager'],
        }
    )
    
    # Add attributes
    shapley_ds['shapley_contribution'].attrs = {
        'long_name': 'Shapley contribution',
        'units': 'kcal',
        'description': 'Shapley value contribution of each forager to each group',
        'method': method,
        'computed_groups': computed_groups,
    }
    
    if return_dict:
        # Return legacy dictionary format
        results = {}
        for group_idx in computed_groups:
            group_forager_ids = data.forager_ids.isel(group=group_idx).values
            group_members = [int(fid) for fid in group_forager_ids if fid >= 0]
            
            # Extract only the foragers in this group
            group_shapley = shapley_ds['shapley_contribution'].sel(group=group_idx).sel(forager=group_members)
            results[group_idx] = group_shapley.rename('shapley_contribution')
        
        return results
    
    return shapley_ds
