"""
Counterfactual analysis utilities for foraging model.

Provides Shapley value computation for attributing group returns to individual foragers.
Uses exact computation for small groups (faster and exact) and Monte Carlo for large groups.
"""
import math
import numpy as np
import xarray as xr
import warnings
from itertools import combinations
from typing import Dict, List, Optional


def compute_mu_for_subset(
    subset_members: List[int],
    S_x: np.ndarray,
    intercept_mu: float,
    b_groupsize_mu: float,
    eta_mu: float,
) -> float:
    """
    Compute mu for a subset using the best-top-k formula (single sample).
    
    Parameters
    ----------
    subset_members : List[int]
        Forager indices in the subset
    S_x : np.ndarray
        Skills for all foragers, shape (n_foragers,)
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
    
    subset_skills = S_x[subset_members]
    sorted_skills = np.sort(subset_skills)[::-1]
    n = len(subset_members)
    
    # Compute g(k) * (top-k average)^eta for all k, return max
    k_values = np.arange(1, n + 1)
    cumsum = np.cumsum(sorted_skills)
    top_k_avgs = cumsum / k_values
    g_k = np.exp(intercept_mu + b_groupsize_mu * np.log(k_values))
    values = g_k * (top_k_avgs ** eta_mu)
    
    return float(np.max(values))


def compute_mu_for_subset_vectorized(
    subset_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Compute mu for a subset, vectorized across posterior samples.
    
    Parameters
    ----------
    subset_members : List[int]
        Forager indices in the subset
    S_x : np.ndarray
        Skills for all foragers, shape (n_samples, n_foragers)
    intercept_mu : np.ndarray
        Intercept parameter, shape (n_samples,)
    b_groupsize_mu : np.ndarray
        Group size coefficient, shape (n_samples,)
    eta_mu : np.ndarray
        Skill elasticity, shape (n_samples,)
        
    Returns
    -------
    np.ndarray
        Expected returns mu for this subset, shape (n_samples,)
    """
    if len(subset_members) == 0:
        return np.zeros(len(intercept_mu))
    
    n_samples = S_x.shape[0]
    n = len(subset_members)
    
    # Get subset skills: (n_samples, n_subset)
    subset_skills = S_x[:, subset_members]
    
    # Sort descending along subset axis
    sorted_skills = np.sort(subset_skills, axis=1)[:, ::-1]
    
    # Cumsum and top-k averages: (n_samples, n_subset)
    cumsum = np.cumsum(sorted_skills, axis=1)
    k_values = np.arange(1, n + 1)  # (n_subset,)
    top_k_avgs = cumsum / k_values[None, :]  # (n_samples, n_subset)
    
    # g(k) depends on parameters: (n_samples, n_subset)
    log_k = np.log(k_values)[None, :]  # (1, n_subset)
    g_k = np.exp(
        intercept_mu[:, None] + b_groupsize_mu[:, None] * log_k
    )  # (n_samples, n_subset)
    
    # Compute values: (n_samples, n_subset)
    # Need to handle eta_mu per sample
    values = g_k * (top_k_avgs ** eta_mu[:, None])
    
    # Return max over k for each sample
    return np.max(values, axis=1)


def compute_mu_for_subset_masked_vectorized(
    subset_mask: np.ndarray,
    S_x_group: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Compute mu for masked subsets, vectorized across posterior samples.

    Parameters
    ----------
    subset_mask : np.ndarray
        Binary mask for subsets, shape (n_samples, n_steps, n_members)
    S_x_group : np.ndarray
        Skills for group members, shape (n_samples, n_members)
    intercept_mu, b_groupsize_mu, eta_mu : np.ndarray
        Model parameters, shape (n_samples,)

    Returns
    -------
    np.ndarray
        Mu for each subset step, shape (n_samples, n_steps)
    """
    n_samples, n_steps, n_members = subset_mask.shape

    if n_members == 0:
        return np.zeros((n_samples, n_steps))

    # Masked skills: (n_samples, n_steps, n_members)
    masked_skills = S_x_group[:, None, :] * subset_mask

    # Sort descending along member axis
    sorted_skills = np.sort(masked_skills, axis=2)[:, :, ::-1]

    # Cumsum and top-k averages
    cumsum = np.cumsum(sorted_skills, axis=2)
    k_values = np.arange(1, n_members + 1)  # (n_members,)
    top_k_avgs = cumsum / k_values[None, None, :]

    # g(k) for each sample
    log_k = np.log(k_values)[None, :]  # (1, n_members)
    g_k = np.exp(
        intercept_mu[:, None] + b_groupsize_mu[:, None] * log_k
    )  # (n_samples, n_members)

    # Values for each k and step
    values = g_k[:, None, :] * (top_k_avgs ** eta_mu[:, None, None])

    # Mask invalid k (k > group_size)
    group_sizes = subset_mask.sum(axis=2)  # (n_samples, n_steps)
    valid_k = k_values[None, None, :] <= group_sizes[:, :, None]
    values = np.where(valid_k, values, 0.0)

    # Max over k
    return np.max(values, axis=2)


def shapley_monte_carlo_joint_vectorized(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
    permutations: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Shapley values using joint sampling (one permutation per sample).

    Parameters
    ----------
    group_members : List[int]
        Forager indices in the group
    S_x : np.ndarray
        Skills for all foragers, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : np.ndarray
        Model parameters, shape (n_samples,)
    permutations : np.ndarray, optional
        Precomputed permutations in local group index space,
        shape (n_samples, n_group_members)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Shapley values, shape (n_samples, n_group_members)
    """
    n_members = len(group_members)
    n_samples = S_x.shape[0]

    if n_members == 0:
        return np.zeros((n_samples, 0))

    # Work in local index space for the group
    S_x_group = S_x[:, group_members]  # (n_samples, n_members)

    if permutations is None:
        rng = np.random.RandomState(random_seed)
        permutations = np.array(
            [rng.permutation(n_members) for _ in range(n_samples)],
            dtype=np.int32
        )
    else:
        if permutations.shape != (n_samples, n_members):
            raise ValueError(
                "permutations must have shape (n_samples, n_group_members)"
            )

    # Build subset masks from permutations: (n_samples, n_members + 1, n_members)
    idx = np.arange(n_members)[None, None, :]
    one_hots = permutations[:, :, None] == idx
    cumsum = np.cumsum(one_hots, axis=1).astype(np.float64)
    zeros = np.zeros((n_samples, 1, n_members))
    subset_mask = np.concatenate([zeros, cumsum], axis=1)

    # Compute mu for all subset steps
    mu_values = compute_mu_for_subset_masked_vectorized(
        subset_mask, S_x_group, intercept_mu, b_groupsize_mu, eta_mu
    )  # (n_samples, n_members + 1)

    # Marginal contributions per position
    marginals = mu_values[:, 1:] - mu_values[:, :-1]

    # Scatter marginals to local member indices
    shapley = np.zeros((n_samples, n_members))
    row_idx = np.arange(n_samples)[:, None]
    shapley[row_idx, permutations] = marginals

    return shapley


def shapley_exact(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: float,
    b_groupsize_mu: float,
    eta_mu: float,
) -> Dict[int, float]:
    """
    Compute exact Shapley values using the closed-form formula (single sample).
    
    Complexity: O(2^n * n) - suitable for groups up to ~15 members.
    
    Parameters
    ----------
    group_members : List[int]
        Forager indices in the group
    S_x : np.ndarray
        Skills for all foragers, shape (n_foragers,)
    intercept_mu, b_groupsize_mu, eta_mu : float
        Model parameters
        
    Returns
    -------
    Dict[int, float]
        Shapley values for each group member
    """
    n = len(group_members)
    if n == 0:
        return {}
    
    # Precompute all 2^n mu values
    mu_cache: Dict[frozenset, float] = {}
    for size in range(n + 1):
        for subset in combinations(group_members, size):
            subset_frozen = frozenset(subset)
            mu_cache[subset_frozen] = compute_mu_for_subset(
                list(subset), S_x, intercept_mu, b_groupsize_mu, eta_mu
            )
    
    # Precompute factorials
    n_factorial = math.factorial(n)
    factorial_cache = {s: math.factorial(s) for s in range(n + 1)}
    
    shapley = {member: 0.0 for member in group_members}
    
    for member in group_members:
        other_members = [m for m in group_members if m != member]
        
        for subset_size in range(n):
            weight = (
                factorial_cache[subset_size] 
                * factorial_cache[n - subset_size - 1]
                / n_factorial
            )
            
            for subset in combinations(other_members, subset_size):
                T = frozenset(subset)
                T_with_member = T | {member}
                marginal = mu_cache[T_with_member] - mu_cache[T]
                shapley[member] += weight * marginal
    
    return shapley


def shapley_exact_vectorized(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Compute exact Shapley values, vectorized across posterior samples.
    
    Complexity: O(2^n * n) - suitable for groups up to ~15 members.
    
    Parameters
    ----------
    group_members : List[int]
        Forager indices in the group
    S_x : np.ndarray
        Skills for all foragers, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : np.ndarray
        Model parameters, shape (n_samples,)
        
    Returns
    -------
    np.ndarray
        Shapley values, shape (n_samples, n_group_members)
    """
    n = len(group_members)
    n_posterior = S_x.shape[0]
    
    if n == 0:
        return np.zeros((n_posterior, 0))
    
    member_to_idx = {m: i for i, m in enumerate(group_members)}
    
    # Precompute all 2^n mu values for all samples
    mu_cache: Dict[frozenset, np.ndarray] = {}
    for size in range(n + 1):
        for subset in combinations(group_members, size):
            subset_frozen = frozenset(subset)
            mu_cache[subset_frozen] = compute_mu_for_subset_vectorized(
                list(subset), S_x, intercept_mu, b_groupsize_mu, eta_mu
            )
    
    # Precompute factorials
    n_factorial = math.factorial(n)
    factorial_cache = {s: math.factorial(s) for s in range(n + 1)}
    
    shapley = np.zeros((n_posterior, n))
    
    for member in group_members:
        other_members = [m for m in group_members if m != member]
        member_idx = member_to_idx[member]
        
        for subset_size in range(n):
            weight = (
                factorial_cache[subset_size] 
                * factorial_cache[n - subset_size - 1]
                / n_factorial
            )
            
            for subset in combinations(other_members, subset_size):
                T = frozenset(subset)
                T_with_member = T | {member}
                marginal = mu_cache[T_with_member] - mu_cache[T]
                shapley[:, member_idx] += weight * marginal
    
    return shapley


def shapley_monte_carlo(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: float,
    b_groupsize_mu: float,
    eta_mu: float,
    n_samples: int = 5000,
    random_seed: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute Shapley values using Monte Carlo sampling (single sample).
    
    Uses lazy mu computation with caching. Only useful for very large groups (n > 15).
    
    Parameters
    ----------
    group_members : List[int]
        Forager indices in the group
    S_x : np.ndarray
        Skills for all foragers
    intercept_mu, b_groupsize_mu, eta_mu : float
        Model parameters
    n_samples : int, default=5000
        Number of permutation samples
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[int, float]
        Shapley values for each group member
    """
    n = len(group_members)
    if n == 0:
        return {}
    
    shapley = {member: 0.0 for member in group_members}
    mu_cache: Dict[frozenset, float] = {frozenset(): 0.0}
    
    rng = np.random.RandomState(random_seed)
    
    for _ in range(n_samples):
        perm = rng.permutation(group_members).tolist()
        
        for i, member in enumerate(perm):
            predecessors = frozenset(perm[:i])
            predecessors_with_member = predecessors | {member}
            
            if predecessors not in mu_cache:
                mu_cache[predecessors] = compute_mu_for_subset(
                    list(predecessors), S_x, intercept_mu, b_groupsize_mu, eta_mu
                )
            if predecessors_with_member not in mu_cache:
                mu_cache[predecessors_with_member] = compute_mu_for_subset(
                    list(predecessors_with_member), S_x, intercept_mu, b_groupsize_mu, eta_mu
                )
            
            marginal = mu_cache[predecessors_with_member] - mu_cache[predecessors]
            shapley[member] += marginal
    
    for member in group_members:
        shapley[member] /= n_samples
    
    return shapley


def shapley_monte_carlo_vectorized(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
    n_permutations: int = 100,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Shapley values using Monte Carlo, vectorized across posterior samples.
    
    Parameters
    ----------
    group_members : List[int]
        Forager indices in the group
    S_x : np.ndarray
        Skills for all foragers, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : np.ndarray
        Model parameters, shape (n_samples,)
    n_permutations : int, default=100
        Number of permutation samples
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Shapley values, shape (n_samples, n_group_members)
    """
    n = len(group_members)
    n_posterior = S_x.shape[0]
    
    if n == 0:
        return np.zeros((n_posterior, 0))
    
    # Map member index to position in output array
    member_to_idx = {m: i for i, m in enumerate(group_members)}
    
    # Accumulate Shapley values: (n_posterior, n_members)
    shapley = np.zeros((n_posterior, n))
    
    # Cache for mu values: subset -> (n_posterior,) array
    mu_cache: Dict[frozenset, np.ndarray] = {
        frozenset(): np.zeros(n_posterior)
    }
    
    rng = np.random.RandomState(random_seed)
    
    for _ in range(n_permutations):
        perm = rng.permutation(group_members).tolist()
        
        for i, member in enumerate(perm):
            predecessors = frozenset(perm[:i])
            predecessors_with_member = predecessors | {member}
            
            # Compute mu for predecessors if not cached
            if predecessors not in mu_cache:
                mu_cache[predecessors] = compute_mu_for_subset_vectorized(
                    list(predecessors), S_x, intercept_mu, b_groupsize_mu, eta_mu
                )
            
            # Compute mu for predecessors + member if not cached
            if predecessors_with_member not in mu_cache:
                mu_cache[predecessors_with_member] = compute_mu_for_subset_vectorized(
                    list(predecessors_with_member), S_x, 
                    intercept_mu, b_groupsize_mu, eta_mu
                )
            
            # Marginal contribution: (n_posterior,)
            marginal = mu_cache[predecessors_with_member] - mu_cache[predecessors]
            shapley[:, member_to_idx[member]] += marginal
    
    return shapley / n_permutations


# Threshold for switching from exact to Monte Carlo
EXACT_THRESHOLD = 15  # 2^15 = 32768 subsets

# Valid computation methods
VALID_METHODS = ("auto", "exact", "monte_carlo", "joint")


def shapley_contributions(
    S_x: xr.DataArray,
    data: xr.Dataset,
    group_idx: int,
    intercept_mu: xr.DataArray,
    b_groupsize_mu: xr.DataArray,
    eta_mu: xr.DataArray,
    kcal_scale: float = 1.0,
    n_permutations: int = 100,
    method: str = "auto",
    n_posterior_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> xr.DataArray:
    """
    Compute Shapley values for a specific group, preserving posterior uncertainty.
    
    Uses vectorized computation across all posterior samples (no Python loops).
    
    Parameters
    ----------
    S_x : xr.DataArray
        Skills for all foragers, shape (chain, draw, forager)
    data : xr.Dataset
        Dataset with forager data
    group_idx : int
        Index of the group to compute Shapley values for
    intercept_mu : xr.DataArray
        Intercept parameter, shape (chain, draw)
    b_groupsize_mu : xr.DataArray
        Group size coefficient, shape (chain, draw)
    eta_mu : xr.DataArray
        Skill elasticity, shape (chain, draw)
    kcal_scale : float, default=1.0
        Scaling factor for kcal (converts back to original scale)
    n_permutations : int, default=100
        Monte Carlo permutations (for method='monte_carlo')
        Ignored for method='joint' (one permutation per draw)
    method : str, default="auto"
        Computation method:
        - "auto": Use exact for groups ≤ 15 members, Monte Carlo otherwise
        - "exact": Always use exact computation (exponential complexity)
        - "monte_carlo": Always use Monte Carlo approximation
        - "joint": One permutation per posterior sample (joint sampling)
    n_posterior_samples : int, optional
        Number of posterior samples to use. If None, uses all samples.
        Subsampling speeds up computation while preserving uncertainty.
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    xr.DataArray
        Shapley values, shape (sample, forager)
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
    
    group_forager_ids = data.forager_ids.isel(group=group_idx).values
    group_members = [int(fid) for fid in group_forager_ids if fid >= 0]
    
    rng = np.random.RandomState(random_seed)
    
    # Stack chains and draws into flat samples: (n_total, n_foragers)
    S_x_stacked = S_x.stack(sample=("chain", "draw")).transpose("sample", "forager")
    intercept_stacked = intercept_mu.stack(sample=("chain", "draw"))
    b_groupsize_stacked = b_groupsize_mu.stack(sample=("chain", "draw"))
    eta_stacked = eta_mu.stack(sample=("chain", "draw"))
    
    n_total_samples = len(S_x_stacked.sample)
    
    # Subsample posterior if requested
    if n_posterior_samples is not None and n_posterior_samples < n_total_samples:
        sample_indices = rng.choice(n_total_samples, n_posterior_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = np.arange(n_total_samples)
        n_posterior_samples = n_total_samples
    
    if len(group_members) == 0:
        return xr.DataArray(
            np.zeros((len(sample_indices), 0)),
            dims=["sample", "forager"],
            coords={"sample": sample_indices, "forager": []},
            name="shapley_contribution"
        )
    
    n_group_members = len(group_members)
    
    # Extract numpy arrays for selected samples: (n_samples, ...)
    S_x_np = S_x_stacked.isel(sample=sample_indices).values
    intercept_np = intercept_stacked.isel(sample=sample_indices).values
    b_groupsize_np = b_groupsize_stacked.isel(sample=sample_indices).values
    eta_np = eta_stacked.isel(sample=sample_indices).values
    
    # Choose method based on parameter or group size
    if method == "auto":
        use_exact = n_group_members <= EXACT_THRESHOLD
    elif method == "exact":
        use_exact = True
    elif method == "monte_carlo":
        use_exact = False
    else:  # joint
        use_exact = False
    
    # Vectorized computation - no Python loop over samples!
    if use_exact:
        shapley_array = shapley_exact_vectorized(
            group_members, S_x_np, intercept_np, b_groupsize_np, eta_np
        )
    elif method == "joint":
        shapley_array = shapley_monte_carlo_joint_vectorized(
            group_members, S_x_np, intercept_np, b_groupsize_np, eta_np,
            random_seed=rng.randint(0, 2**31),
        )
    else:
        shapley_array = shapley_monte_carlo_vectorized(
            group_members, S_x_np, intercept_np, b_groupsize_np, eta_np,
            n_permutations=n_permutations,
            random_seed=rng.randint(0, 2**31),
        )
    
    shapley_array *= kcal_scale
    
    return xr.DataArray(
        shapley_array,
        dims=["sample", "forager"],
        coords={
            "sample": sample_indices,
            "forager": group_members,
        },
        name="shapley_contribution"
    )


def shapley_contributions_all_groups(
    S_x: xr.DataArray,
    data: xr.Dataset,
    intercept_mu: xr.DataArray,
    b_groupsize_mu: xr.DataArray,
    eta_mu: xr.DataArray,
    kcal_scale: float = 1.0,
    n_permutations: int = 100,
    max_group_size: Optional[int] = None,
    method: str = "auto",
    n_posterior_samples: int = 100,
    random_seed: Optional[int] = None,
) -> xr.Dataset:
    """
    Compute Shapley values for all groups in the dataset.
    
    Parameters
    ----------
    S_x : xr.DataArray
        Skills for all foragers, shape (chain, draw, forager)
    data : xr.Dataset
        Dataset with forager data
    intercept_mu : xr.DataArray
        Intercept parameter, shape (chain, draw)
    b_groupsize_mu : xr.DataArray
        Group size coefficient, shape (chain, draw)
    eta_mu : xr.DataArray
        Skill elasticity, shape (chain, draw)
    kcal_scale : float, default=1.0
        Scaling factor for kcal
    n_permutations : int, default=100
        Monte Carlo permutations per posterior sample (for method='monte_carlo')
        Ignored for method='joint' (one permutation per draw)
    max_group_size : int, optional
        Skip groups larger than this
    method : str, default="auto"
        Computation method:
        - "auto": Use exact for groups ≤ 15 members, Monte Carlo otherwise
        - "exact": Always use exact computation (exponential complexity)
        - "monte_carlo": Always use Monte Carlo approximation
        - "joint": One permutation per posterior sample (joint sampling)
    n_posterior_samples : int, default=100
        Number of posterior samples to use. Subsampling speeds up computation
        while preserving uncertainty quantification.
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    xr.Dataset
        Dataset with 'shapley_contribution' variable,
        dimensions (sample, group, forager).
        NaN for foragers not in a given group.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")
    
    rng = np.random.RandomState(random_seed)
    
    n_groups = len(data.coords['group'])
    n_foragers = len(data.coords['forager'])
    
    # Stack chains and draws into flat samples: (n_total, n_foragers)
    S_x_stacked = S_x.stack(sample=("chain", "draw")).transpose("sample", "forager")
    intercept_stacked = intercept_mu.stack(sample=("chain", "draw"))
    b_groupsize_stacked = b_groupsize_mu.stack(sample=("chain", "draw"))
    eta_stacked = eta_mu.stack(sample=("chain", "draw"))

    n_total_samples = len(S_x_stacked.sample)

    # Subsample posterior if requested
    if n_posterior_samples is not None and n_posterior_samples < n_total_samples:
        sample_indices = rng.choice(n_total_samples, n_posterior_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = np.arange(n_total_samples)
        n_posterior_samples = n_total_samples

    n_samples_actual = len(sample_indices)
    
    shapley_array = np.full(
        (n_samples_actual, n_groups, n_foragers), 
        np.nan, 
        dtype=np.float64
    )
    
    computed_groups = []
    
    # Joint sampling uses one permutation per posterior sample (per group size).
    joint_permutations: Dict[int, np.ndarray] = {}

    for group_idx in range(n_groups):
        group_forager_ids = data.forager_ids.isel(group=group_idx).values
        group_members = [int(fid) for fid in group_forager_ids if fid >= 0]
        group_size = len(group_members)
        
        if group_size == 0:
            continue
            
        if max_group_size is not None and group_size > max_group_size:
            warnings.warn(
                f"Skipping group {group_idx} with size {group_size} "
                f"(exceeds max_group_size={max_group_size})"
            )
            continue
        
        if method == "joint":
            if group_size not in joint_permutations:
                joint_permutations[group_size] = np.array(
                    [rng.permutation(group_size) for _ in range(n_samples_actual)],
                    dtype=np.int32,
                )

            shapley_group = shapley_monte_carlo_joint_vectorized(
                group_members,
                S_x_stacked.isel(sample=sample_indices).values,
                intercept_stacked.isel(sample=sample_indices).values,
                b_groupsize_stacked.isel(sample=sample_indices).values,
                eta_stacked.isel(sample=sample_indices).values,
                permutations=joint_permutations[group_size],
            )
            shapley_group *= kcal_scale
            shapley_data = xr.DataArray(
                shapley_group,
                dims=["sample", "forager"],
                coords={
                    "sample": sample_indices,
                    "forager": group_members,
                },
                name="shapley_contribution",
            )
        else:
            shapley_data = shapley_contributions(
                S_x=S_x,
                data=data,
                group_idx=group_idx,
                intercept_mu=intercept_mu,
                b_groupsize_mu=b_groupsize_mu,
                eta_mu=eta_mu,
                kcal_scale=kcal_scale,
                n_permutations=n_permutations,
                method=method,
                n_posterior_samples=n_posterior_samples,
                random_seed=rng.randint(0, 2**31),
            )
        
        for forager_idx in shapley_data.coords['forager'].values:
            contrib_values = shapley_data.sel(forager=forager_idx).values
            shapley_array[:, group_idx, forager_idx] = contrib_values
        
        computed_groups.append(group_idx)
    
    shapley_ds = xr.Dataset(
        {
            'shapley_contribution': (
                ['sample', 'group', 'forager'],
                shapley_array,
            )
        },
        coords={
            'sample': np.arange(n_samples_actual),
            'group': data.coords['group'],
            'forager': data.coords['forager'],
        }
    )
    
    shapley_ds['shapley_contribution'].attrs = {
        'long_name': 'Shapley contribution',
        'units': 'kcal',
        'description': 'Shapley value contribution of each forager to each group',
        'exact_threshold': EXACT_THRESHOLD,
        'n_permutations': n_permutations,
        'n_posterior_samples': n_samples_actual,
        'method': method,
        'computed_groups': computed_groups,
    }
    
    return shapley_ds
