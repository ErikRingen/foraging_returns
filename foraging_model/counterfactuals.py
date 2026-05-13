"""
Counterfactual analysis: Shapley value attribution of group returns to foragers.

Uses exact enumeration for small groups (n <= 15) and joint Monte Carlo
sampling for larger groups. All engines are vectorized across posterior samples.
"""
import math
import numpy as np
import xarray as xr
import warnings
from itertools import combinations
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXACT_THRESHOLD = 15  # 2^15 = 32768 subsets
VALID_METHODS = ("auto", "exact", "joint")


# ---------------------------------------------------------------------------
# Internal helpers: mu computation for subsets
# ---------------------------------------------------------------------------

def _compute_mu_for_subset_vectorized(
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
    subset_members : list of int
        Forager indices in the subset.
    S_x : ndarray, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : ndarray, shape (n_samples,)

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    if len(subset_members) == 0:
        return np.zeros(len(intercept_mu))

    n = len(subset_members)
    subset_skills = S_x[:, subset_members]
    sorted_skills = np.sort(subset_skills, axis=1)[:, ::-1]

    cumsum = np.cumsum(sorted_skills, axis=1)
    k_values = np.arange(1, n + 1)
    top_k_avgs = cumsum / k_values[None, :]

    log_k = np.log(k_values)[None, :]
    g_k = np.exp(intercept_mu[:, None] + b_groupsize_mu[:, None] * log_k)
    values = g_k * (top_k_avgs ** eta_mu[:, None])

    return np.max(values, axis=1)


def _compute_mu_for_subset_masked_vectorized(
    subset_mask: np.ndarray,
    S_x_group: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Compute mu for batched masked subsets (used by joint sampling).

    Parameters
    ----------
    subset_mask : ndarray, shape (n_samples, n_steps, n_members)
    S_x_group : ndarray, shape (n_samples, n_members)
    intercept_mu, b_groupsize_mu, eta_mu : ndarray, shape (n_samples,)

    Returns
    -------
    ndarray, shape (n_samples, n_steps)
    """
    n_samples, n_steps, n_members = subset_mask.shape
    if n_members == 0:
        return np.zeros((n_samples, n_steps))

    masked_skills = S_x_group[:, None, :] * subset_mask
    sorted_skills = np.sort(masked_skills, axis=2)[:, :, ::-1]

    cumsum = np.cumsum(sorted_skills, axis=2)
    k_values = np.arange(1, n_members + 1)
    top_k_avgs = cumsum / k_values[None, None, :]

    log_k = np.log(k_values)[None, :]
    g_k = np.exp(intercept_mu[:, None] + b_groupsize_mu[:, None] * log_k)

    values = g_k[:, None, :] * (top_k_avgs ** eta_mu[:, None, None])

    group_sizes = subset_mask.sum(axis=2)
    valid_k = k_values[None, None, :] <= group_sizes[:, :, None]
    values = np.where(valid_k, values, 0.0)

    return np.max(values, axis=2)


# ---------------------------------------------------------------------------
# Internal engines: exact and joint Monte Carlo
# ---------------------------------------------------------------------------

def _shapley_exact_vectorized(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
) -> np.ndarray:
    """
    Exact Shapley values via closed-form enumeration, vectorized over samples.

    Complexity: O(2^n * n) -- suitable for n <= ~15 members.

    Parameters
    ----------
    group_members : list of int
        Forager indices in the group.
    S_x : ndarray, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : ndarray, shape (n_samples,)

    Returns
    -------
    ndarray, shape (n_samples, n_group_members)
    """
    n = len(group_members)
    n_posterior = S_x.shape[0]

    if n == 0:
        return np.zeros((n_posterior, 0))

    member_to_idx = {m: i for i, m in enumerate(group_members)}

    mu_cache: Dict[frozenset, np.ndarray] = {}
    for size in range(n + 1):
        for subset in combinations(group_members, size):
            subset_frozen = frozenset(subset)
            mu_cache[subset_frozen] = _compute_mu_for_subset_vectorized(
                list(subset), S_x, intercept_mu, b_groupsize_mu, eta_mu
            )

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


def _shapley_joint_vectorized(
    group_members: List[int],
    S_x: np.ndarray,
    intercept_mu: np.ndarray,
    b_groupsize_mu: np.ndarray,
    eta_mu: np.ndarray,
    permutations: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Shapley values via joint sampling (one permutation per posterior draw).

    Fully vectorized -- no Python loop over samples or permutations.

    Parameters
    ----------
    group_members : list of int
        Forager indices in the group.
    S_x : ndarray, shape (n_samples, n_foragers)
    intercept_mu, b_groupsize_mu, eta_mu : ndarray, shape (n_samples,)
    permutations : ndarray, optional
        Precomputed permutations in local index space,
        shape (n_samples, n_group_members).
    random_seed : int, optional

    Returns
    -------
    ndarray, shape (n_samples, n_group_members)
    """
    n_members = len(group_members)
    n_samples = S_x.shape[0]

    if n_members == 0:
        return np.zeros((n_samples, 0))

    S_x_group = S_x[:, group_members]

    if permutations is None:
        rng = np.random.RandomState(random_seed)
        permutations = np.array(
            [rng.permutation(n_members) for _ in range(n_samples)],
            dtype=np.int32,
        )
    else:
        if permutations.shape != (n_samples, n_members):
            raise ValueError(
                "permutations must have shape (n_samples, n_group_members)"
            )

    idx = np.arange(n_members)[None, None, :]
    one_hots = permutations[:, :, None] == idx
    cumsum = np.cumsum(one_hots, axis=1).astype(np.float64)
    zeros = np.zeros((n_samples, 1, n_members))
    subset_mask = np.concatenate([zeros, cumsum], axis=1)

    mu_values = _compute_mu_for_subset_masked_vectorized(
        subset_mask, S_x_group, intercept_mu, b_groupsize_mu, eta_mu
    )

    marginals = mu_values[:, 1:] - mu_values[:, :-1]

    shapley = np.zeros((n_samples, n_members))
    row_idx = np.arange(n_samples)[:, None]
    shapley[row_idx, permutations] = marginals

    return shapley


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shapley_contributions_all_groups(
    S_x: xr.DataArray,
    data: xr.Dataset,
    intercept_mu: xr.DataArray,
    b_groupsize_mu: xr.DataArray,
    eta_mu: xr.DataArray,
    kcal_scale: float = 1.0,
    sigma_kcal: Optional[xr.DataArray] = None,
    gp_returns: Optional[xr.DataArray] = None,
    group_date_idx: Optional[np.ndarray] = None,
    proportional_to_observed: bool = True,
    obs_kcal: Optional[np.ndarray] = None,
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
        Skills for all foragers, shape (chain, draw, forager).
    data : xr.Dataset
        Dataset with forager data.
    intercept_mu, b_groupsize_mu, eta_mu : xr.DataArray
        Returns-component parameters with shape (chain, draw).
    kcal_scale : float, default 1.0
        Scaling factor for kcal (converts back to original scale).
    sigma_kcal : xr.DataArray, optional
        Posterior of the LogNormal scale parameter (= ``shape`` in the
        model code), shape (chain, draw). If provided, the per-sample
        Shapley contributions are multiplied by ``exp(sigma_kcal**2 / 2)``
        so that the contributions sum to the **expected (mean) group
        kcal** rather than the median.
    gp_returns : xr.DataArray, optional
        Posterior of the date random effect on returns, shape
        (chain, draw, date). If provided together with
        ``group_date_idx``, per-group per-sample Shapley contributions
        are multiplied by ``exp(gp_returns[date_g, sample])`` so the
        sum across foragers equals that day's expected kcal under the
        full model (not just the date-effect-stripped baseline).
        Shapley axioms are preserved because the multiplier is a
        per-group scalar shared by every member.

        Note: when ``proportional_to_observed=True`` (the default), the
        date-RE factor cancels in the per-group share computation, so
        passing ``gp_returns`` does not affect the final attributions.
        It is still accepted for diagnostics that inspect the
        pre-rescaling values.
    group_date_idx : np.ndarray, optional
        Per-group date index into ``gp_returns``'s ``date`` dim, shape
        (n_groups,). Required if ``gp_returns`` is supplied.
    proportional_to_observed : bool, default True
        If True (default), rescale per-(sample, group) so that the sum
        across foragers equals the *observed* group kcal exactly:
            phi_i  ←  phi_i  ×  Y_obs[g] / sum_j phi_j[s, g]
        Equivalent to allocating the observed total in proportion to
        each forager's Shapley share. The four Shapley axioms hold
        with respect to the observed-total value function (efficiency
        is now against ``Y_obs``, not the model-expected kcal). Use
        this rule when the per-forager numbers are intended to be
        "share of what was actually brought back".
    obs_kcal : np.ndarray, optional
        Observed kcal per group, shape (n_groups,), aligned with
        ``data.coords['group']``. Required when
        ``proportional_to_observed=True``.
    max_group_size : int, optional
        Skip groups larger than this.
    method : str, default "auto"
        - "auto": exact for groups <= 15 members, joint otherwise
        - "exact": always exact (exponential complexity)
        - "joint": one random permutation per posterior sample
    n_posterior_samples : int, default 100
        Number of posterior samples to use (subsampled for speed).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xr.Dataset
        Dataset with 'shapley_contribution' variable,
        dims (sample, group, forager). NaN where a forager is not in a group.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")

    rng = np.random.RandomState(random_seed)

    n_groups = len(data.coords["group"])
    n_foragers = len(data.coords["forager"])

    # Stack chains x draws once, subsample once
    S_x_stacked = S_x.stack(sample=("chain", "draw")).transpose("sample", "forager")
    intercept_stacked = intercept_mu.stack(sample=("chain", "draw"))
    b_groupsize_stacked = b_groupsize_mu.stack(sample=("chain", "draw"))
    eta_stacked = eta_mu.stack(sample=("chain", "draw"))

    n_total = len(S_x_stacked.sample)

    if n_posterior_samples is not None and n_posterior_samples < n_total:
        sample_indices = np.sort(
            rng.choice(n_total, n_posterior_samples, replace=False)
        )
    else:
        sample_indices = np.arange(n_total)
        n_posterior_samples = n_total

    n_samples = len(sample_indices)

    S_x_np = S_x_stacked.isel(sample=sample_indices).values
    intercept_np = intercept_stacked.isel(sample=sample_indices).values
    b_groupsize_np = b_groupsize_stacked.isel(sample=sample_indices).values
    eta_np = eta_stacked.isel(sample=sample_indices).values

    # Optional LogNormal mean-vs-median correction.
    # The deterministic ``mu`` produced by _compute_mu_for_subset_vectorized
    # equals the LogNormal location parameter passed into the likelihood, i.e.
    # the median of the predicted kcal distribution. To convert each subset
    # value to the expected (mean) kcal we multiply by exp(sigma**2 / 2).
    # Shapley axioms are preserved under per-sample scalar multiplication.
    if sigma_kcal is not None:
        sigma_stacked = sigma_kcal.stack(sample=("chain", "draw"))
        sigma_np = sigma_stacked.isel(sample=sample_indices).values
        mean_correction = np.exp(np.asarray(sigma_np) ** 2 / 2)
    else:
        mean_correction = None

    # Optional date random-effect rescaling. Per group, the model's
    # expected kcal is mu_base × kcal_scale × exp(sigma²/2) × exp(gp_t).
    # _compute_mu_for_subset_vectorized produces mu_base only; we apply
    # exp(gp_t[sample]) per group so contributions sum to the full
    # per-day expected kcal.
    if gp_returns is not None:
        if group_date_idx is None:
            raise ValueError(
                "gp_returns supplied without group_date_idx; both required"
            )
        gp_stacked = gp_returns.stack(sample=("chain", "draw"))
        gp_np = gp_stacked.isel(sample=sample_indices).values  # (date, sample)
        # Ensure shape is (date, sample): handle either dim ordering.
        if gp_np.shape[0] == n_samples:
            gp_np = gp_np.T
        date_factor_per_sample = np.exp(np.asarray(gp_np))
    else:
        date_factor_per_sample = None

    shapley_array = np.full(
        (n_samples, n_groups, n_foragers), np.nan, dtype=np.float64
    )

    computed_groups: List[int] = []
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

        use_exact = (
            method == "exact"
            or (method == "auto" and group_size <= EXACT_THRESHOLD)
        )

        if use_exact:
            shapley_group = _shapley_exact_vectorized(
                group_members, S_x_np, intercept_np, b_groupsize_np, eta_np
            )
        else:
            if group_size not in joint_permutations:
                joint_permutations[group_size] = np.array(
                    [rng.permutation(group_size) for _ in range(n_samples)],
                    dtype=np.int32,
                )
            shapley_group = _shapley_joint_vectorized(
                group_members, S_x_np, intercept_np, b_groupsize_np, eta_np,
                permutations=joint_permutations[group_size],
            )

        shapley_group *= kcal_scale
        if mean_correction is not None:
            shapley_group *= mean_correction[:, None]
        if date_factor_per_sample is not None and group_date_idx is not None:
            t = int(group_date_idx[group_idx])
            shapley_group *= date_factor_per_sample[t][:, None]

        if proportional_to_observed:
            if obs_kcal is None:
                raise ValueError(
                    "proportional_to_observed=True requires obs_kcal"
                )
            row_sum = shapley_group.sum(axis=1)
            target = float(obs_kcal[group_idx])
            # Avoid divide-by-zero on rare per-sample-zero rows; keep zero in.
            scale = np.where(row_sum > 0, target / row_sum, 0.0)
            shapley_group = shapley_group * scale[:, None]

        for j, fid in enumerate(group_members):
            shapley_array[:, group_idx, fid] = shapley_group[:, j]

        computed_groups.append(group_idx)

    shapley_ds = xr.Dataset(
        {
            "shapley_contribution": (
                ["sample", "group", "forager"],
                shapley_array,
            )
        },
        coords={
            "sample": np.arange(n_samples),
            "group": data.coords["group"],
            "forager": data.coords["forager"],
        },
    )

    if proportional_to_observed:
        anchored = "observed (proportional residual)"
        desc = (
            "Per-forager kcal share of each group's observed kcal, "
            "allocated in proportion to the per-sample Shapley share. "
            "Sum across foragers equals observed group kcal."
        )
    elif sigma_kcal is not None:
        anchored = "mean"
        desc = (
            "Shapley value contribution of each forager to each group, "
            "in expected (mean) kcal."
        )
    else:
        anchored = "median"
        desc = (
            "Shapley value contribution of each forager to each group, "
            "in median kcal (LogNormal location)."
        )
    shapley_ds["shapley_contribution"].attrs = {
        "long_name": "Shapley contribution",
        "units": "kcal",
        "description": desc,
        "anchored_to": anchored,
        "includes_date_re": "yes" if gp_returns is not None else "no",
        "proportional_to_observed": "yes" if proportional_to_observed else "no",
        "exact_threshold": EXACT_THRESHOLD,
        "n_posterior_samples": n_samples,
        "method": method,
        "computed_groups": computed_groups,
    }

    return shapley_ds
