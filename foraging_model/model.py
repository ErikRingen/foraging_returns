"""
Foraging Returns Model

A Bayesian model for foraging returns with two components:
1. Bernoulli component: Forager-daily success probability
2. Gamma component: Group-trip returns magnitude

The model supports two group skill aggregation methods:
- "best_top_k": Optimized selection of best k foragers (default)
- "mean": Simple mean of forager skills in the group
"""
import xarray as xr
import pymc as pm
import numpy as np
import pandas as pd
import nutpie
import pytensor.tensor as pt
from typing import Dict, Any, Optional

from .counterfactuals import shapley_contributions_all_groups


# Mapping from string names to PyMC distributions
DISTRIBUTIONS = {
    "normal": pm.Normal,
    "halfnormal": pm.HalfNormal,
    "truncatednormal": pm.TruncatedNormal,
    "gamma": pm.Gamma,
    "beta": pm.Beta,
    "exponential": pm.Exponential,
    "uniform": pm.Uniform,
    "halfcauchy": pm.HalfCauchy,
    "studentt": pm.StudentT,
    "lognormal": pm.LogNormal,
}


# Default prior specifications
# Each prior has "dist" (distribution name) and distribution-specific parameters
DEFAULT_PRIORS = {
    # Gamma component
    "intercept_mu": {"dist": "normal", "mu": 0, "sigma": 0.5},
    "b_groupsize_mu0": {"dist": "normal", "mu": -1, "sigma": 0.5},  # log-scale, exp(-1) ≈ 0.37
    "shape": {"dist": "gamma", "alpha": 10, "beta": 10},
    "eta_mu0": {"dist": "normal", "mu": 0, "sigma": 1},
    
    # Skill curve (log-scale)
    "m0": {"dist": "normal", "mu": -1, "sigma": 1},
    "k0": {"dist": "normal", "mu": 1, "sigma": 1},
    "b0": {"dist": "normal", "mu": 0, "sigma": 1},
    
    # Success component
    "intercept_success": {"dist": "normal", "mu": 0, "sigma": 0.5},
    "eta_success0": {"dist": "normal", "mu": 0, "sigma": 1},
    
    # Effort component (age polynomial) - binary
    "effort_intercept": {"dist": "normal", "mu": 0, "sigma": 1},
    "effort_age": {"dist": "normal", "mu": 0, "sigma": 1},
    "effort_age2": {"dist": "normal", "mu": 0, "sigma": 1},
    
    # Correlated random effects: (effort, skill)
    "sigma_re": {"dist": "exponential", "lam": 3},  # SD for each random effect
    "lkj_eta": 2.0,  # LKJ concentration (higher = less correlation)
    
    # Gaussian Process for temporal effects (date)
    "gp_lengthscale": {"dist": "gamma", "alpha": 2, "beta": 2},  # ~1 day expected, day-to-day variation
    "gp_sigma_effort": {"dist": "exponential", "lam": 2},  # GP amplitude for effort (binary)
    "gp_sigma_success": {"dist": "exponential", "lam": 2},  # GP amplitude for success
    "gp_sigma_returns": {"dist": "exponential", "lam": 2},  # GP amplitude for returns
}


def _create_prior(name: str, spec: Dict[str, Any]):
    """Create a PyMC prior from a specification dict."""
    spec = spec.copy()
    dist_name = spec.pop("dist", "normal").lower()
    
    if dist_name not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{dist_name}'. "
            f"Available: {list(DISTRIBUTIONS.keys())}"
        )
    
    dist_class = DISTRIBUTIONS[dist_name]
    return dist_class(name, **spec)


class ForagingModel:
    """
    Bayesian model for foraging returns.
    
    The model has two components:
    1. Bernoulli: Models whether each forager succeeds on a given day
    2. Gamma: Models the magnitude of group-level returns
    
    The model is built automatically on instantiation.
    
    Parameters
    ----------
    data : xr.Dataset
        Dataset with unscaled data (from ForagingData.to_dataset())
    aggregation_method : str, default="best_top_k"
        Method for aggregating forager skills to group level:
        - "best_top_k": Select optimal k foragers
        - "mean": Simple average of all forager skills
    priors : dict, optional
        Dictionary of prior specifications. Keys are parameter names,
        values are dicts with "dist" (distribution name) and parameters.
        Missing keys use defaults. See DEFAULT_PRIORS for structure.
    target_scaling : str | None, optional
        Scaling method for target variable ('mean', 'max', or None)
    age_scaling : str | None, optional
        Scaling method for age variable ('mean', 'max', or None)
    
    Examples
    --------
    >>> # Use default priors
    >>> model = ForagingModel(data)
    
    >>> # Custom distribution and parameters
    >>> model = ForagingModel(data, priors={
    ...     "m0": {"dist": "normal", "mu": -2, "sigma": 0.5},
    ...     "shape": {"dist": "exponential", "lam": 0.1},
    ... })
    
    >>> # Use HalfNormal for positive parameters
    >>> model = ForagingModel(data, priors={
    ...     "eta_mu0": {"dist": "halfnormal", "sigma": 1},
    ... })
    """
    
    VALID_AGGREGATION_METHODS = ("best_top_k", "mean")
    VALID_SCALING_OPTIONS = ("mean", "max", None)
    
    def __init__(
        self,
        data: xr.Dataset,
        aggregation_method: str = "best_top_k",
        priors: Optional[Dict[str, Dict[str, Any]]] = None,
        target_scaling: str | None = None,
        age_scaling: str | None = None,
    ) -> None:
        self.data = data
        self.aggregation_method = aggregation_method
        self.target_scaling = target_scaling
        self.age_scaling = age_scaling
        
        # Merge user priors with defaults
        self.priors = {
            k: v.copy() if isinstance(v, dict) else v 
            for k, v in DEFAULT_PRIORS.items()
        }
        if priors is not None:
            for key, value in priors.items():
                if key not in self.priors:
                    raise ValueError(
                        f"Unknown prior '{key}'. Valid priors: {list(DEFAULT_PRIORS.keys())}"
                    )
                # Handle both dict priors and scalar hyperparameters
                if isinstance(self.priors[key], dict) and isinstance(value, dict):
                    self.priors[key] = {**self.priors[key], **value}
                else:
                    self.priors[key] = value
        
        # Validate options
        if self.aggregation_method not in self.VALID_AGGREGATION_METHODS:
            raise ValueError(
                f"aggregation_method must be one of {self.VALID_AGGREGATION_METHODS}, "
                f"got '{self.aggregation_method}'"
            )
        if self.target_scaling not in self.VALID_SCALING_OPTIONS:
            raise ValueError(
                f"target_scaling must be one of {self.VALID_SCALING_OPTIONS}, "
                f"got '{self.target_scaling}'"
            )
        if self.age_scaling not in self.VALID_SCALING_OPTIONS:
            raise ValueError(
                f"age_scaling must be one of {self.VALID_SCALING_OPTIONS}, "
                f"got '{self.age_scaling}'"
            )
        
        # Build the model
        self._build_model()
    
    # =========================================================================
    # Coordinate Preparation
    # =========================================================================
    
    def _prepare_coords(self) -> dict:
        """Prepare coordinates for PyMC model, converting to nutpie-compatible format."""
        coords = {}
        for key, value in self.data.coords.items():
            if hasattr(value, 'to_numpy'):
                try:
                    coord_values = value.to_numpy()
                    if pd.api.types.is_datetime64_any_dtype(coord_values.dtype):
                        coords[key] = np.arange(len(coord_values)).tolist()
                    elif (coord_values.dtype == object or 
                          (hasattr(coord_values.dtype, 'kind') and 
                           coord_values.dtype.kind in ['U', 'S', 'O'])):
                        coords[key] = np.arange(len(coord_values)).tolist()
                    else:
                        coords[key] = coord_values.tolist()
                except (ValueError, TypeError):
                    coords[key] = (
                        list(range(len(value))) if hasattr(value, '__len__') else [0]
                    )
            else:
                coords[key] = value
        return coords
    
    def _compute_scale_factors(self) -> None:
        """Compute scale factors for kcal and age based on scaling options."""
        if self.target_scaling == 'mean':
            target_col = self.data['kcal']
            non_zero = target_col.where(target_col != 0)
            self.kcal_scale = float(non_zero.mean().values)
        elif self.target_scaling == 'max':
            self.kcal_scale = float(self.data['kcal'].max().values)
        else:
            self.kcal_scale = 1.0
        
        if self.age_scaling == 'mean':
            age_col = self.data['age']
            non_zero = age_col.where(age_col != 0)
            self.age_scale = float(non_zero.mean().values)
        elif self.age_scaling == 'max':
            self.age_scale = float(self.data['age'].max().values)
        else:
            self.age_scale = 1.0
    
    # =========================================================================
    # Skill Curve Computation
    # =========================================================================
    
    def _compute_skill_curves(self, age, m, k, b):
        """
        Compute base forager skill curves using the M-K-B parametrization.
        
        S_base(x) = M(x) * K(x)^b
        
        where:
            M(x) = exp(-m * x)  : Senescence (declining with age)
            K(x) = 1 - exp(-k * x)  : Learning (increasing with age)
            b : Elasticity of skill on learning
            
        Returns the base skill (without random effects).
        The final S is computed in _build_model with multiplicative RE.
        """
        M_x = pm.Deterministic("M", pt.exp(-m * age), dims="forager")
        K_x = pm.Deterministic("K", 1 - pt.exp(-k * age), dims="forager")
        S_x_base = pm.Deterministic("S_base", M_x * K_x**b, dims="forager")
        return M_x, K_x, S_x_base
    
    # =========================================================================
    # Group Skill Aggregation Methods
    # =========================================================================
    
    def _compute_group_mu(self, S_x, forager_ids, valid_foragers, intercept_mu, 
                          b_groupsize_mu, eta_mu):
        """Compute expected group returns (mu) using the configured aggregation method."""
        if self.aggregation_method == "best_top_k":
            return self._compute_group_mu_best_top_k(
                S_x, forager_ids, valid_foragers, intercept_mu, b_groupsize_mu, eta_mu
            )
        else:
            return self._compute_group_mu_mean(
                S_x, forager_ids, valid_foragers, intercept_mu, b_groupsize_mu, eta_mu
            )
    
    def _compute_group_mu_mean(self, S_x, forager_ids, valid_foragers, intercept_mu,
                               b_groupsize_mu, eta_mu):
        """
        Compute group mu using simple mean of forager skills.
        
        mu = exp(intercept_mu + b_groupsize_mu * log(n)) * mean(S_x)^eta_mu
        """
        safe_forager_ids = pt.where(valid_foragers, forager_ids, 0)
        S_x_selected = S_x[safe_forager_ids]
        S_x_masked = pt.where(valid_foragers, S_x_selected, 0.0)
        
        group_sizes = pt.sum(valid_foragers, axis=1).astype('float64')
        group_sizes_safe = pt.maximum(group_sizes, 1.0)
        
        skill_sums = pt.sum(S_x_masked, axis=1)
        mean_skills = skill_sums / group_sizes_safe
        mean_skills_safe = pt.maximum(mean_skills, 1e-10)
        
        log_n = pt.log(group_sizes_safe)
        g_n = pt.exp(intercept_mu + b_groupsize_mu * log_n)
        
        mu_raw = g_n * (mean_skills_safe ** eta_mu)
        valid_groups = group_sizes > 0
        mu = pt.where(valid_groups, mu_raw, 0.0)
        
        return mu  # Raw tensor, Deterministic created in _build_model
    
    def _compute_group_mu_best_top_k(self, S_x, forager_ids, valid_foragers, 
                                      intercept_mu, b_groupsize_mu, eta_mu):
        """
        Compute group mu using best-top-k optimization.
        
        mu(S) = max_{1 <= k <= |S|} g(k) * (avg of top k skills)^eta
        where g(k) = exp(intercept_mu + b_groupsize_mu * log(k))
        """
        safe_forager_ids = pt.where(valid_foragers, forager_ids, 0)
        S_x_selected = S_x[safe_forager_ids]
        
        valid_groups = pt.sum(valid_foragers, axis=1) > 0
        
        max_group_size = int(self.data.forager_ids.shape[1])
        n_groups = len(self.data.coords['group'])
        
        # Sort skills descending
        S_x_neg = -S_x_selected
        S_x_padded = pt.where(valid_foragers, S_x_neg, 1e10)
        sort_indices = pt.argsort(S_x_padded, axis=1)
        row_indices = pt.arange(n_groups)[:, None]
        S_x_sorted = -S_x_padded[row_indices, sort_indices]
        
        group_sizes_int = pt.sum(valid_foragers, axis=1)
        group_sizes_float = pt.cast(group_sizes_int, dtype='float64')
        
        valid_mask = pt.cast(
            pt.arange(max_group_size, dtype='int32')[None, :] < 
            pt.cast(group_sizes_int, dtype='int32')[:, None],
            dtype='float64'
        )
        
        cumsum = pt.cumsum(pt.cast(S_x_sorted, dtype='float64') * valid_mask, axis=1)
        k_values = pt.arange(1, max_group_size + 1, dtype='float64')
        
        log_k = pt.log(k_values)
        g_k = pt.exp(intercept_mu + b_groupsize_mu * log_k)
        
        k_matrix = pt.cast(
            pt.arange(max_group_size, dtype='int32') + 1, dtype='float64'
        )[None, :]
        group_sizes_expanded = group_sizes_float[:, None]
        
        valid_k_mask = k_matrix <= group_sizes_expanded
        top_k_averages = pt.where(valid_k_mask, cumsum / k_matrix, 0.0)
        
        top_k_averages_masked = top_k_averages * valid_k_mask
        top_k_averages_safe = pt.maximum(top_k_averages_masked, 1e-10)
        g_k_expanded = g_k[None, :]
        values_by_k = g_k_expanded * (top_k_averages_safe ** eta_mu)
        
        mu_best_topk = pt.max(values_by_k, axis=1)
        mu_best_topk_safe = pt.where(valid_groups, mu_best_topk, 0.0)
        
        return mu_best_topk_safe  # Raw tensor, Deterministic created in _build_model
    
    # =========================================================================
    # Model Building
    # =========================================================================
    
    def _build_model(self):
        """Build the PyMC model."""
        coords = self._prepare_coords()
        # Random effects: effort, skill
        coords["re_response"] = ["effort", "skill"]
        coords["unique_date"] = self.data.unique_date.values
        self._compute_scale_factors()
        
        n_foragers = len(self.data.coords['forager'])

        with pm.Model(coords=coords) as self.model:
            # =================================================================
            # Data
            # =================================================================
            kcal_raw = pm.Data('kcal_raw', self.data['kcal'].values, dims="group")
            age_raw = pm.Data('age_raw', self.data['age'].values, dims="forager")
            
            kcal = pm.Deterministic(
                'kcal_scaled', 
                kcal_raw / self.kcal_scale if self.kcal_scale != 1.0 else kcal_raw,
                dims="group"
            )
            age = pm.Deterministic(
                'age_scaled',
                age_raw / self.age_scale if self.age_scale != 1.0 else age_raw,
                dims="forager"
            )
            
            # Z-scored age for effort model (better for polynomial regression)
            age_mean = pt.mean(age_raw)
            age_std = pt.std(age_raw)
            age_z = pm.Deterministic(
                'age_z',
                (age_raw - age_mean) / age_std,
                dims="forager"
            )
            
            forager_ids = pm.Data(
                "forager_ids", 
                self.data.forager_ids.values, 
                dims=("group", "forager_in_group")
            )
            
            # Success data
            forager_success = pm.Data(
                'forager_success', 
                self.data.forager_success.values, 
                dims="forager_date"
            )
            forager_idx = pm.Data(
                'forager_idx', 
                self.data.forager_idx.values, 
                dims="forager_date"
            )
            
            # Effort data
            forager_effort = pm.Data(
                'forager_effort',
                self.data.forager_effort.values,
                dims="effort_obs"
            )
            effort_forager_idx = pm.Data(
                'effort_forager_idx',
                self.data.effort_forager_idx.values,
                dims="effort_obs"
            )
            
            # Date data for Gaussian Processes
            date_numeric = pm.Data(
                'date_numeric',
                self.data.date_numeric.values,
                dims="unique_date"
            )
            n_dates = len(self.data.unique_date)
            
            effort_date_idx = pm.Data(
                'effort_date_idx',
                self.data.effort_date_idx.values,
                dims="effort_obs"
            )
            success_date_idx = pm.Data(
                'success_date_idx',
                self.data.success_date_idx.values,
                dims="forager_date"
            )
            group_date_idx = pm.Data(
                'group_date_idx',
                self.data.group_date_idx.values,
                dims="group"
            )
            
            # Group Structure
            valid_foragers = forager_ids >= 0
            present_foragers = pt.or_(forager_ids >= 0, pt.eq(forager_ids, -99))
            group_size = pm.Deterministic(
                "group_size", 
                pt.sum(present_foragers, axis=1), 
                dims="group"
            )

            # =================================================================
            # Correlated Random Effects: (effort, skill)
            # =================================================================
            n_re = 2
            
            chol, corr, sigma_re = pm.LKJCholeskyCov(
                "chol_cov",
                n=n_re,
                eta=self.priors["lkj_eta"],
                sd_dist=pm.Exponential.dist(lam=self.priors["sigma_re"]["lam"]),
                compute_corr=True,
            )
            
            # Random effects: non-centered parameterization
            re_raw = pm.Normal("re_raw", mu=0, sigma=1, shape=(n_foragers, n_re))
            re = pm.Deterministic(
                "re",
                pt.dot(re_raw, chol.T),
                dims=("forager", "re_response")
            )
            
            # Extract individual random effects
            re_effort = re[:, 0]
            re_skill = re[:, 1]

            # =================================================================
            # Gaussian Processes for temporal effects
            # One latent GP value per unique date, shared across observations
            # =================================================================
            
            # Shared lengthscale across all GPs (in days)
            gp_lengthscale = _create_prior("gp_lengthscale", self.priors["gp_lengthscale"])
            
            # GP amplitudes (separate for each response)
            gp_sigma_effort = _create_prior("gp_sigma_effort", self.priors["gp_sigma_effort"])
            gp_sigma_success = _create_prior("gp_sigma_success", self.priors["gp_sigma_success"])
            gp_sigma_returns = _create_prior("gp_sigma_returns", self.priors["gp_sigma_returns"])
            
            # Build covariance matrix using squared exponential kernel
            # K(t, t') = sigma^2 * exp(-0.5 * ((t - t') / l)^2)
            date_diff = date_numeric[:, None] - date_numeric[None, :]  # (n_dates, n_dates)
            K_base = pt.exp(-0.5 * (date_diff / gp_lengthscale) ** 2)
            
            # Add small jitter for numerical stability
            jitter = 1e-4
            K_base_jittered = K_base + jitter * pt.eye(n_dates)
            
            # Cholesky decomposition for sampling
            L_cov = pt.linalg.cholesky(K_base_jittered)
            
            # Latent GP values (non-centered parameterization)
            gp_effort_raw = pm.Normal("gp_effort_raw", mu=0, sigma=1, dims="unique_date")
            gp_success_raw = pm.Normal("gp_success_raw", mu=0, sigma=1, dims="unique_date")
            gp_returns_raw = pm.Normal("gp_returns_raw", mu=0, sigma=1, dims="unique_date")
            
            # Transform to correlated GP values
            gp_effort = pm.Deterministic(
                "gp_effort",
                gp_sigma_effort * pt.dot(L_cov, gp_effort_raw),
                dims="unique_date"
            )
            gp_success = pm.Deterministic(
                "gp_success",
                gp_sigma_success * pt.dot(L_cov, gp_success_raw),
                dims="unique_date"
            )
            gp_returns = pm.Deterministic(
                "gp_returns",
                gp_sigma_returns * pt.dot(L_cov, gp_returns_raw),
                dims="unique_date"
            )

            # =================================================================
            # Priors - Gamma component
            # =================================================================
            intercept_mu = _create_prior("intercept_mu", self.priors["intercept_mu"])
            shape = _create_prior("shape", self.priors["shape"])
            
            b_groupsize_mu0 = _create_prior("b_groupsize_mu0", self.priors["b_groupsize_mu0"])
            b_groupsize_mu = pm.Deterministic("b_groupsize_mu", pt.exp(b_groupsize_mu0))
            
            eta_mu0 = _create_prior("eta_mu0", self.priors["eta_mu0"])
            eta_mu = pm.Deterministic("eta_mu", pt.exp(eta_mu0))

            # =================================================================
            # Priors - Skill curve (log-scale for positivity)
            # =================================================================
            m0 = _create_prior("m0", self.priors["m0"])
            k0 = _create_prior("k0", self.priors["k0"])
            b0 = _create_prior("b0", self.priors["b0"])
            
            m = pm.Deterministic("m", pt.exp(m0))
            k = pm.Deterministic("k", pt.exp(k0))
            b = pm.Deterministic("b", pt.exp(b0))

            # =================================================================
            # Priors - Success component
            # =================================================================
            intercept_success = _create_prior("intercept_success", self.priors["intercept_success"])
            alpha_success = pm.Deterministic("alpha_success", pt.exp(intercept_success))
            
            eta_success0 = _create_prior("eta_success0", self.priors["eta_success0"])
            eta_success = pm.Deterministic("eta_success", pt.exp(eta_success0))

            # =================================================================
            # Priors - Effort component (age polynomial)
            # =================================================================
            effort_intercept = _create_prior("effort_intercept", self.priors["effort_intercept"])
            effort_age = _create_prior("effort_age", self.priors["effort_age"])
            effort_age2 = _create_prior("effort_age2", self.priors["effort_age2"])

            # =================================================================
            # Skill Curves (with multiplicative random effect)
            # =================================================================
            M_x, K_x, S_x_base = self._compute_skill_curves(age, m, k, b)
            
            # Apply multiplicative random effect: S_i = S(x_i) * exp(u_skill_i)
            S_x = pm.Deterministic(
                "S",
                S_x_base * pt.exp(re_skill),
                dims="forager"
            )

            # =================================================================
            # Effort Component: P(went foraging | in camp)
            # Uses z-scored age for numerical stability in polynomial
            # Includes GP for temporal effects
            # =================================================================
            effort_linear = (
                effort_intercept 
                + effort_age * age_z[effort_forager_idx]
                + effort_age2 * age_z[effort_forager_idx]**2
                + re_effort[effort_forager_idx]
                + gp_effort[effort_date_idx]  # Temporal GP effect
            )
            p_effort = pm.Deterministic(
                "p_effort",
                pt.sigmoid(effort_linear),
                dims="effort_obs"
            )
            pm.Bernoulli(
                "effort",
                p=p_effort,
                observed=forager_effort,
                dims="effort_obs"
            )

            # =================================================================
            # Success Component: P(returned with food | went foraging)
            # Individual variation captured through skill RE (S_x includes re_skill)
            # Includes GP for temporal effects
            # =================================================================
            success_linear = (
                pt.log(S_x[forager_idx]**eta_success * alpha_success + 1e-10)
                + gp_success[success_date_idx]  # Temporal GP effect
            )
            theta = pm.Deterministic(
                "theta", 
                pt.sigmoid(success_linear),
                dims="forager_date"
            )
            pm.Bernoulli(
                "non_zero_prod", 
                p=theta, 
                observed=forager_success, 
                dims="forager_date"
            )

            # =================================================================
            # Gamma Component: Group Returns
            # Includes GP for temporal effects (multiplicative via exp)
            # =================================================================
            mu_base = self._compute_group_mu(
                S_x, forager_ids, valid_foragers, 
                intercept_mu, b_groupsize_mu, eta_mu
            )
            
            # Apply temporal GP effect multiplicatively
            mu = pm.Deterministic(
                "mu",
                mu_base * pt.exp(gp_returns[group_date_idx]),
                dims="group"
            )
            
            rate = pm.Deterministic("rate", shape / mu, dims="group")
            pm.Gamma("kcal", alpha=shape, beta=rate, observed=kcal, dims="group")
    
    # =========================================================================
    # Model Fitting
    # =========================================================================
        
    def fit(
        self,
        tune: int = 500,
        draws: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        **kwargs
    ):
        """
        Fit the model using nutpie sampler.
        
        Parameters
        ----------
        tune : int, default=500
            Number of tuning steps
        draws : int, default=1000
            Number of posterior samples per chain
        chains : int, default=4
            Number of chains
        random_seed : int | None, default=None
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to nutpie.sample()
        
        Returns
        -------
        InferenceData
            ArviZ InferenceData object with posterior samples
        """
        compiled_model = nutpie.compile_pymc_model(self.model, backend="jax")
        
        sample_kwargs = {
            "tune": tune,
            "draws": draws,
            "chains": chains,
            **kwargs
        }
        if random_seed is not None:
            sample_kwargs["seed"] = random_seed
        
        self.idata = nutpie.sample(compiled_model, **sample_kwargs)
        return self.idata
    
    def rescale_predictive(self):
        """Rescale predictive samples back to original scale."""
        if not hasattr(self, "idata"):
            raise ValueError("Model must be fitted before rescaling. Call fit() first.")
        
        if getattr(self, "_predictive_rescaled", False):
            return self
        
        self.idata.posterior_predictive["kcal"] *= self.kcal_scale
        self.idata.prior_predictive["kcal"] *= self.kcal_scale
        if "kcal" in self.idata.observed_data:
            self.idata.observed_data["kcal"] *= self.kcal_scale

        self._predictive_rescaled = True
        return self
    
    # =========================================================================
    # Shapley Value Computation
    # =========================================================================
    
    def compute_shapley(
        self,
        idata=None,
        method: str = "auto",
        n_posterior_samples: int = 100,
        n_permutations: int = 100,
        max_group_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> xr.Dataset:
        """
        Compute Shapley values for contribution attribution.
        
        This method handles all the internal parameter extraction, so you don't
        need to manually pass intercept_mu, b_groupsize_mu, etc.
        
        Parameters
        ----------
        idata : InferenceData, optional
            ArviZ InferenceData with posterior samples. If None, uses self.idata.
        method : str, default="auto"
            Computation method:
            - "auto": Use exact for groups ≤ 15 members, Monte Carlo otherwise
            - "exact": Always use exact computation (exponential complexity)
            - "monte_carlo": Always use Monte Carlo approximation
        n_posterior_samples : int, default=100
            Number of posterior samples to use. Subsampling speeds up computation
            while preserving uncertainty quantification.
        n_permutations : int, default=100
            Monte Carlo permutations per posterior sample (for method='monte_carlo')
        max_group_size : int, optional
            Skip groups larger than this
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        xr.Dataset
            Dataset with 'shapley_contribution' variable,
            dimensions (sample, group, forager).
            NaN for foragers not in a given group.
            
        Examples
        --------
        >>> model = ForagingModel(data)
        >>> idata = model.fit()
        >>> shapley_ds = model.compute_shapley(method="monte_carlo", n_posterior_samples=100)
        """
        if idata is None:
            if not hasattr(self, "idata"):
                raise ValueError(
                    "No idata provided and model hasn't been fitted. "
                    "Either call fit() first or pass idata explicitly."
                )
            idata = self.idata
        
        # Compute deterministic variables (S) for all MCMC samples
        with self.model:
            deterministics = pm.compute_deterministics(
                dataset=idata.posterior,
                var_names=['S']
            )
        
        S_x = deterministics['S']
        
        # Extract posterior parameters
        intercept_mu = idata.posterior["intercept_mu"]
        b_groupsize_mu = idata.posterior["b_groupsize_mu"]
        eta_mu = idata.posterior["eta_mu"]
        
        return shapley_contributions_all_groups(
            S_x=S_x,
            data=self.data,
            intercept_mu=intercept_mu,
            b_groupsize_mu=b_groupsize_mu,
            eta_mu=eta_mu,
            kcal_scale=self.kcal_scale,
            n_permutations=n_permutations,
            max_group_size=max_group_size,
            method=method,
            n_posterior_samples=n_posterior_samples,
            random_seed=random_seed,
        )
