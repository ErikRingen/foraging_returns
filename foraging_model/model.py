"""
Foraging Returns Model

A Bayesian hierarchical model for foraging effort and production with three
components:
1. Effort (Bernoulli): Whether each forager goes foraging on a given day
2. Success (Bernoulli): Whether a forager returns with food, given they foraged
3. Returns (LogNormal): Group-level kcal returns for successful trips

Note: the posterior variable named ``shape`` is the LogNormal scale (sigma)
parameter, not a Gamma shape — the name is preserved for backward
compatibility with existing ``idata.nc`` files. See ``priors.py``.

Individual foraging skill follows an age-dependent learning-senescence curve with
multiplicative random effects. Correlated random effects on effort and skill
allow estimation of the residual effort-skill association (psi). Temporal
variation is captured by shared-lengthscale Matérn-3/2 Gaussian processes.

The model supports two group skill aggregation methods:
- "best_top_k": Optimized selection of best k foragers (default)
- "mean": Simple mean of forager skills in the group
"""
# pyright: reportOperatorIssue=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportIndexIssue=false
# pyright: reportOptionalSubscript=false, reportGeneralTypeIssues=false
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
# PyTensor tensor stubs are incomplete; arithmetic, indexing, .T, .astype()
# all work at runtime but are untyped.
import xarray as xr
import pymc as pm
import numpy as np
import pandas as pd
import nutpie
import pytensor.tensor as pt
from typing import Dict, Any, Optional

from .counterfactuals import shapley_contributions_all_groups
from .priors import get_priors


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


DEFAULT_PRIORS = get_priors()


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
    Bayesian hierarchical model for foraging effort and production.
    
    The model has three components:
    1. Effort (Bernoulli): P(went foraging | in camp)
    2. Success (Bernoulli): P(returned with food | went foraging)
    3. Returns (LogNormal): Group-level kcal magnitude (the posterior var
       ``shape`` is the LogNormal sigma, kept under that name for
       backward compatibility with cached idata.nc files).
    
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
        measurement_error_age: bool = False,
        use_gp: bool = True,
    ) -> None:
        self.data = data
        self.aggregation_method = aggregation_method
        self.target_scaling = target_scaling
        self.age_scaling = age_scaling
        self.measurement_error_age = measurement_error_age
        self.use_gp = use_gp

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
        if self.measurement_error_age and "age_sigma" not in data.data_vars:
            raise ValueError(
                "measurement_error_age=True requires 'age_sigma' in data. "
                "Ensure preprocessing adds age_sigma to foragers."
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
        """Compute scale factors for kcal and age based on scaling options.

        Note on ``age_scaling='max'`` (used in the canonical fit): the scale
        equals the maximum observed age in the dataset (70 yr for the
        published BaYaka camp). Adding new participants whose ages exceed
        70 yr would change the scale and shift the prior calibration of
        ``m0``, ``k0``, ``b0`` (which are calibrated on age/age_scale).
        Adding only younger participants is safe.
        """
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
        Compute base forager skill curves using the learning-senescence parametrization.
        
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
        sort_indices = pt.argsort(S_x_padded, axis=1)  # pyright: ignore[reportPrivateImportUsage]
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
        coords["re_response"] = ["effort", "skill"]
        coords["gender"] = ["male", "female"]
        self._compute_scale_factors()

        with pm.Model(coords=coords) as self.model:
            # =================================================================
            # Data
            # =================================================================
            kcal_raw = pm.Data('kcal_raw', self.data['kcal'].values, dims="group")
            age_raw = pm.Data('age_raw', self.data['age'].values, dims="forager")

            # Age: optional measurement error.
            # Canonical specification: age_obs ~ Normal(age_true, age_sigma),
            # with a uniform prior on age_true over a plausible range.
            if self.measurement_error_age:
                age_sigma = pm.Data('age_sigma', self.data['age_sigma'].values, dims="forager")
                age_true = pm.Uniform(
                    'age_true', lower=1.0, upper=80.0,
                    dims="forager",
                )
                pm.Normal(
                    'age_obs', mu=age_true, sigma=age_sigma,
                    observed=self.data['age'].values, dims="forager",
                )
                age_for_model = age_true
            else:
                age_for_model = age_raw

            kcal_for_model = kcal_raw

            kcal = pm.Deterministic(
                'kcal_scaled',
                kcal_for_model / self.kcal_scale,
                dims="group"
            )
            age = pm.Deterministic(
                'age_scaled',
                age_for_model / self.age_scale,
                dims="forager"
            )

            # Z-scored age for effort model (better for polynomial regression).
            # Always use observed ages for centering/scaling constants so that
            # z-scores don't shift when age measurement error is enabled.
            age_mean = pt.mean(age_raw)
            age_std = pt.std(age_raw)
            age_z = pm.Deterministic(
                'age_z',
                (age_for_model - age_mean) / age_std,
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
                dims="date"
            )
            
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

            gender_idx = pm.Data(
                "gender_idx",
                self.data["gender_idx"].values,
                dims="forager"
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
            # Correlated Random Effects: (effort, skill), stratified by gender
            # =================================================================
            n_re = 2

            # Separate Cholesky-covariance per gender so the (effort, skill)
            # scales and correlation can differ between men and women.
            # `chol_cov_male` / `chol_cov_female` follow LKJCholeskyCov with
            # the same prior; `corr_*[0,1]` is the per-gender psi.
            chol_male, corr_male, sigma_re_male = pm.LKJCholeskyCov(
                "chol_cov_male",
                n=n_re,
                eta=self.priors["lkj_eta"],
                sd_dist=pm.Exponential.dist(lam=self.priors["sigma_re"]["lam"]),
                compute_corr=True,
            )
            chol_female, corr_female, sigma_re_female = pm.LKJCholeskyCov(
                "chol_cov_female",
                n=n_re,
                eta=self.priors["lkj_eta"],
                sd_dist=pm.Exponential.dist(lam=self.priors["sigma_re"]["lam"]),
                compute_corr=True,
            )

            # Random effects: non-centered parameterization, gender-specific
            # Cholesky factor selected per forager.
            re_raw = pm.Normal(
                "re_raw", mu=0, sigma=1,
                dims=("forager", "re_response"),
            )
            re_male = pt.dot(re_raw, chol_male.T)      # (forager, re_response)
            re_female = pt.dot(re_raw, chol_female.T)  # (forager, re_response)
            # gender_idx: 0=male, 1=female. Broadcast over re_response.
            re = pm.Deterministic(
                "re",
                pt.where(pt.eq(gender_idx[:, None], 0), re_male, re_female),
                dims=("forager", "re_response"),
            )

            # Extract individual random effects
            re_effort = re[:, 0]
            re_skill = re[:, 1]

            # =================================================================
            # Temporal effects: GP or independent date random effects
            # =================================================================
            gp_sigma_effort = _create_prior("gp_sigma_effort", self.priors["gp_sigma_effort"])
            gp_sigma_success = _create_prior("gp_sigma_success", self.priors["gp_sigma_success"])
            gp_sigma_returns = _create_prior("gp_sigma_returns", self.priors["gp_sigma_returns"])

            if self.use_gp:
                gp_lengthscale = _create_prior("gp_lengthscale", self.priors["gp_lengthscale"])
                n_dates = len(self.data.date_numeric.values)
                X_date = date_numeric[:, None]
                matern32_base = pm.gp.cov.Matern32(1, ls=gp_lengthscale)(X_date)
                jitter = 1e-6 * pt.eye(n_dates)

                def _gp_prior(name, sigma):
                    K = sigma**2 * matern32_base + jitter
                    L = pt.linalg.cholesky(K)
                    raw = pm.Normal(f"{name}_raw", 0, 1, dims="date")
                    return pm.Deterministic(name, L @ raw, dims="date")

                gp_effort = _gp_prior("gp_effort", gp_sigma_effort)
                gp_success = _gp_prior("gp_success", gp_sigma_success)
                gp_returns = _gp_prior("gp_returns", gp_sigma_returns)
            else:
                # Independent date random effects (non-centered)
                gp_effort_raw = pm.Normal("gp_effort_raw", 0, 1, dims="date")
                gp_effort = pm.Deterministic("gp_effort", gp_sigma_effort * gp_effort_raw, dims="date")
                gp_success_raw = pm.Normal("gp_success_raw", 0, 1, dims="date")
                gp_success = pm.Deterministic("gp_success", gp_sigma_success * gp_success_raw, dims="date")
                gp_returns_raw = pm.Normal("gp_returns_raw", 0, 1, dims="date")
                gp_returns = pm.Deterministic("gp_returns", gp_sigma_returns * gp_returns_raw, dims="date")

            # =================================================================
            # Gender effect scales (shared within each model component)
            # =================================================================
            # Fixed-effect gender scales (no hierarchical estimation)
            sigma_g_effort = self.priors["sigma_gender_effort"]["sigma"]
            sigma_g_skill = self.priors["sigma_gender_skill"]["sigma"]
            sigma_g_success = self.priors["sigma_gender_success"]["sigma"]

            # =================================================================
            # Priors - LogNormal returns component
            # `shape` is historically named — it is the LogNormal sigma
            # (kept under this name for idata.nc backward compatibility).
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

            m0_gender = pm.ZeroSumNormal("m0_gender", sigma=sigma_g_skill, shape=2, dims="gender")
            k0_gender = pm.ZeroSumNormal("k0_gender", sigma=sigma_g_skill, shape=2, dims="gender")
            b0_gender = pm.ZeroSumNormal("b0_gender", sigma=sigma_g_skill, shape=2, dims="gender")

            m = pm.Deterministic("m", pt.exp(m0 + m0_gender[gender_idx]), dims="forager")
            k = pm.Deterministic("k", pt.exp(k0 + k0_gender[gender_idx]), dims="forager")
            b = pm.Deterministic("b", pt.exp(b0 + b0_gender[gender_idx]), dims="forager")

            # =================================================================
            # Priors - Success component
            # =================================================================
            intercept_success = _create_prior("intercept_success", self.priors["intercept_success"])
            eta_success0 = _create_prior("eta_success0", self.priors["eta_success0"])

            intercept_success_gender = pm.ZeroSumNormal("intercept_success_gender", sigma=sigma_g_success, shape=2, dims="gender")
            eta_success0_gender = pm.ZeroSumNormal("eta_success0_gender", sigma=sigma_g_success, shape=2, dims="gender")

            alpha_success = pm.Deterministic(
                "alpha_success",
                pt.exp(intercept_success + intercept_success_gender[gender_idx]),
                dims="forager"
            )
            eta_success = pm.Deterministic(
                "eta_success",
                pt.exp(eta_success0 + eta_success0_gender[gender_idx]),
                dims="forager"
            )

            # =================================================================
            # Priors - Effort component (age polynomial)
            # =================================================================
            effort_intercept = _create_prior("effort_intercept", self.priors["effort_intercept"])
            effort_age = _create_prior("effort_age", self.priors["effort_age"])
            effort_age2 = _create_prior("effort_age2", self.priors["effort_age2"])

            effort_intercept_gender = pm.ZeroSumNormal("effort_intercept_gender", sigma=sigma_g_effort, shape=2, dims="gender")
            effort_age_gender = pm.ZeroSumNormal("effort_age_gender", sigma=sigma_g_effort, shape=2, dims="gender")
            effort_age2_gender = pm.ZeroSumNormal("effort_age2_gender", sigma=sigma_g_effort, shape=2, dims="gender")

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
            g_effort = gender_idx[effort_forager_idx]
            effort_linear = (
                effort_intercept + effort_intercept_gender[g_effort]
                + (effort_age + effort_age_gender[g_effort]) * age_z[effort_forager_idx]
                + (effort_age2 + effort_age2_gender[g_effort]) * age_z[effort_forager_idx]**2
                + re_effort[effort_forager_idx]
                + gp_effort[effort_date_idx]
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
                pt.log(S_x[forager_idx]**eta_success[forager_idx] * alpha_success[forager_idx] + 1e-10)
                + gp_success[success_date_idx]
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
            # LogNormal Returns Component
            # Includes GP for temporal effects (multiplicative via exp).
            # Every group fitted to the LogNormal has at least one valid
            # forager (filtered upstream in ForagingData.to_dataset), so
            # mu_base > 0 for every group dim — but we still floor at
            # 1e-10 inside ``log`` as a numerical safety net.
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

            # LogNormal likelihood: `shape` is the LogNormal sigma (legacy name).
            log_mu = pt.log(pt.maximum(mu, 1e-10))
            pm.LogNormal(
                "kcal", mu=log_mu, sigma=shape,
                observed=kcal, dims="group",
            )
    
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
        max_group_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> xr.Dataset:
        """
        Compute Shapley values for contribution attribution.
        
        Parameters
        ----------
        idata : InferenceData, optional
            ArviZ InferenceData with posterior samples. If None, uses self.idata.
        method : str, default="auto"
            - "auto": exact for groups <= 15 members, joint MC otherwise
            - "exact": always exact (exponential complexity)
            - "joint": one random permutation per posterior sample
        n_posterior_samples : int, default=100
            Number of posterior samples to use (subsampled for speed).
        max_group_size : int, optional
            Skip groups larger than this.
        random_seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        xr.Dataset
            Dataset with 'shapley_contribution' variable,
            dims (sample, group, forager). NaN where a forager is not in a group.
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
        sigma_kcal = (
            idata.posterior["shape"]
            if "shape" in idata.posterior.data_vars else None
        )
        gp_returns = (
            idata.posterior["gp_returns"]
            if "gp_returns" in idata.posterior.data_vars else None
        )
        group_date_idx = (
            np.asarray(self.data["group_date_idx"].values, dtype=int)
            if "group_date_idx" in self.data.data_vars else None
        )
        obs_kcal = np.asarray(self.data["kcal"].values, dtype=float)

        return shapley_contributions_all_groups(
            S_x=S_x,
            data=self.data,
            intercept_mu=intercept_mu,
            b_groupsize_mu=b_groupsize_mu,
            eta_mu=eta_mu,
            kcal_scale=self.kcal_scale,
            sigma_kcal=sigma_kcal,
            gp_returns=gp_returns,
            group_date_idx=group_date_idx,
            obs_kcal=obs_kcal,
            max_group_size=max_group_size,
            method=method,
            n_posterior_samples=n_posterior_samples,
            random_seed=random_seed,
        )
