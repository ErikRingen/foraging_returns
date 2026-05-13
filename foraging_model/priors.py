"""
Default prior specifications for the foraging returns model.

Single source of truth for priors used by ForagingModel.
"""

PRIORS = {
    "intercept_mu": {"dist": "normal", "mu": 0, "sigma": 0.5},
    "b_groupsize_mu0": {"dist": "normal", "mu": -1, "sigma": 0.5},
    # `shape` is the LogNormal sigma (scale) for the returns likelihood.
    # Named "shape" historically (model was Gamma originally); kept under
    # this name so cached idata.nc files remain readable. Gamma(10, 10)
    # has mean 1, mostly in [0.5, 2] — informative but flexible.
    "shape": {"dist": "gamma", "alpha": 10, "beta": 10},
    # Koster et al. 2020 cross-cultural posterior on `seh` (harvest skill
    # elasticity, log scale): mean of (society x draw) pooled draws,
    # SD pooled to capture both within-society uncertainty and between-site
    # variation. Direct map: Koster's `seh` is the exponent on S(x) in
    # mu_harvest = lm_h * S(x)^exp(seh) — same parameterization as our
    # eta_mu0. See scripts/_derive_priors.py.
    "eta_mu0": {"dist": "normal", "mu": -0.14, "sigma": 0.54},
    # Skill curve priors derived from Koster et al. 2020 cross-cultural
    # posteriors (40 sites, 1821 hunters). See cross_cultural_priors.json.
    # mu = global posterior mean (adjusted for age_scale=70)
    # sigma = sqrt(global_posterior_sd² + between_site_sd²)
    "m0": {"dist": "normal", "mu": -0.37, "sigma": 0.23},
    "k0": {"dist": "normal", "mu": 1.51, "sigma": 0.31},
    "b0": {"dist": "normal", "mu": 0.11, "sigma": 0.37},
    "intercept_success": {"dist": "normal", "mu": 1.5, "sigma": 0.75},
    # Koster et al. 2020 cross-cultural posterior on `sef` (failure skill
    # elasticity, log scale). Map is by interpretation rather than direct
    # form: in Koster, p_failure = 2*(1 - inv_logit(S(x)^exp(sef) * lm_f));
    # in ours, logit(p_success) = const + eta_success0 * log S(x). Both
    # are log-scale exponents on S(x) governing the skill->success/failure
    # coupling, with consistent sign convention (bigger -> tighter coupling).
    "eta_success0": {"dist": "normal", "mu": 0.06, "sigma": 0.57},
    "effort_intercept": {"dist": "normal", "mu": 2, "sigma": 1},
    "effort_age": {"dist": "normal", "mu": 0, "sigma": 1},
    "effort_age2": {"dist": "normal", "mu": 0, "sigma": 1},
    "sigma_re": {"dist": "exponential", "lam": 3},
    "lkj_eta": 1.0,
    "gp_lengthscale": {"dist": "gamma", "alpha": 2, "beta": 2},
    "gp_sigma_effort": {"dist": "exponential", "lam": 2},
    "gp_sigma_success": {"dist": "exponential", "lam": 2},
    "gp_sigma_returns": {"dist": "exponential", "lam": 2},
    # Gender offsets are ZeroSumNormal random variables with a *fixed*
    # scale (not a prior); the values below are the scalar sigma that
    # parametrises ZeroSumNormal in `model.py`. The `{"sigma": 0.5}` form
    # is preserved (rather than a bare scalar) so user-supplied priors=
    # overrides keep working with the dict-shape API.
    "sigma_gender_effort": {"sigma": 0.5},
    "sigma_gender_skill": {"sigma": 0.5},
    "sigma_gender_success": {"sigma": 0.5},
}


def get_priors() -> dict:
    """Return the prior specification dict (copy for safety)."""
    return {k: v.copy() if isinstance(v, dict) else v for k, v in PRIORS.items()}
