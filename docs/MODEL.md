# Model Specification

This document provides the mathematical specification for the foraging returns model. For the canonical specification used in the manuscript (with the latent-age treatment, the no-GP date-RE specification, and Koster cross-cultural priors), refer to `supplement.qmd`. The text here is a methodological overview.

## Model Overview

The model has three components:

1. **Effort (Bernoulli)**: Models probability of going foraging (binary)
2. **Success (Bernoulli)**: Models probability of returning with food given foraging
3. **Returns (LogNormal)**: Models group-level return magnitude on the log scale (the posterior variable named `shape` is the LogNormal sigma — preserved under that name for backward compatibility with cached idata).

These components share:
- A common **skill curve** (age-dependent foraging ability)
- **Correlated random effects** linking individual effort propensity and skill
- **Temporal Gaussian processes** for day-to-day variation

## Skill Curve

Individual forager skill $S(x)$ depends on age $x$ via a learning-senescence parametrization (Koster et al. 2020):

$$S(x) = M(x) \cdot K(x)^b$$

where:

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| $M(x)$ | $\exp(-m \cdot x)$ | Senescence factor (declines with age) |
| $K(x)$ | $1 - \exp(-k \cdot x)$ | Learning factor (increases with age) |
| $b$ | (parameter) | Elasticity of skill on learning |

The skill curve captures the life-history pattern where foraging ability:
- Increases from zero at birth (learning)
- Peaks at some intermediate age
- Declines in later life (senescence)

## Correlated Random Effects

Individual-level variation is captured through correlated random effects:

$$\begin{pmatrix} u_{\text{effort},i} \\ u_{\text{skill},i} \end{pmatrix} \sim \text{MVN}\left(\mathbf{0}, \Sigma\right)$$

The correlation matrix is given an LKJ prior. The final skill for individual $i$ is:

$$S_i = S(x_i) \cdot \exp(u_{\text{skill},i})$$

This allows individuals with higher skill propensity to also have higher/lower effort propensity, capturing unobserved heterogeneity.

## Temporal Gaussian Processes

Day-to-day variation is modeled with shared-lengthscale GPs:

$$\text{GP}(t) \sim \mathcal{GP}(0, K(t, t'))$$

where $K(t, t') = \sigma^2 \left(1 + \frac{\sqrt{3}\,|t-t'|}{\ell}\right) \exp\!\left(-\frac{\sqrt{3}\,|t-t'|}{\ell}\right)$ (Matérn-3/2 kernel).

Separate amplitude parameters ($\sigma$) for effort, success, and returns; shared lengthscale ($\ell$).

## Component 1: Effort (Bernoulli)

For each (forager $i$, date $t$) when in camp:

$$\text{effort}_{i,t} \sim \text{Bernoulli}(p_{i,t})$$

where the probability of going foraging is:

$$\text{logit}(p_{i,t}) = \beta_0 + \beta_1 \cdot \text{age}_z + \beta_2 \cdot \text{age}_z^2 + u_{\text{effort},i} + \text{GP}_{\text{effort}}(t)$$

- $\text{age}_z$ = z-scored age for numerical stability
- $u_{\text{effort},i}$ = individual random effect (correlated with skill)
- $\text{GP}_{\text{effort}}(t)$ = temporal Gaussian process effect

**Eligibility**: Forager was in camp (`in_camp == 1`)

## Component 2: Success (Bernoulli)

For each (forager $i$, date $t$) when they went foraging:

$$\text{success}_{i,t} \sim \text{Bernoulli}(\theta_{i,t})$$

where the success probability is:

$$\text{logit}(\theta_{i,t}) = \log(S_i^{\eta_{\text{success}}} \cdot \alpha_{\text{success}}) + \text{GP}_{\text{success}}(t)$$

with $S_i$ being the individual's skill (includes random effect).

**Eligibility**: Forager was in camp and went foraging

## Component 3: Group Returns (LogNormal)

For each group $g$ with positive returns:

$$\log \text{kcal}_g \sim \mathcal{N}\bigl(\log\mu_g,\,\sigma^2\bigr)$$

i.e. group kcal is LogNormal with median $\mu_g$ and scale $\sigma$ (the latter named `shape` in the code for legacy reasons). The expected return $\mu_g$ depends on the aggregation method; the posterior mean of group kcal is $\mu_g \cdot \exp(\sigma^2/2)$.

### Aggregation Method: Mean (Baseline)

Simple average of forager skills in the group:

$$\mu_g = g(n_g) \cdot \bar{S}_g^{\eta_\mu}$$

where:
- $n_g$ = number of foragers in group $g$
- $\bar{S}_g = \frac{1}{n_g} \sum_{i \in g} S(x_i)$ = mean skill
- $g(n) = \exp(\alpha_\mu + \beta_n \log n)$ = group size effect
- $\eta_\mu$ = skill elasticity

### Aggregation Method: Best-Top-K

Optimal selection of $k$ best foragers:

$$\mu_g = \max_{1 \leq k \leq n_g} \left[ g(k) \cdot \bar{S}_{g,k}^{\eta_\mu} \right]$$

where:
- $\bar{S}_{g,k}$ = mean of top $k$ skills in group $g$
- $g(k) = \exp(\alpha_\mu + \beta_n \log k)$

This finds the optimal subset size $k$ that balances:
- Including more foragers (higher $g(k)$)
- Maintaining high average skill (which decreases as $k$ increases)

## Priors

### Default Priors

| Parameter | Distribution | Parameters | Description |
|-----------|--------------|------------|-------------|
| **Skill Curve** (Koster cross-cultural; see `cross_cultural_priors.json`) | | | |
| `m0` | `normal` | mu=-0.37, sigma=0.23 | Log senescence rate |
| `k0` | `normal` | mu=1.51, sigma=0.31 | Log learning rate |
| `b0` | `normal` | mu=0.11, sigma=0.37 | Log skill elasticity on learning |
| **Gender offsets** (zero-sum) | | | |
| `sigma_gender_*` | (scalar) | sigma=0.5 | Half-Normal scale for ZeroSumNormal gender offsets on skill / effort / success |
| **Effort** | | | |
| `effort_intercept` | `normal` | mu=0, sigma=1 | Effort intercept |
| `effort_age` | `normal` | mu=0, sigma=1 | Effort age coefficient |
| `effort_age2` | `normal` | mu=0, sigma=1 | Effort age² coefficient |
| **Success** | | | |
| `intercept_success` | `normal` | mu=0, sigma=0.5 | Log success intercept |
| `eta_success0` | `normal` | mu=0, sigma=1 | Log skill elasticity (success) |
| **Returns** | | | |
| `intercept_mu` | `normal` | mu=0, sigma=0.5 | Intercept for μ |
| `b_groupsize_mu0` | `normal` | mu=-1, sigma=0.5 | Log group size coefficient |
| `eta_mu0` | `normal` | mu=0, sigma=1 | Log skill elasticity (returns) |
| `shape` | `gamma` | alpha=10, beta=10 | LogNormal sigma (named `shape` for legacy compat) |
| **Random Effects** | | | |
| `sigma_re` | `exponential` | lam=3 | SD for each random effect |
| `lkj_eta` | (scalar) | 1.0 | LKJ concentration parameter |
| **Gaussian Processes** | | | |
| `gp_lengthscale` | `gamma` | alpha=2, beta=2 | Shared GP lengthscale (days) |
| `gp_sigma_effort` | `exponential` | lam=2 | GP amplitude for effort |
| `gp_sigma_success` | `exponential` | lam=2 | GP amplitude for success |
| `gp_sigma_returns` | `exponential` | lam=2 | GP amplitude for returns |

Parameters with `0` suffix are on the log scale; the actual parameter is exp(·) for positivity.

### Customizing Priors

Pass a `priors` dictionary to override specific defaults. Each prior spec has `dist` (distribution name) plus distribution parameters:

```python
# Change parameters only (keeps default distribution)
model = ForagingModel(data, priors={
    "m0": {"mu": -1.5, "sigma": 0.5},
})

# Change distribution type
model = ForagingModel(data, priors={
    "shape": {"dist": "exponential", "lam": 0.1},
    "eta_mu0": {"dist": "halfnormal", "sigma": 1},
})

# Use informative priors
model = ForagingModel(data, priors={
    "m0": {"dist": "normal", "mu": -2, "sigma": 0.3},
    "k0": {"dist": "normal", "mu": 0.5, "sigma": 0.3},
})
```

### Available Distributions

`normal`, `halfnormal`, `gamma`, `beta`, `exponential`, `uniform`, `halfcauchy`, `studentt`, `lognormal`

## Dimensions

| Dimension | Description |
|-----------|-------------|
| `forager` | Individual forager identifiers |
| `group` | Group/trip identifiers |
| `forager_in_group` | Position within group (max group size, padded) |
| `effort_obs` | (forager, date) observations for effort model |
| `forager_date` | (forager, date) observations for success model |
| `date` | Unique dates in the study period |
| `re_response` | Random effect responses: ["effort", "skill"] |

## Special Values

| Value | Meaning |
|-------|---------|
| `-1` | Invalid/padding entry in `forager_ids` |
| `-99` | Forager present but not in database |

## Model Comparison

To compare aggregation methods using LOO-CV:

```python
import arviz as az

model_topk = ForagingModel(data, aggregation_method="best_top_k")
model_mean = ForagingModel(data, aggregation_method="mean")

idata_topk = model_topk.fit()
idata_mean = model_mean.fit()

comparison = az.compare({
    "best_top_k": idata_topk, 
    "mean": idata_mean
})
```

## References

- Koster, J., et al. (2020). The life history of human foraging: Cross-cultural and individual variation.
- PyMC documentation: https://www.pymc.io/

