# Shapley Values for Group Contribution Attribution

## Overview

Shapley values provide a fair, additive, and order-invariant way to attribute group-level returns (kcal) to individual group members. 

The implementation supports multiple methods:
- **Exact computation** for groups ≤ 15 members (faster and exact)
- **Monte Carlo sampling** for larger groups (averages over permutations)
- **Joint sampling** (one permutation per posterior draw)

## Mathematical Foundation

Shapley values satisfy four key properties:

1. **Efficiency**: Contributions sum to total value: $\sum_{i \in S} \phi_i(S) = \mu(S)$
2. **Symmetry**: Identical contributors get equal values
3. **Dummy Player**: Members who don't contribute get zero
4. **Additivity**: Contributions are additive across different group compositions

The Shapley value for member $i$ in group $S$ is:

$$
\phi_i(S) = \sum_{T \subseteq S \setminus \{i\}} \frac{|T|! (|S| - |T| - 1)!}{|S|!} \left[\mu(T \cup \{i\}) - \mu(T)\right]
$$

where:
- $T$ is a subset of $S$ excluding member $i$
- $\mu(T)$ is the expected returns for subset $T$ (computed using the model's aggregation formula)
- $\mu(T \cup \{i\}) - \mu(T)$ is the marginal contribution of member $i$ to subset $T$

## Implementation

The implementation automatically selects the optimal method:

**Exact method (groups ≤ 15):**
1. Precompute μ for all 2^n subsets
2. Apply Shapley formula directly
3. O(2^n) complexity, exact results

**Monte Carlo method (groups > 15):**
1. Sample random permutations of group members
2. For each permutation, compute each member's marginal contribution (with caching)
3. Average over permutations
4. O(n_samples × n) complexity, approximate results

**Joint sampling method:**
1. Sample **one permutation per posterior draw**
2. Compute marginal contributions from that single permutation
3. Treats permutation uncertainty as part of each draw (joint sampling)
4. O(n_draws × n) complexity, efficient and scalable

### Usage

```python
from foraging_model.counterfactuals import shapley_contributions_all_groups
import pymc as pm

# After fitting the model
model = ForagingModel(data=dataset)
model.build_model()
idata = model.fit()

# Compute deterministics for Shapley calculation
with model.model:
    idata = pm.compute_deterministics(idata, var_names=["S"])

# Extract required parameters from posterior
posterior = idata.posterior
S_x = posterior["S"]
intercept_mu = posterior["intercept_mu"]
b_groupsize_mu = posterior["b_groupsize_mu"]
eta_mu = posterior["eta_mu"]

# Compute Shapley values for all groups (Monte Carlo average)
shapley_ds = shapley_contributions_all_groups(
    S_x=S_x,
    data=dataset,
    intercept_mu=intercept_mu,
    b_groupsize_mu=b_groupsize_mu,
    eta_mu=eta_mu,
    kcal_scale=model.kcal_scale,
    n_samples=1000,        # Monte Carlo permutations per draw
    max_group_size=15,     # Skip groups larger than 15
)

# Joint sampling (one permutation per posterior draw)
shapley_joint = shapley_contributions_all_groups(
    S_x=S_x,
    data=dataset,
    intercept_mu=intercept_mu,
    b_groupsize_mu=b_groupsize_mu,
    eta_mu=eta_mu,
    kcal_scale=model.kcal_scale,
    method="joint",
)

# shapley_ds is an xarray Dataset with dimensions (chain, draw, group, forager)
```

### Analyzing Results

```python
# Get mean contributions per group
mean_contrib = shapley_ds['shapley_contribution'].mean(dim=['chain', 'draw'])

# Get credible intervals
ci = shapley_ds['shapley_contribution'].quantile(
    [0.05, 0.95], dim=['chain', 'draw']
)

# Sum contributions per group (should equal mu)
total_per_group = shapley_ds['shapley_contribution'].sum(dim='forager')

# Compare foragers across groups
forager_totals = shapley_ds['shapley_contribution'].sum(dim='group')
```

## Computational Complexity

| Group Size | Method | Subsets/Samples | Notes |
|------------|--------|-----------------|-------|
| ≤ 10 | Exact | 2^10 = 1,024 | Very fast, exact |
| 11-15 | Exact | 2^15 = 32,768 | Fast, exact |
| 16-20 | Monte Carlo | 5,000 samples | Slower, approximate |
| > 20 | Monte Carlo | 5,000+ samples | Consider skipping |

Use `max_group_size` to skip very large groups:

```python
shapley_ds = shapley_contributions_all_groups(
    ...,
    max_group_size=20,  # Skip groups > 20 members
)
```

## Uncertainty Propagation

Shapley values are computed for each MCMC sample, preserving posterior uncertainty:

- Each (chain, draw) gets its own Shapley computation
- Results include full posterior distributions
- Can compute credible intervals and compare uncertainty across foragers

```python
# Posterior distribution for forager 0's contribution to group 5
forager_contrib = shapley_ds['shapley_contribution'].sel(group=5, forager=0)

# Mean ± std
mean = float(forager_contrib.mean())
std = float(forager_contrib.std())
print(f"Contribution: {mean:.1f} ± {std:.1f} kcal")
```

## Example Interpretation

For a group with 3 members having skills [0.8, 0.5, 0.3]:

The Monte Carlo method samples random orderings like:
- Order [A, B, C]: A contributes μ({A}) - 0, B contributes μ({A,B}) - μ({A}), ...
- Order [B, A, C]: B contributes μ({B}) - 0, A contributes μ({A,B}) - μ({B}), ...

Averaging over many orderings gives fair attribution that accounts for:
- Synergies between members
- Order-independence
- Relative skill differences

## API Reference

### `shapley_contributions_all_groups()`

Main function for computing Shapley values across all groups.

**Parameters:**
- `S_x`: Skills array, shape (chain, draw, forager)
- `data`: xarray Dataset with forager data
- `intercept_mu`, `b_groupsize_mu`, `eta_mu`: Model parameters
- `kcal_scale`: Scale factor for converting to original units
- `max_group_size`: Skip groups larger than this (optional)
- `method`: "auto" (exact for n<=15, joint MC otherwise), "exact", or "joint"
- `n_posterior_samples`: Number of posterior samples to use (default: 100)

**Returns:** xarray Dataset with 'shapley_contribution' variable, dims (sample, group, forager)

## References

- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS.
