# Scripts

## Workflow

1. **Fit the model** (takes ~10-30 minutes depending on hardware):
   ```bash
   pixi run python scripts/fit_model.py
   ```
   
   Outputs:
   - `results/idata.nc` - Full InferenceData with posterior samples
   - `results/model_summary.csv` - ArviZ summary table

2. **Extract results for manuscript**:
   ```bash
   pixi run python scripts/extract_results.py
   ```
   
   Outputs:
   - `results/manuscript_numbers.json` - Key posterior summaries

## Output Format

The `manuscript_numbers.json` file contains:

```json
{
  "skill_curve": {
    "m": {"mean": ..., "hdi_low": ..., "hdi_high": ...},
    "k": {...},
    "b": {...},
    "peak_skill_age_years": {...},
    "age_50pct_skill_years": {...}
  },
  "effort": {
    "effort_intercept": {...},
    "effort_age": {...},
    "effort_age2": {...}
  },
  "random_effects": {
    "effort_skill_correlation": {"mean": ..., "prob_negative": ...},
    "sigma_effort": {...},
    "sigma_skill": {...}
  },
  "returns": {
    "intercept_mu": {...},
    "b_groupsize_mu": {...},
    "eta_mu": {...},
    "gamma_shape": {...}
  }
}
```

Each parameter includes:
- `mean`: Posterior mean
- `hdi_low`, `hdi_high`: 95% highest density interval bounds

## Notes

- The `idata.nc` file is large (~50-100MB) and excluded from git
- Re-run `fit_model.py` to regenerate if needed
- The `manuscript_numbers.json` is small and can be committed
