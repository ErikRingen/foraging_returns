# Scripts

The end-to-end pipeline is `run_all.sh`. Individual steps are listed below in
the order they are normally run.

## Pipeline driver

| Script | Purpose |
|---|---|
| `run_all.sh` | Chains all steps below: fit canonical + sensitivity, extract numbers, compute Shapley, build summaries, run model comparison, regenerate figures, render the supplement. Use `--skip-fit` to reuse cached `idata.nc`. |

## Fitting

| Script | Purpose | Output |
|---|---|---|
| `fit_model.py` | Fit one model variant by name. Flags `--me-age`, `--no-gp`, `--aggregation {best_top_k, mean}`, plus sampling controls. | `results/<name>/idata.nc`, `model_summary.csv`, `config.json`, `model_graph.png` |

The canonical fit is `--name ln_nogp_meage_long --me-age --no-gp` with 6 chains × 2,000 warm-up + 5,000 sampling; the aggregation-sensitivity variant adds `--aggregation mean` and is `ln_nogp_meage_avg`.

## Posterior summaries

| Script | Purpose | Output |
|---|---|---|
| `extract_results.py` | Headline-number extraction (peak skill age, ψ by gender, effort coefficients, etc.) | `results/manuscript_numbers.json` |
| `compute_shapley.py` | Per-forager Shapley attributions under the variant's value function (best-top-$k$ for both, since the engine in `foraging_model/counterfactuals.py` always uses best-top-$k$). | `results/<variant>/shapley.nc` |
| `compute_shapley_mean_agg.py` | Per-forager Shapley using the **true** mean-aggregation value function for the avg-variant fit. Required for the §9.4 per-package comparison in the supplement. | `results/ln_nogp_meage_avg/shapley_mean_agg.nc` |
| `compute_kcal_summary.py` | Single source of truth for per-forager kcal/foraging-day and kcal/in-camp-day averages, with and without recall. | `results/kcal_summary.csv` |
| `compare_models.py` | PSIS-LOO across observables (kcal, effort, non_zero_prod) for all variants found in `results/`. | `results/model_comparison_*.csv` |
| `marginal_effects.py` | Skill/effort marginal-effect sweeps used by some figures. | written under `results/` |

## Figures

| Script | Purpose | Output |
|---|---|---|
| `make_figure2.py` | Per-forager Shapley boxes by age (main-text Figure 2). | `results/figures/figure2_*.png` |
| `make_figure3.py` | Skill curve + effort polynomial by gender (main-text Figure 3). | `results/figures/figure3_*.png` |
| `make_figure4.py` | Gender-stratified ψ scatter + posterior (main-text Figure 4). | `results/figures/figure4_psi.png` |
| `eda_resource_by_skill.py` | Per-forager kcal share by resource × within-gender skill tercile (supplement Figure S25). | `results/figures/eda_parallel_resource_skill.png` |

## Data preparation

| Script | Purpose | Output |
|---|---|---|
| `build_public_dataset.py` | Build the anonymised public dataset from `raw_data/`. Requires `BAYAKA_SALT` (≥ 16 chars) in env or `--salt`. | `public_data/*.csv` |
| `export_dashboard_data.py` | Tidy data export for the static web dashboard (`dashboard/foraging_explorer.html`). | `dashboard/data.json` |
| `_derive_priors.py` | One-off: derive Koster-based priors from `raw_data/model_fix_17092018.RData` and write `foraging_model/cross_cultural_priors.json`. The output is checked in, so this script does not need to be re-run unless the priors change. | `foraging_model/cross_cultural_priors.json` |

## Notes

- `results/` is gitignored. Running `run_all.sh` from a fresh clone regenerates everything.
- `idata.nc` files are not shipped (they're a few hundred MB each); the canonical fit takes ~30 minutes on an M-series Mac.
- `manuscript_numbers.json` and `kcal_summary.csv` are the canonical numerical references for the manuscript and supplement; do not regenerate by hand.
