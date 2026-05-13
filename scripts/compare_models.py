#!/usr/bin/env python3
"""
Compare fitted model variants using PSIS-LOO.

Computes LOO for each observed component separately — `kcal` (LogNormal
returns), `effort` (Bernoulli), and `non_zero_prod` (Bernoulli success) —
because the three components have separate likelihoods. Variants are
grouped by observation count within each component (variants fit on
different data subsets cannot be LOO-compared directly), and we write
one comparison table per (component, group).

Usage:
    pixi run python scripts/compare_models.py

Outputs (per observable):
    results/model_comparison_<observable>.csv             - main comparison
    results/model_comparison_<observable>_n<n_obs>.csv    - other groups
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import arviz as az
from pyprojroot import here

OBSERVABLES = ("kcal", "effort", "non_zero_prod")


def _compute_loo_for_observable(idatas: dict, var_name: str) -> dict:
    loo_dict = {}
    for name, idata in idatas.items():
        if var_name not in idata.log_likelihood:
            print(f"  {name}: skipping {var_name} (not in log_likelihood)")
            continue
        try:
            loo_dict[name] = az.loo(idata, pointwise=True, var_name=var_name)
        except Exception as e:
            print(f"  {name}: LOO failed for {var_name} - {e}")
    return loo_dict


def _write_comparisons(loo_dict: dict, results_dir: Path, label: str) -> list[Path]:
    """Group by n_obs and write per-group comparison tables."""
    if not loo_dict:
        return []

    groups: dict[int, dict] = {}
    for name, loo in loo_dict.items():
        n = int(loo.n_data_points)
        groups.setdefault(n, {})[name] = loo

    out_paths = []
    for is_main, (n_obs, group) in zip(
        [True] + [False] * (len(groups) - 1),
        sorted(groups.items(), key=lambda kv: -len(kv[1])),
    ):
        out_path = (
            results_dir / f"model_comparison_{label}.csv" if is_main
            else results_dir / f"model_comparison_{label}_n{n_obs}.csv"
        )
        if len(group) >= 2:
            comparison = az.compare(group)
            print(f"\n--- {label} | n_obs={n_obs} ({len(group)} variants) ---")
            print(comparison)
            comparison.to_csv(out_path)
        else:
            name, loo = next(iter(group.items()))
            row = pd.DataFrame([{
                "model": name,
                "elpd_loo": loo.elpd_loo,
                "se": loo.se,
                "p_loo": loo.p_loo,
                "n_data_points": loo.n_data_points,
            }])
            print(f"\n--- {label} | n_obs={n_obs} ({name}, single variant) ---")
            print(row.to_string(index=False))
            row.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
        out_paths.append(out_path)
    return out_paths


def main():
    results_dir = here("results")
    if not results_dir.exists():
        print("No results/ directory found. Run fit_model.py first.")
        sys.exit(1)

    idatas = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        idata_path = d / "idata.nc"
        if idata_path.exists():
            idatas[d.name] = az.from_netcdf(idata_path)
        else:
            print(f"Skipping {d.name}: no idata.nc found")

    if not idatas:
        print("No fitted variants found. Run fit_model.py first.")
        sys.exit(1)

    out_paths: list[Path] = []
    for var_name in OBSERVABLES:
        print(f"\n=== PSIS-LOO for {var_name!r} ===")
        loo_dict = _compute_loo_for_observable(idatas, var_name)
        out_paths.extend(_write_comparisons(loo_dict, Path(results_dir), var_name))
    return out_paths


if __name__ == "__main__":
    main()
