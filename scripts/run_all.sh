#!/usr/bin/env bash
# Reproduce all analyses for the manuscript end-to-end.
#
# Prerequisite: real participant data in raw_data/. (See public_data/ for
# the anonymized subset that's sufficient for aggregate analyses.)
#
# Wall-clock estimate on an M-series Mac: ~1 hour total
#   - Canonical fit:   ~30 min
#   - Sensitivity fit: ~30 min
#   - Everything else: a few minutes
#
# Usage:
#   bash scripts/run_all.sh                # fit + extract + figures + supplement
#   bash scripts/run_all.sh --skip-fit     # skip refits if cached idata.nc exists

set -euo pipefail

SKIP_FIT=0
for arg in "$@"; do
  case "$arg" in
    --skip-fit) SKIP_FIT=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

echo "=========================================="
echo "1/9  Canonical model: ln_nogp_meage_long"
echo "=========================================="
if [ "$SKIP_FIT" -eq 0 ] || [ ! -f results/ln_nogp_meage_long/idata.nc ]; then
  pixi run python scripts/fit_model.py \
      --name ln_nogp_meage_long \
      --me-age --no-gp \
      --tune 2000 --draws 5000 --chains 6 --target-accept 0.95
else
  echo "Cached fit found — skipping."
fi
# Keep the rendered model graph in lock-step with the canonical fit so
# the supplement's static reference (../results/figures/model_graph_canonical.png)
# always reflects what was fit.
mkdir -p results/figures
cp results/ln_nogp_meage_long/model_graph.png results/figures/model_graph_canonical.png

echo
echo "=========================================="
echo "2/9  Sensitivity (averaging): ln_nogp_meage_avg"
echo "=========================================="
if [ "$SKIP_FIT" -eq 0 ] || [ ! -f results/ln_nogp_meage_avg/idata.nc ]; then
  pixi run python scripts/fit_model.py \
      --name ln_nogp_meage_avg \
      --aggregation mean --me-age --no-gp \
      --tune 2000 --draws 5000 --chains 6 --target-accept 0.95
else
  echo "Cached fit found — skipping."
fi

echo
echo "=========================================="
echo "3/9  Manuscript numbers (manuscript_numbers.json)"
echo "=========================================="
pixi run python scripts/extract_results.py

echo
echo "=========================================="
echo "4/9  Shapley contributions (best-top-k for both fits)"
echo "=========================================="
pixi run python scripts/compute_shapley.py --variant ln_nogp_meage_long
pixi run python scripts/compute_shapley.py --variant ln_nogp_meage_avg

echo
echo "=========================================="
echo "5/9  Mean-aggregation Shapley (true value function)"
echo "=========================================="
# This computes Shapley using the true mean-aggregation value function for
# the avg-variant fit (counterfactuals.py uses best-top-k regardless of
# which fit it's working from). Output:
#   results/ln_nogp_meage_avg/shapley_mean_agg.nc
pixi run python scripts/compute_shapley_mean_agg.py

echo
echo "=========================================="
echo "6/9  Per-forager kcal summary (kcal_summary.csv)"
echo "=========================================="
pixi run python scripts/compute_kcal_summary.py

echo
echo "=========================================="
echo "7/9  Model comparison (PSIS-LOO)"
echo "=========================================="
pixi run python scripts/compare_models.py

echo
echo "=========================================="
echo "8/9  Main-paper figures 1-4 + supplement EDA"
echo "=========================================="
cp img/dag.png results/figures/figure1_dag.png
pixi run python scripts/make_figure2.py
pixi run python scripts/make_figure3.py
pixi run python scripts/make_figure4.py
# Resource-by-skill EDA figure embedded in supplement §8 (fig-resource-by-skill)
pixi run python scripts/eda_resource_by_skill.py

echo
echo "=========================================="
echo "9/9  Render supplement (HTML + DOCX)"
echo "=========================================="
pixi run quarto render docs/supplement.qmd

echo
echo "=========================================="
echo "Done. Key artifacts:"
echo "  results/manuscript_numbers.json"
echo "  results/kcal_summary.csv"
echo "  results/figures/figure{1..4}_*.png"
echo "  results/figures/eda_parallel_resource_skill.png"
echo "  docs/supplement.html"
echo "  docs/supplement.docx"
echo "=========================================="
