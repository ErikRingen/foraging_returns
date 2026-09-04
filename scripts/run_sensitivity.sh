#!/usr/bin/env bash
# Fit the revision sensitivity variants and extract per-variant estimands.
#
# Variants (all match the canonical ln_nogp_meage_long specification apart
# from the one perturbation each tests):
#   sens_all_outings   effort = any out-of-camp day (canonical: foraging trips)
#   sens_no_recall     production = in-camp returns only (no field consumption)
#   sens_no_palm       all oil-palm resources excluded
#   sens_no_top_palm   the single largest oil-palm harvest excluded (one
#                      date x contributor set, ~68 kg summed across packages)
#   sens_wide_priors   every prior twice as wide
#   sens_wide_priors_koster  only the Koster et al.-derived priors widened
#
# Reduced sampling settings relative to the canonical fit (4 chains x 2500
# draws instead of 6 x 5000) — sufficient for the forest-plot estimands.
set -euo pipefail
cd "$(dirname "$0")/.."

FIT="pixi run python scripts/fit_model.py --me-age --no-gp \
     --tune 2000 --draws 2500 --chains 4 --target-accept 0.95"

declare -a VARIANTS=(
    "sens_all_outings --all-outings"
    "sens_no_recall --no-recall"
    "sens_no_part_gifts --exclude-part-gifts"
    "sens_no_palm --exclude-palm"
    "sens_no_top_palm --exclude-top-palm-harvest"
    "sens_wide_priors --prior-scale 2.0"
    "sens_wide_priors_koster --prior-scale 2.0 --prior-scale-scope koster"
)

for spec in "${VARIANTS[@]}"; do
    name="${spec%% *}"
    flags="${spec#* }"
    echo "=== Fitting ${name} (${flags}) ==="
    $FIT --name "${name}" ${flags} "$@"
    pixi run python scripts/extract_results.py \
        --variant "${name}" \
        --output "results/${name}/manuscript_numbers.json"
done

echo "=== All sensitivity variants complete ==="
