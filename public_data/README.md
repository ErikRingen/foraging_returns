# Public dataset (anonymized)

This is the publicly shareable subset of the BaYaka subsistence dataset
underlying the manuscript. It has been processed by
`scripts/build_public_dataset.py` to:

1. Replace original participant IDs with anonymized labels (`P001`, `P002`,
   ...). The mapping is generated with a private salt (held by the authors
   and never committed); it cannot be reproduced from the published source
   code and is unrelated to IDs in other BaYaka papers.
2. Replace `BirthYear` with `age` (years, = `2018 - BirthYear`), the form
   the model consumes. Age and birth year are mathematically equivalent
   given a known study year; this is a presentational choice, not a
   privacy step.
3. Remove resource-level attribution and raw weights. Specifically, the
   `index`, `article`, `article_consumed`, `source`, free-text `state`,
   and the per-package weight columns (`net_food_weight_gram` in
   `returns.csv`; `quantity`, `x1_unit_weight_grams`, `total_weight_grams`
   in `recall.csv`) are dropped. Only the pre-computed `kcal` totals
   remain as the dependent variable.
4. Exclude rows recorded as outright gifts from non-camp members
   (`gift` = 1), matching the canonical analysis. Rows flagged as partly
   gifted (`gift` = 0.5) are retained as recorded.

For the full dataset (with real participant IDs and resource attribution),
contact the corresponding authors. Use is subject to ethical approval.
