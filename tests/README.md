# Tests

Pytest suite covering the data-pipeline + model construction + Shapley engine.

## Running

```bash
pixi run pytest tests/                   # full suite (~25 s with one model fit)
pixi run pytest tests/ -k "not Fitting"  # skip the actual fit (~3 s)
pixi run pytest tests/test_shapley_values.py -v
```

## What's covered

| File | Scope |
|---|---|
| `test_preprocessing.py` | Raw CSV → analysis-ready DataFrames (`preprocess_data`). |
| `test_foraging_data.py` | `ForagingData` → `xr.Dataset` shape and coords. |
| `test_foraging_model.py` | `ForagingModel` build, expected RVs, end-to-end short fit. |
| `test_shapley_values.py` | Best-top-k aggregation (`_compute_mu`), exact Shapley axioms (efficiency / symmetry / monotonicity), joint Monte Carlo engine. |

## Status

- 41 passing, 1 expected failure.
- The `xfail` in `test_shapley_values.py::TestShapleyJoint::test_positive_and_ordered` documents that strict per-forager Shapley monotonicity does **not** hold under the joint best-top-k engine. This is by design: when the optimal subset size shifts as foragers are added/removed, marginal contributions can be non-monotone in the underlying skill ordering. The test is retained as a regression check for the positivity assertion.

## When tests break

If a fixture or model-building test breaks after editing `foraging_model/`, check first whether `ForagingData.to_dataset()` produces variables that the test fixture's `sample_dataset` doesn't include. The fixture in `conftest.py` mirrors the schema returned by `to_dataset()` — it must be kept in sync with any new dimensions or data variables the model consumes.
