"""
Tests for trip-count effort functionality.

Tests cover:
1. Preprocessing: trip_count column computation
2. ForagingData: effort_trip_count in the xarray dataset
3. ForagingModel: trip_count effort_type builds correctly
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

from preprocessing import preprocess_data
from foraging_model.data import ForagingData
from foraging_model.model import ForagingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_data_dir():
    d = Path("raw_data")
    if not d.exists():
        pytest.skip("Raw data directory not found")
    return d


@pytest.fixture
def preprocessed(raw_data_dir, tmp_path):
    return preprocess_data(
        returns_file=raw_data_dir / "returns.csv",
        recall_file=raw_data_dir / "recall.csv",
        kcal_file=raw_data_dir / "kcal.csv",
        group_file=raw_data_dir / "groups.csv",
        camp_members_file=raw_data_dir / "camp_members.csv",
        days_in_camp_file=raw_data_dir / "daysincamp.csv",
        combine_returns_recall=True,
        output_dir=tmp_path,
    )


@pytest.fixture
def foraging_dataset(preprocessed):
    df_foragers, df_time_agg, df_production, df_days_long = preprocessed
    fd = ForagingData(
        foragers_df=df_foragers,
        time_allocation_df=df_time_agg,
        production_df=df_production,
        days_in_camp_df=df_days_long,
        target_column="kcal",
        group_id_col="group_id",
        forager_id_col="id",
    )
    return fd.to_dataset()


@pytest.fixture
def foragers_df():
    return pd.DataFrame({
        "id": ["1", "2", "3"],
        "sex": ["male", "female", "male"],
        "age": [25.0, 30.0, 35.0],
    })


@pytest.fixture
def time_allocation_with_trips():
    """Time allocation with trip_count column."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "id": ["1"] * 5,
        "date": dates,
        "total.minutes": [120.0, 0.0, 180.0, 90.0, 200.0],
        "trip_count": [2, 0, 3, 1, 4],
    })


@pytest.fixture
def time_allocation_without_trips():
    """Time allocation without trip_count column (backward compat)."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "id": ["1"] * 5,
        "date": dates,
        "total.minutes": [120.0, 0.0, 180.0, 90.0, 200.0],
    })


@pytest.fixture
def production_df():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "group_id": [f"{d.strftime('%Y-%m-%d')}_1" for d in dates],
        "kcal": [100.0, 200.0, 0.0, 150.0, 250.0],
        "type": ["foraging", "foraging", "zero_inferred", "foraging", "foraging"],
        "date": dates,
        "group_size": [1, 1, 1, 1, 1],
        "forager_ids": [{"1"}, {"1"}, {"1"}, {"1"}, {"1"}],
    })


@pytest.fixture
def days_in_camp_df():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "forager_id": ["1"] * 5,
        "date": dates,
        "in_camp": [1, 1, 1, 1, 1],
    })


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessingTripCount:

    def test_trip_count_column_exists(self, preprocessed):
        """Preprocessing should produce a trip_count column."""
        _, df_time_agg, _, _ = preprocessed
        assert "trip_count" in df_time_agg.columns

    def test_trip_count_is_nonnegative(self, preprocessed):
        """Trip counts should be non-negative where defined."""
        _, df_time_agg, _, _ = preprocessed
        valid = df_time_agg["trip_count"].dropna()
        assert (valid >= 0).all()

    def test_trip_count_zero_when_no_minutes(self, preprocessed):
        """When total.minutes is 0 and forager was in camp, trip_count should be 0."""
        _, df_time_agg, _, _ = preprocessed
        zero_time = df_time_agg[
            (df_time_agg["total.minutes"] == 0)
            & df_time_agg["trip_count"].notna()
        ]
        assert (zero_time["trip_count"] == 0).all()

    def test_trip_count_positive_when_minutes_positive(self, preprocessed):
        """When total.minutes > 0, trip_count should be >= 1."""
        _, df_time_agg, _, _ = preprocessed
        pos_time = df_time_agg[
            (df_time_agg["total.minutes"] > 0)
            & df_time_agg["trip_count"].notna()
        ]
        assert (pos_time["trip_count"] >= 1).all()

    def test_trip_count_max_reasonable(self, preprocessed):
        """Trip counts should be reasonable (no more than ~10/day)."""
        _, df_time_agg, _, _ = preprocessed
        valid = df_time_agg["trip_count"].dropna()
        assert valid.max() <= 20


# ---------------------------------------------------------------------------
# ForagingData tests
# ---------------------------------------------------------------------------

class TestForagingDataTripCount:

    def test_effort_trip_count_in_dataset_with_trips(
        self, foragers_df, time_allocation_with_trips, production_df, days_in_camp_df
    ):
        """Dataset should contain effort_trip_count when trip_count is in time_allocation."""
        fd = ForagingData(
            foragers_df=foragers_df,
            time_allocation_df=time_allocation_with_trips,
            production_df=production_df,
            days_in_camp_df=days_in_camp_df,
            target_column="kcal",
            group_id_col="group_id",
            forager_id_col="id",
        )
        ds = fd.to_dataset()
        assert "effort_trip_count" in ds.data_vars

    def test_effort_trip_count_values_match_binary(
        self, foragers_df, time_allocation_with_trips, production_df, days_in_camp_df
    ):
        """effort_trip_count > 0 should align with forager_effort == 1."""
        fd = ForagingData(
            foragers_df=foragers_df,
            time_allocation_df=time_allocation_with_trips,
            production_df=production_df,
            days_in_camp_df=days_in_camp_df,
            target_column="kcal",
            group_id_col="group_id",
            forager_id_col="id",
        )
        ds = fd.to_dataset()
        binary = ds["forager_effort"].values
        trips = ds["effort_trip_count"].values
        assert np.all((trips > 0) == (binary == 1))

    def test_backward_compat_without_trip_count(
        self, foragers_df, time_allocation_without_trips, production_df, days_in_camp_df
    ):
        """ForagingData should still work when trip_count column is absent."""
        fd = ForagingData(
            foragers_df=foragers_df,
            time_allocation_df=time_allocation_without_trips,
            production_df=production_df,
            days_in_camp_df=days_in_camp_df,
            target_column="kcal",
            group_id_col="group_id",
            forager_id_col="id",
        )
        ds = fd.to_dataset()
        assert "forager_effort" in ds.data_vars
        # Should still have trip count (falls back to binary)
        assert "effort_trip_count" in ds.data_vars

    def test_real_data_effort_trip_count(self, foraging_dataset):
        """Real data should have effort_trip_count in the dataset."""
        assert "effort_trip_count" in foraging_dataset.data_vars
        trips = foraging_dataset["effort_trip_count"].values
        assert trips.max() > 1, "Should have multi-trip days"
        assert (trips >= 0).all()


# ---------------------------------------------------------------------------
# ForagingModel tests
# ---------------------------------------------------------------------------

class TestForagingModelTripCountEffort:

    def test_effort_type_defaults_to_binary(self, foraging_dataset):
        """Default effort_type should be 'binary'."""
        model = ForagingModel(data=foraging_dataset)
        assert model.effort_type == "binary"

    def test_invalid_effort_type_raises(self, foraging_dataset):
        """Invalid effort_type should raise ValueError."""
        with pytest.raises(ValueError, match="effort_type"):
            ForagingModel(data=foraging_dataset, effort_type="invalid")

    def test_trip_count_model_builds(self, foraging_dataset):
        """ForagingModel with effort_type='trip_count' should build successfully."""
        model = ForagingModel(data=foraging_dataset, effort_type="trip_count")
        assert hasattr(model, "model")

    def test_trip_count_model_has_negbin_likelihood(self, foraging_dataset):
        """Trip-count model should use NegativeBinomial for effort."""
        model = ForagingModel(data=foraging_dataset, effort_type="trip_count")
        observed_names = [rv.name for rv in model.model.observed_RVs]
        assert "effort" in observed_names

    def test_trip_count_model_has_overdispersion_param(self, foraging_dataset):
        """Trip-count model should have an overdispersion parameter."""
        model = ForagingModel(data=foraging_dataset, effort_type="trip_count")
        free_names = [rv.name for rv in model.model.free_RVs]
        assert "effort_overdispersion" in free_names

    def test_trip_count_model_has_mu_effort(self, foraging_dataset):
        """Trip-count model should have mu_effort deterministic."""
        model = ForagingModel(data=foraging_dataset, effort_type="trip_count")
        det_names = [d.name for d in model.model.deterministics]
        assert "mu_effort" in det_names

    def test_binary_model_has_p_effort(self, foraging_dataset):
        """Binary model should have p_effort deterministic (not mu_effort)."""
        model = ForagingModel(data=foraging_dataset, effort_type="binary")
        det_names = [d.name for d in model.model.deterministics]
        assert "p_effort" in det_names
        assert "mu_effort" not in det_names

    def test_trip_count_requires_data(self, foraging_dataset):
        """If effort_trip_count is missing from data, trip_count should fail."""
        ds_no_trips = foraging_dataset.drop_vars("effort_trip_count")
        with pytest.raises(ValueError, match="effort_trip_count"):
            ForagingModel(data=ds_no_trips, effort_type="trip_count")
