"""
Tests for preprocessing module.

These tests define the expected behavior of preprocessing functions.
They should fail initially and only pass when preprocessing is correctly implemented.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing import preprocess_data


class TestPreprocessingOutputs:
    """Test that preprocessing produces expected outputs."""
    
    def test_preprocessing_returns_four_dataframes(self):
        """Preprocessing should return exactly four dataframes."""
        # This test will fail until preprocessing returns the correct structure
        raw_data_dir = Path("raw_data")
        
        # Skip if raw data doesn't exist
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        result = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        assert len(result) == 4, "Preprocessing should return 4 dataframes"
        df_foragers, df_time_agg, df_production, df_days_long = result
        
        assert isinstance(df_foragers, pd.DataFrame)
        assert isinstance(df_time_agg, pd.DataFrame)
        assert isinstance(df_production, pd.DataFrame)
        assert isinstance(df_days_long, pd.DataFrame)
    
    def test_foragers_dataframe_has_required_columns(self):
        """Foragers dataframe should have id, sex, and age columns."""
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        df_foragers, _, _, _ = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        required_cols = ['id', 'sex', 'age']
        for col in required_cols:
            assert col in df_foragers.columns, f"Foragers dataframe missing column: {col}"
    
    def test_production_dataframe_has_group_id_and_kcal(self):
        """Production dataframe should have group_id and kcal columns."""
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        _, _, df_production, _ = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        assert 'group_id' in df_production.columns
        assert 'kcal' in df_production.columns
        assert df_production['kcal'].dtype in [np.float64, np.float32, float]
    
    def test_zero_inferred_records_have_correct_type(self):
        """Zero-inferred records should have type='zero_inferred'."""
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        _, _, df_production, _ = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        if 'zero_inferred' in df_production['type'].values:
            zero_records = df_production[df_production['type'] == 'zero_inferred']
            assert (zero_records['kcal'] == 0).all(), "Zero-inferred records should have kcal=0"
    
    def test_group_id_format_is_consistent(self):
        """Group IDs should follow consistent format: date_forager_ids."""
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        _, _, df_production, _ = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        # Group IDs should have format: YYYY-MM-DD_id1_id2_...
        for group_id in df_production['group_id']:
            parts = group_id.split('_')
            assert len(parts) >= 2, f"Group ID '{group_id}' doesn't match expected format"
            # First part should be a date
            assert len(parts[0]) == 10, f"Date part of group_id '{group_id}' incorrect format"


class TestPreprocessingForNewStructure:
    """Tests for preprocessing that supports new model structure (forager-daily + group-trip)."""
    
    def test_preprocessing_can_create_forager_daily_dataset(self):
        """Preprocessing should be able to create forager-daily level dataset.
        
        This test will fail until preprocessing is updated to support the new structure.
        """
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        # This function signature may need to change
        # For now, we test that current preprocessing can be extended
        df_foragers, df_time_agg, df_production, df_days_long = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        # We should be able to create forager-daily level data
        # This requires merging df_time_agg with df_days_long
        assert 'date' in df_time_agg.columns, "Time allocation should have date column"
        assert 'id' in df_time_agg.columns, "Time allocation should have forager id column"
        
        # Forager-daily level should have one row per (forager, date) combination
        # where the forager was in camp
        if 'forager_id' in df_days_long.columns and 'date' in df_days_long.columns:
            # This is a placeholder test - actual implementation may differ
            assert len(df_days_long) > 0, "Days in camp dataframe should not be empty"
    
    def test_preprocessing_preserves_group_trip_structure(self):
        """Preprocessing should preserve group-trip level structure for returns data.
        
        Group-trip level data should have one row per group_id, with aggregated returns.
        """
        raw_data_dir = Path("raw_data")
        if not raw_data_dir.exists():
            pytest.skip("Raw data directory not found")
        
        _, _, df_production, _ = preprocess_data(
            returns_file=raw_data_dir / 'returns.csv',
            recall_file=raw_data_dir / 'recall.csv',
            kcal_file=raw_data_dir / 'kcal.csv',
            group_file=raw_data_dir / 'groups.csv',
            camp_members_file=raw_data_dir / 'camp_members.csv',
            days_in_camp_file=raw_data_dir / 'daysincamp.csv',
        )
        
        # Group ID should be unique (one row per group/trip)
        assert df_production['group_id'].is_unique, "Each group_id should appear only once"
        
        # Should have forager_ids column indicating group composition
        assert 'forager_ids' in df_production.columns, "Production dataframe should have forager_ids"
        
        # forager_ids should be a set or list
        if len(df_production) > 0:
            first_forager_ids = df_production['forager_ids'].iloc[0]
            assert hasattr(first_forager_ids, '__iter__'), "forager_ids should be iterable"


class TestKcalAggregation:
    """Regression tests for the per-group kcal aggregation step.

    The raw returns/recall tables are long: one row per (forager × food
    package). For cooperative packages, the package weight is repeated
    across every contributor. Earlier preprocessing took ``groupby
    .mean()`` over those rows, which silently divided multi-item / multi-
    forager group totals by the row count, under-counting the true
    kcal by ~45% across the dataset. These tests pin the corrected
    behavior in place.
    """

    def _write_fixture(self, tmp_path):
        """Build a minimal raw-data shaped tree for preprocess_data()."""
        rd = tmp_path / "raw_data"
        rd.mkdir()

        # Two foragers (1 adult male, 1 adult female), one cooperative day.
        camp = pd.DataFrame({
            "ID": [1001, 1002],
            "sex": ["M", "F"],
            "day_in": ["01/08/2018", "01/08/2018"],
            "day_out": ["02/08/2018", "02/08/2018"],
            "BirthYear": [1990, 1992],
            "Strength_Left1": [20, 18], "Strength_Right1": [22, 19],
            "Strength_Left2": [21, 18], "Strength_Right2": [22, 19],
            "Strength_Left3": [21, 18], "Strength_Right3": [22, 19],
        })
        camp.to_csv(rd / "camp_members.csv", index=False)

        # Days in camp: both foragers in camp for both days.
        days = pd.DataFrame({
            "id/date": [1001, 1002],
            "01/08/2018": [1, 1],
            "02/08/2018": [1, 1],
            "Total": [2, 2],
        })
        days.to_csv(rd / "daysincamp.csv", index=False)

        # Out-of-camp activity log: both went out on 01/08 together.
        groups = pd.DataFrame({
            "Date": ["01/08/2018", "01/08/2018"],
            "ID": [1001, 1002],
            "Group": [0, 0],
            "Date_Depart": ["01/08/2018", "01/08/2018"],
            "Depart": [8.0, 8.0],
            "Date_Arrive": ["01/08/2018", "01/08/2018"],
            "Arrive": [16.0, 16.0],
            "Activity": ["Foraging", "Foraging"],
            "Activity.Type": ["Foraging", "Foraging"],
            "Time.Out": [8.0, 8.0],
            "Total.Minutes": [480, 480],
            "Precision": ["Observed", "Observed"],
        })
        groups.to_csv(rd / "groups.csv", index=False)

        # kcal lookup: two items.
        kcal = pd.DataFrame({
            "Index": [1, 2],
            "Yaka": ["A", "B"],
            "English": ["Item-A", "Item-B"],
            "Processing_Condition": ["whole", "whole"],
            "kcal.g": [2.0, 4.0],
        })
        kcal.to_csv(rd / "kcal.csv", index=False)

        # Returns: two cooperative packages (pooled_group 1 and 2),
        # weight repeated per contributor. Item 1 = 1000g, item 2 = 500g.
        returns = pd.DataFrame({
            "Date":               ["01.08.18"] * 4,
            "pooled_group":       [1, 1, 2, 2],
            "index":              [1, 1, 2, 2],
            "net_food_weight_gram": [1000, 1000, 500, 500],
            "gift":               [0, 0, 0, 0],
            "ID":                 [1001, 1002, 1001, 1002],
            "article":            ["a", "a", "b", "b"],
            "state":              ["raw"] * 4,
        })
        returns.to_csv(rd / "returns.csv", index=False)

        # Minimal recall: one solo recall row so the recall DataFrame is
        # non-empty (preprocess_data's concat path warns on empty recall).
        recall = pd.DataFrame({
            "date": ["01/08/2018"],
            "gift": [0],
            "id": [1001],
            "pooled_group": [0],
            "article_consumed": ["a"],
            "quantity": [1.0],
            "x1_unit_weight_grams": [10.0],
            "index": [1.0],
            "source": ["test"],
            "total_weight_grams": [10.0],
        })
        recall.to_csv(rd / "recall.csv", index=False)

        return rd

    def test_cooperative_multi_package_summed_not_averaged(self, tmp_path):
        """Two foragers, two distinct cooperative packages on one day.

        Item 1: 1000g × 2.0 kcal/g = 2000 kcal.
        Item 2:  500g × 4.0 kcal/g = 2000 kcal.
        Correct group-level total: 4000 kcal (sum across packages,
        deduplicating the per-contributor row replication).

        The bug returned 2000 (= 4000 / n_packages), divided by item count.
        """
        rd = self._write_fixture(tmp_path)
        _, _, df_prod, _ = preprocess_data(
            returns_file=rd / "returns.csv",
            recall_file=rd / "recall.csv",
            kcal_file=rd / "kcal.csv",
            group_file=rd / "groups.csv",
            camp_members_file=rd / "camp_members.csv",
            days_in_camp_file=rd / "daysincamp.csv",
            combine_returns_recall=False,
        )
        coop = df_prod[
            (df_prod["group_id"] == "2018-08-01_1001_1002")
            & (df_prod["type"] == "foraging")
        ]
        assert len(coop) == 1, "Cooperative day should produce one foraging row"
        assert coop["kcal"].iloc[0] == pytest.approx(4000.0), (
            f"Expected 4000 kcal (sum of both packages), got {coop['kcal'].iloc[0]}"
        )

    def test_solo_multi_item_summed_not_averaged(self, tmp_path):
        """Single forager, two distinct items on one solo day.

        Both items have pooled_group=NaN. Correct: sum across items.
        """
        rd = self._write_fixture(tmp_path)
        # Overwrite returns with a solo multi-item case.
        returns = pd.DataFrame({
            "Date":               ["01.08.18", "01.08.18"],
            "pooled_group":       [np.nan, np.nan],
            "index":              [1, 2],
            "net_food_weight_gram": [1000, 500],
            "gift":               [0, 0],
            "ID":                 [1001, 1001],
            "article":            ["a", "b"],
            "state":              ["raw", "raw"],
        })
        returns.to_csv(rd / "returns.csv", index=False)

        _, _, df_prod, _ = preprocess_data(
            returns_file=rd / "returns.csv",
            recall_file=rd / "recall.csv",
            kcal_file=rd / "kcal.csv",
            group_file=rd / "groups.csv",
            camp_members_file=rd / "camp_members.csv",
            days_in_camp_file=rd / "daysincamp.csv",
            combine_returns_recall=False,
        )
        solo = df_prod[
            (df_prod["group_id"] == "2018-08-01_1001")
            & (df_prod["type"] == "foraging")
        ]
        assert len(solo) == 1
        # Item 1: 1000 × 2.0 = 2000; Item 2: 500 × 4.0 = 2000; total = 4000
        assert solo["kcal"].iloc[0] == pytest.approx(4000.0), (
            f"Expected 4000 kcal (sum of both solo items), got {solo['kcal'].iloc[0]}"
        )

