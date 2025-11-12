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
    
    def test_preprocessing_returns_four_dataframes(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        assert len(result) == 4, "Preprocessing should return 4 dataframes"
        df_foragers, df_time_agg, df_production, df_days_long = result
        
        assert isinstance(df_foragers, pd.DataFrame)
        assert isinstance(df_time_agg, pd.DataFrame)
        assert isinstance(df_production, pd.DataFrame)
        assert isinstance(df_days_long, pd.DataFrame)
    
    def test_foragers_dataframe_has_required_columns(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        required_cols = ['id', 'sex', 'age']
        for col in required_cols:
            assert col in df_foragers.columns, f"Foragers dataframe missing column: {col}"
    
    def test_production_dataframe_has_group_id_and_kcal(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        assert 'group_id' in df_production.columns
        assert 'kcal' in df_production.columns
        assert df_production['kcal'].dtype in [np.float64, np.float32, float]
    
    def test_zero_inferred_records_have_correct_type(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        if 'zero_inferred' in df_production['type'].values:
            zero_records = df_production[df_production['type'] == 'zero_inferred']
            assert (zero_records['kcal'] == 0).all(), "Zero-inferred records should have kcal=0"
    
    def test_group_id_format_is_consistent(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        # Group IDs should have format: YYYY-MM-DD_id1_id2_...
        for group_id in df_production['group_id']:
            parts = group_id.split('_')
            assert len(parts) >= 2, f"Group ID '{group_id}' doesn't match expected format"
            # First part should be a date
            assert len(parts[0]) == 10, f"Date part of group_id '{group_id}' incorrect format"


class TestPreprocessingForNewStructure:
    """Tests for preprocessing that supports new model structure (forager-daily + group-trip)."""
    
    def test_preprocessing_can_create_forager_daily_dataset(self, tmp_path):
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
            output_dir=tmp_path
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
    
    def test_preprocessing_preserves_group_trip_structure(self, tmp_path):
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
            output_dir=tmp_path
        )
        
        # Group ID should be unique (one row per group/trip)
        assert df_production['group_id'].is_unique, "Each group_id should appear only once"
        
        # Should have forager_ids column indicating group composition
        assert 'forager_ids' in df_production.columns, "Production dataframe should have forager_ids"
        
        # forager_ids should be a set or list
        if len(df_production) > 0:
            first_forager_ids = df_production['forager_ids'].iloc[0]
            assert hasattr(first_forager_ids, '__iter__'), "forager_ids should be iterable"

