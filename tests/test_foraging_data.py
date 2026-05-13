"""
Tests for ForagingData class.

These tests define the expected behavior of the ForagingData class.
They should fail initially and only pass when the class is correctly implemented.
"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
from foraging_model.data import ForagingData


class TestForagingDataInitialization:
    """Test ForagingData initialization and basic properties."""
    
    def test_foraging_data_initializes_with_required_dataframes(
        self, sample_foragers_df, sample_time_allocation_df, 
        sample_production_df, sample_days_in_camp_df
    ):
        """ForagingData should initialize with all required dataframes."""
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        assert data.foragers_df is not None
        assert data.time_allocation_df is not None
        assert data.production_df is not None
        assert data.days_in_camp_df is not None
    
    def test_foraging_data_validates_scaling_options(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """ForagingData should no longer accept scaling options (moved to ForagingModel)."""
        # Valid initialization should work without scaling parameters
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        assert data.target_column == 'kcal'
        
        # Should raise TypeError if scaling parameters are provided
        with pytest.raises(TypeError):
            ForagingData(
                foragers_df=sample_foragers_df,
                time_allocation_df=sample_time_allocation_df,
                production_df=sample_production_df,
                days_in_camp_df=sample_days_in_camp_df,
                target_column='kcal',
                group_id_col='group_id',
                forager_id_col='id',
                target_scaling='mean'  # Should not be accepted
            )


class TestForagingDataToDataset:
    """Test ForagingData.to_dataset() method."""
    
    def test_to_dataset_returns_xarray_dataset(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should return an xarray Dataset."""
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        dataset = data.to_dataset()
        assert isinstance(dataset, xr.Dataset)
    
    def test_to_dataset_does_not_create_scaled_columns(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should NOT create scaled columns.
        
        Scaling should be moved to ForagingModel class, not ForagingData.
        """
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        dataset = data.to_dataset()
        
        # Should have original unscaled columns
        assert 'kcal' in dataset.data_vars, "Should have original kcal column"
        assert 'age' in dataset.data_vars, "Should have original age column"
        
        # Should NOT have scaled columns (scaling happens in model)
        assert 'kcal_scaled' not in dataset.data_vars, "Scaling should not be in data class"
        assert 'age_scaled' not in dataset.data_vars, "Scaling should not be in data class"
        assert 'kcal_scale' not in dataset.data_vars and 'kcal_scale' not in dataset.attrs, "Scale factors should not be in dataset"
        assert 'age_scale' not in dataset.data_vars and 'age_scale' not in dataset.attrs, "Scale factors should not be in dataset"
    
    def test_to_dataset_creates_correct_dimensions(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should create datasets with correct dimensions."""
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        dataset = data.to_dataset()
        
        # Should have 'forager' dimension
        assert 'forager' in dataset.dims or 'forager' in dataset.coords, "Should have forager dimension"
        
        # Should have 'group' dimension
        assert 'group' in dataset.dims or 'group' in dataset.coords, "Should have group dimension"
        
        # Note: xarray merge may create alignment warnings but should still work
    
    def test_to_dataset_creates_forager_ids_array(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should create forager_ids array with correct structure."""
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        dataset = data.to_dataset()
        
        # Should have forager_ids if production_df has it
        if 'forager_ids' in sample_production_df.columns:
            assert 'forager_ids' in dataset.data_vars, "Should have forager_ids in dataset"
            
            # Should have dimensions (group, forager_in_group)
            forager_ids = dataset['forager_ids']
            assert 'group' in forager_ids.dims, "forager_ids should have group dimension"
            assert 'forager_in_group' in forager_ids.dims, "forager_ids should have forager_in_group dimension"
            
            # Values should be integers (indices or sentinel values)
            assert forager_ids.dtype in [np.int64, np.int32, int], "forager_ids should be integer type"


class TestForagingDataForNewStructure:
    """Tests for ForagingData class that supports new model structure."""
    
    def test_to_dataset_can_create_forager_daily_level(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should be able to create forager-daily level data.
        
        This test will fail until ForagingData is updated to support the new structure.
        """
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        # This method may need to be added or modified
        # For now, we test that we can create a forager-daily level view
        dataset = data.to_dataset()
        
        # Should be able to identify forager-daily combinations
        # This might require a new method or parameter
        # Placeholder: test that we have the necessary data
        assert 'forager' in dataset.coords or 'forager' in dataset.dims
        # Date information should be available
        assert 'group_date' in dataset.coords or 'date' in dataset.coords
    
    def test_to_dataset_preserves_group_trip_structure(
        self, sample_foragers_df, sample_time_allocation_df,
        sample_production_df, sample_days_in_camp_df
    ):
        """to_dataset() should preserve group-trip level structure."""
        data = ForagingData(
            foragers_df=sample_foragers_df,
            time_allocation_df=sample_time_allocation_df,
            production_df=sample_production_df,
            days_in_camp_df=sample_days_in_camp_df,
            target_column='kcal',
            group_id_col='group_id',
            forager_id_col='id'
        )
        
        dataset = data.to_dataset()
        
        # Group-level data should be preserved
        assert 'group' in dataset.coords or 'group' in dataset.dims
        assert 'kcal' in dataset.data_vars or 'kcal_scaled' in dataset.data_vars

