import pandas as pd
import xarray as xr
import numpy as np


class ForagingData:
    """
    Handles data loading, validation, and structure combination.
    
    This class combines multiple DataFrames into a structured xarray Dataset.
    Scaling operations are handled by ForagingModel, not this class.
    """
    def __init__(
            self,
            foragers_df: pd.DataFrame,
            time_allocation_df: pd.DataFrame,
            production_df: pd.DataFrame,
            days_in_camp_df: pd.DataFrame,
            target_column: str,
            group_id_col: str,
            forager_id_col: str,
            )-> None:
        """
        Initialize ForagingData.
        
        Parameters
        ----------
        foragers_df : pd.DataFrame
            Forager demographics (id, sex, age)
        time_allocation_df : pd.DataFrame
            Time allocation per (forager, date)
        production_df : pd.DataFrame
            Production per group (group_id, kcal, type, date, etc.)
        days_in_camp_df : pd.DataFrame
            Days in camp per (forager_id, date)
        target_column : str
            Name of target column in production_df (e.g., 'kcal')
        group_id_col : str
            Name of group identifier column in production_df
        forager_id_col : str
            Name of forager identifier column in foragers_df
        """
        self.foragers_df = foragers_df  
        self.time_allocation_df = time_allocation_df
        self.production_df = production_df
        self.days_in_camp_df = days_in_camp_df
        self.target_column = target_column
        self.group_id_col = group_id_col
        self.forager_id_col = forager_id_col

    def to_dataset(self) -> xr.Dataset:
        """
        Combine the dataframes into an xarray dataset.
        
        Returns unscaled data. Scaling should be handled by ForagingModel.
        
        Returns
        -------
        xr.Dataset
            Dataset with dimensions:
            - forager, date: For forager-daily level data (Bernoulli component)
            - group: For group-trip level data (Gamma component)
            - forager_in_group: For group composition mapping
            Contains unscaled data only.
        """
        # Filter production to exclude zero-inferred records and any records with zero or negative kcal
        # Only successful trips with positive returns have group-level returns for Gamma distribution
        production_successful = self.production_df[
            (self.production_df.get('type', pd.Series(['success'])) != 'zero_inferred') &
            (self.production_df[self.target_column] > 0)  # Ensure kcal is strictly positive
        ].copy()
        
        ds_production = production_successful.set_index(self.group_id_col).to_xarray().rename({self.group_id_col: 'group', 'date': 'group_date'}).set_coords('group')

        ds_foragers = self.foragers_df.set_index(self.forager_id_col).to_xarray().rename({self.forager_id_col: 'forager'})

        # Convert forager_ids to indices before merging
        # Use filtered production_successful for group composition
        if 'forager_ids' in production_successful.columns:
            # Create mapping from forager ID to index
            forager_id_to_idx = {str(id): idx for idx, id in enumerate(self.foragers_df[self.forager_id_col])}
            
            # Convert sets to lists of consistent length
            max_foragers = max(len(ids) for ids in production_successful['forager_ids']) if len(production_successful) > 0 else 0
            if max_foragers > 0:
                forager_ids_array = np.full((len(production_successful), max_foragers), -1, dtype=np.int64)
                
                for i, ids in enumerate(production_successful['forager_ids']):
                    ids_list = list(ids) if hasattr(ids, '__iter__') else [ids]
                    for j, id in enumerate(ids_list):
                        # -99 for foragers not in the database
                        # This distinguishes them from invalid entries (-1)
                        if str(id) not in forager_id_to_idx:
                            forager_ids_array[i, j] = -99
                        else:
                            forager_ids_array[i, j] = forager_id_to_idx[str(id)]
                    
                # Store as a DataArray with proper dimensions
                forager_ids_da = xr.DataArray(
                    forager_ids_array,
                    dims=['group', 'forager_in_group'],
                    coords={'group': production_successful[self.group_id_col]}
                )
                ds_production['forager_ids'] = forager_ids_da

        # Forager days in camp (forager-daily level)
        ds_forager_days_in_camp = (
            self.days_in_camp_df
            .set_index(['forager_id', 'date'])
            .astype(bool)
            .to_xarray()
            .rename({'forager_id': 'forager'})
            .set_coords(['date', 'forager'])
        )

        # Time allocation (forager-daily level)
        # Map forager ID column name
        time_id_col = 'id' if 'id' in self.time_allocation_df.columns else self.forager_id_col
        ds_time_allocation = (
            self.time_allocation_df
            .set_index([time_id_col, 'date'])
            .to_xarray()
            .rename({time_id_col: 'forager'})
            .set_coords(['date', 'forager'])
        )

        # Create forager-daily success indicator
        # Rules:
        #   1. Only create observations where in_camp == 1 AND total.minutes > 0
        #   2. For those observations: success = 1 if forager was in ANY group on that date with kcal > 0, else 0
        #   3. Only include eligible (forager, date) pairs - no NaN entries
        
        # First, build mapping: (forager_id, date) -> list of groups with kcal > 0
        forager_date_to_groups = {}
        if 'forager_ids' in production_successful.columns and 'date' in production_successful.columns:
            for _, row in production_successful.iterrows():
                group_date = pd.to_datetime(row['date'])
                if row[self.target_column] > 0:  # Only groups with kcal > 0
                    forager_ids_in_group = row['forager_ids']
                    if hasattr(forager_ids_in_group, '__iter__') and not isinstance(forager_ids_in_group, str):
                        for forager_id in forager_ids_in_group:
                            forager_id_str = str(forager_id)
                            key = (forager_id_str, group_date)
                            if key not in forager_date_to_groups:
                                forager_date_to_groups[key] = []
                            forager_date_to_groups[key].append(row[self.group_id_col])
        
        # Merge days_in_camp with time_allocation to get (forager, date) pairs where 
        # in_camp == 1 AND total.minutes > 0
        time_id_col = 'id' if 'id' in self.time_allocation_df.columns else self.forager_id_col
        
        # Create merged dataframe for eligible observations
        if ('date' in self.days_in_camp_df.columns and 'forager_id' in self.days_in_camp_df.columns and
            'date' in self.time_allocation_df.columns and time_id_col in self.time_allocation_df.columns):
            
            # Merge days_in_camp with time_allocation
            merged_df = pd.merge(
                self.days_in_camp_df[['forager_id', 'date', 'in_camp']],
                self.time_allocation_df[[time_id_col, 'date', 'total.minutes']],
                left_on=['forager_id', 'date'],
                right_on=[time_id_col, 'date'],
                how='inner'
            )
            
            # Filter to only (forager, date) pairs where in_camp == 1 AND total.minutes > 0
            eligible = merged_df[
                (merged_df['in_camp'] == 1) & 
                (merged_df['total.minutes'] > 0)
            ].copy()
            
            if len(eligible) > 0:
                # Create mapping forager ID -> index
                forager_id_to_idx = {str(id): idx for idx, id in enumerate(self.foragers_df[self.forager_id_col])}
                
                # Create long-format arrays: only eligible observations
                n_observations = len(eligible)
                forager_indices = []
                dates = []
                success_values = []
                
                for _, row in eligible.iterrows():
                    forager_id = str(row['forager_id'])
                    date = pd.to_datetime(row['date'])
                    
                    if forager_id in forager_id_to_idx:
                        forager_idx = forager_id_to_idx[forager_id]
                        # Check if forager was in any group on this date with kcal > 0
                        key = (forager_id, date)
                        success = 1 if key in forager_date_to_groups else 0
                        
                        forager_indices.append(forager_idx)
                        dates.append(date)
                        success_values.append(success)
                
                # Create long-format dataset
                # Use integer indices for forager_date dimension
                # Rename date to forager_date_date to avoid merge conflicts with other datasets' date coordinates
                ds_forager_success = xr.Dataset({
                    'forager_success': (['forager_date'], np.array(success_values, dtype=np.int32)),
                    'forager_idx': (['forager_date'], np.array(forager_indices, dtype=np.int32)),
                    'forager_date_date': (['forager_date'], np.array(dates))  # Renamed to avoid conflict
                }, coords={
                    'forager_date': np.arange(len(success_values))  # Simple integer index
                })
                
                # Add to merge list
                datasets_to_merge = [ds_foragers, ds_production, ds_forager_days_in_camp, ds_time_allocation, ds_forager_success]
            else:
                # No eligible observations
                datasets_to_merge = [ds_foragers, ds_production, ds_forager_days_in_camp, ds_time_allocation]
        else:
            datasets_to_merge = [ds_foragers, ds_production, ds_forager_days_in_camp, ds_time_allocation]
        
        # Merge all datasets
        combined_ds = xr.merge(
            datasets_to_merge,
            join='outer'  # Explicitly use outer join for compatibility
        )

        return combined_ds
        
        