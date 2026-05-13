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

        gender_cats = ["male", "female"]
        gender_arr = pd.Categorical(self.foragers_df["sex"], categories=gender_cats).codes.astype(np.int32)
        ds_foragers["gender_idx"] = ("forager", gender_arr)

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

        # Build mapping: (forager_id, date) -> list of groups with kcal > 0
        forager_date_to_groups = {}
        if 'forager_ids' in production_successful.columns and 'date' in production_successful.columns:
            for _, row in production_successful.iterrows():
                group_date = pd.to_datetime(row['date'])
                if row[self.target_column] > 0:
                    forager_ids_in_group = row['forager_ids']
                    if hasattr(forager_ids_in_group, '__iter__') and not isinstance(forager_ids_in_group, str):
                        for forager_id in forager_ids_in_group:
                            forager_id_str = str(forager_id)
                            key = (forager_id_str, group_date)
                            if key not in forager_date_to_groups:
                                forager_date_to_groups[key] = []
                            forager_date_to_groups[key].append(row[self.group_id_col])
        
        time_id_col = 'id' if 'id' in self.time_allocation_df.columns else self.forager_id_col
        forager_id_to_idx = {str(id): idx for idx, id in enumerate(self.foragers_df[self.forager_id_col])}
        
        datasets_to_merge = [ds_foragers, ds_production, ds_forager_days_in_camp, ds_time_allocation]
        
        if ('date' in self.days_in_camp_df.columns and 'forager_id' in self.days_in_camp_df.columns and
            'date' in self.time_allocation_df.columns and time_id_col in self.time_allocation_df.columns):
            
            # Merge days_in_camp with time_allocation
            merged_df = pd.merge(
                self.days_in_camp_df[['forager_id', 'date', 'in_camp']],
                self.time_allocation_df[[time_id_col, 'date', 'total.minutes']],
                left_on=['forager_id', 'date'],
                right_on=[time_id_col, 'date'],
                how='left'  # Left join to keep all in_camp records
            )
            merged_df['total.minutes'] = merged_df['total.minutes'].fillna(0)
            
            # =====================================================================
            # EFFORT: P(went foraging | in camp)
            # Eligibility: in_camp == 1
            # Outcome: 1 if total.minutes > 0 OR if forager has returns (in-camp foraging)
            # =====================================================================
            effort_eligible = merged_df[merged_df['in_camp'] == 1].copy()
            
            if len(effort_eligible) > 0:
                effort_forager_indices = []
                effort_values = []
                effort_dates = []
                
                for _, row in effort_eligible.iterrows():
                    forager_id = str(row['forager_id'])
                    date = pd.to_datetime(row['date'])
                    
                    if forager_id in forager_id_to_idx:
                        effort_forager_indices.append(forager_id_to_idx[forager_id])
                        effort_dates.append(date)
                        
                        # Effort = 1 if went out OR foraged in-camp (has returns but no minutes)
                        has_returns = (forager_id, date) in forager_date_to_groups
                        effort = 1 if row['total.minutes'] > 0 or has_returns else 0
                        effort_values.append(effort)
                
                ds_effort = xr.Dataset({
                    'forager_effort': (['effort_obs'], np.array(effort_values, dtype=np.int32)),
                    'effort_forager_idx': (['effort_obs'], np.array(effort_forager_indices, dtype=np.int32)),
                    'effort_date': (['effort_obs'], np.array(effort_dates, dtype='datetime64[ns]')),
                }, coords={
                    'effort_obs': np.arange(len(effort_values))
                })
                datasets_to_merge.append(ds_effort)
            
            # =====================================================================
            # SUCCESS: P(returned with food | went foraging)
            # Eligibility: in_camp == 1 AND (total.minutes > 0 OR has returns)
            # Outcome: 1 if in any group with kcal > 0, else 0
            # =====================================================================
            # First, identify in-camp foragers (those with returns but no minutes)
            success_forager_indices = []
            success_values = []
            success_dates = []
            
            for _, row in merged_df[merged_df['in_camp'] == 1].iterrows():
                forager_id = str(row['forager_id'])
                date = pd.to_datetime(row['date'])
                
                if forager_id in forager_id_to_idx:
                    has_returns = (forager_id, date) in forager_date_to_groups
                    went_foraging = row['total.minutes'] > 0 or has_returns
                    
                    # Only include if they went foraging (out or in-camp)
                    if went_foraging:
                        forager_idx = forager_id_to_idx[forager_id]
                        success = 1 if has_returns else 0
                        
                        success_forager_indices.append(forager_idx)
                        success_values.append(success)
                        success_dates.append(date)
            
            if len(success_values) > 0:
                
                ds_forager_success = xr.Dataset({
                    'forager_success': (['forager_date'], np.array(success_values, dtype=np.int32)),
                    'forager_idx': (['forager_date'], np.array(success_forager_indices, dtype=np.int32)),
                    'success_date': (['forager_date'], np.array(success_dates, dtype='datetime64[ns]')),
                }, coords={
                    'forager_date': np.arange(len(success_values))
                })
                datasets_to_merge.append(ds_forager_success)
        
        # Merge all datasets
        combined_ds = xr.merge(
            datasets_to_merge,
            join='outer'  # Explicitly use outer join for compatibility
        )
        
        # =====================================================================
        # DATE PROCESSING FOR GAUSSIAN PROCESSES
        # Compute numeric date values and index mappings on the existing
        # 'date' dimension (from the forager-daily panel).
        # =====================================================================
        if 'date' in combined_ds.dims:
            sorted_dates = np.sort(combined_ds.date.values)
            min_date = sorted_dates.min()
            date_numeric = (sorted_dates - min_date).astype('timedelta64[D]').astype(float)

            date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(sorted_dates)}

            combined_ds['date_numeric'] = ('date', date_numeric)

            def _map_dates(dates, label):
                """Map a sequence of dates to indices on the master date axis,
                raising explicitly if any date is outside the date dim
                (instead of letting a KeyError leak to the model build)."""
                ts = [pd.Timestamp(d) for d in pd.to_datetime(dates)]
                missing = [d for d in ts if d not in date_to_idx]
                if missing:
                    raise ValueError(
                        f"{label} dates not present in the in-camp date axis: "
                        f"{missing[:5]}{' …' if len(missing) > 5 else ''}"
                    )
                return np.array([date_to_idx[d] for d in ts], dtype=np.int32)

            # Map group / effort / success dates to indices on the unified
            # date axis. Outer merge can in principle keep dates that don't
            # appear in days_in_camp (e.g. a non-resident's group day); we
            # surface that explicitly rather than silently masking.
            if 'date' in production_successful.columns:
                combined_ds['group_date_idx'] = (
                    'group',
                    _map_dates(production_successful['date'], 'group'),
                )
            if 'effort_date' in combined_ds:
                combined_ds['effort_date_idx'] = (
                    'effort_obs',
                    _map_dates(combined_ds.effort_date.values, 'effort'),
                )
            if 'success_date' in combined_ds:
                combined_ds['success_date_idx'] = (
                    'forager_date',
                    _map_dates(combined_ds.success_date.values, 'success'),
                )

        return combined_ds
        
        