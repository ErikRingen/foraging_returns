import pandas as pd
import xarray as xr
import numpy as np

def _mean_scaler(x: float) -> float:
    """
    Scale a float by dividing by the mean of non-zero values
    """
    return x / x.where(x != 0).mean()

def _max_scaler(x: float) -> float:
    """
    Scale a float by dividing by the max
    """
    return x / x.max()

class ForagingData:
    def __init__(
            self,
            foragers_df: pd.DataFrame,
            time_allocation_df: pd.DataFrame,
            production_df: pd.DataFrame,
            days_in_camp_df: pd.DataFrame,
            target_column: str,
            group_id_col: str,
            forager_id_col: str,
            target_scaling: str | None = None,
            age_scaling: str | None = None,
            )-> None:
        self.foragers_df = foragers_df  
        self.time_allocation_df = time_allocation_df
        self.production_df = production_df
        self.days_in_camp_df = days_in_camp_df
        self.target_column = target_column
        self.group_id_col = group_id_col
        self.forager_id_col = forager_id_col
        self.target_scaling = target_scaling
        self.age_scaling = age_scaling

        if self.target_scaling not in ['mean', 'max', None]:
            raise ValueError(f"Target scaling must be 'mean', 'max', or None, got {self.target_scaling}")

        if self.age_scaling not in ['mean', 'max', None]:
            raise ValueError(f"Age scaling must be 'mean', 'max', or None, got {self.age_scaling}")

    def to_dataset(self) -> xr.Dataset:
        """
        Combine the dataframes into an xarray dataset
        """
        ds_production = self.production_df.set_index(self.group_id_col).to_xarray().rename({self.group_id_col: 'group', 'date': 'group_date'}).set_coords('group')

        # scale the target column
        if self.target_scaling == 'mean':
            ds_production[f'{self.target_column}_scaled'] = _mean_scaler(ds_production[self.target_column])

            ds_production[f'{self.target_column}_scale'] = ds_production[self.target_column].mean()

        elif self.target_scaling == 'max':
            ds_production[f'{self.target_column}_scaled'] = _max_scaler(ds_production[self.target_column])

            ds_production[f'{self.target_column}_scale'] = ds_production[self.target_column].max()

        ds_foragers = self.foragers_df.set_index(self.forager_id_col).to_xarray().rename({self.forager_id_col: 'forager'})
        
        if self.age_scaling == 'mean':
            ds_foragers['age_scaled'] = _mean_scaler(ds_foragers['age'])

            ds_foragers['age_scale'] = ds_foragers['age'].mean()

        elif self.age_scaling == 'max':
            ds_foragers['age_scaled'] = _max_scaler(ds_foragers['age'])

            ds_foragers['age_scale'] = ds_foragers['age'].max()

        # Convert forager_ids to indices before merging
        if 'forager_ids' in self.production_df.columns:
            # Create mapping from forager ID to index
            forager_id_to_idx = {str(id): idx for idx, id in enumerate(self.foragers_df[self.forager_id_col])}
            
            # Convert sets to lists of consistent length
            max_foragers = max(len(ids) for ids in self.production_df['forager_ids'])
            forager_ids_array = np.full((len(self.production_df), max_foragers), -1, dtype=np.int64)
            
            for i, ids in enumerate(self.production_df['forager_ids']):
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
                coords={'group': self.production_df[self.group_id_col]}
            )
            ds_production['forager_ids'] = forager_ids_da

        # Forager days in camp
        ds_forager_days_in_camp = (
            self.days_in_camp_df
            .set_index(['forager_id', 'date'])
            .astype(bool)
            .to_xarray()
            .rename({'forager_id': 'forager'})
            .set_coords(['date', 'forager'])
        )

        combined_ds = xr.merge(
            [
                ds_foragers,
                ds_production,
                ds_forager_days_in_camp
            ]
        )

        return combined_ds
        
        