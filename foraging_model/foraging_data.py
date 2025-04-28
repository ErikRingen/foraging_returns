import pandas as pd
import xarray as xr

def _mean_scaler(x: float) -> float:
    """
    Scale a float by dividing by the mean
    """
    return x / x.mean()

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
            target_column: str,
            group_id_col: str,
            forager_id_col: str,
            target_scaling: str | None = None,
            age_scaling: str | None = None,
            )-> None:
        self.foragers_df = foragers_df  
        self.time_allocation_df = time_allocation_df
        self.production_df = production_df
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
        ds_production = self.production_df.set_index(self.group_id_col).to_xarray().rename({self.group_id_col: 'group'}).set_coords('date')

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

        combined_ds = xr.merge([ds_foragers, ds_production])

        return combined_ds
        
        