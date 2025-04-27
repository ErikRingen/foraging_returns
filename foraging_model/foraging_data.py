import pandas as pd
import xarray as xr

class ForagingData:
    def __init__(
            self,
            foragers_df: pd.DataFrame,
            groups_df: pd.DataFrame,
            returns_df: pd.DataFrame,
            target_columns: str | list[str],
            group_id_col: str,
            forager_id_col: str,
            target_scaling: str | None = None,
            )-> None:
        self.foragers_df = foragers_df  
        self.groups_df = groups_df
        self.returns_df = returns_df
        self.target_columns = target_columns
        self.group_id_col = group_id_col
        self.forager_id_col = forager_id_col
        self.target_scaling = target_scaling

    def to_dataset(self) -> xr.Dataset:
        """
        Combine the dataframes into an xarray dataset
        """
        ds_groups = self.groups_df.set_index(self.group_id_col).to_xarray().rename({self.group_id_col: 'group', 'id': 'foragers'}).set_coords('date')

        ds_returns = self.returns_df.set_index(self.group_id_col).to_xarray().rename({self.group_id_col: 'group'}).set_coords('date')

        ds_foragers = self.foragers_df.set_index(self.forager_id_col).to_xarray().rename({self.forager_id_col: 'forager'})

        combined_ds = xr.merge([ds_foragers, ds_groups, ds_returns])

        return combined_ds
        
        