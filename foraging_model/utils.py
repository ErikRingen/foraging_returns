"""
Utility functions for foraging model.
"""

import xarray as xr
import numpy as np


def mean_scaler(x: xr.DataArray) -> tuple[xr.DataArray, float]:
    """
    Scale a DataArray by dividing by the mean of non-zero values.
    
    Parameters
    ----------
    x : xr.DataArray
        DataArray to scale
        
    Returns
    -------
    scaled : xr.DataArray
        Scaled data
    scale_factor : float
        Scale factor used (mean of non-zero values)
    """
    non_zero = x.where(x != 0)
    scale_factor = float(non_zero.mean().values)
    scaled = x / scale_factor
    return scaled, scale_factor


def max_scaler(x: xr.DataArray) -> tuple[xr.DataArray, float]:
    """
    Scale a DataArray by dividing by the maximum value.
    
    Parameters
    ----------
    x : xr.DataArray
        DataArray to scale
        
    Returns
    -------
    scaled : xr.DataArray
        Scaled data
    scale_factor : float
        Scale factor used (maximum value)
    """
    scale_factor = float(x.max().values)
    scaled = x / scale_factor
    return scaled, scale_factor
