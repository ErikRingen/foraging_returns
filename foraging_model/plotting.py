"""Plotting utilities for foraging model results."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az
import scipy.special as sp
from typing import Dict, List, Optional, Union


def _shapley_to_dict(shapley_results: Union[xr.Dataset, Dict[int, xr.DataArray]]) -> Dict[int, xr.DataArray]:
    """
    Convert Shapley results to dictionary format for backward compatibility.
    
    Parameters:
    -----------
    shapley_results : Union[xr.Dataset, Dict[int, xr.DataArray]]
        Either a Dataset with 'shapley_contribution' variable, or a dict
        
    Returns:
    --------
    Dict[int, xr.DataArray]
        Dictionary mapping group indices to Shapley DataArrays
    """
    if isinstance(shapley_results, xr.Dataset):
        # Convert Dataset to dict format
        results = {}
        shapley_var = shapley_results['shapley_contribution']
        
        # If sample dimension exists, take mean across samples first
        if 'sample' in shapley_var.dims:
            shapley_var = shapley_var.mean(dim='sample')
        
        for group_idx in shapley_var.group.values:
            # Get group index (handle both integer and string group IDs)
            if isinstance(group_idx, (int, np.integer)):
                group_idx_int = int(group_idx)
            else:
                # Find index in coordinates
                group_idx_int = int(np.where(shapley_var.coords['group'].values == group_idx)[0][0])
            
            # Extract non-NaN values for this group
            group_data = shapley_var.sel(group=group_idx)
            valid_mask = ~np.isnan(group_data.values)
            
            if np.any(valid_mask):
                # Get forager indices with valid values
                forager_indices = np.where(valid_mask)[0]
                valid_data = group_data.isel(forager=forager_indices)
                results[group_idx_int] = valid_data.rename('shapley_contribution')
        
        return results
    else:
        # Already a dict
        return shapley_results


def plot_curve(
    idata: az.InferenceData,
    age_scale: float = 1.0,
    prior: bool = False,
    ndraws: int = 100,
    curve_type: str = "S",
) -> None:
    """Plot implied skill curves.
    
    Parameters
    ----------
    idata : az.InferenceData
        InferenceData object from fitted model
    age_scale : float, optional
        Scaling factor for age (to convert back to original scale), by default 1.0
    prior : bool, optional
        Whether to use prior or posterior samples, by default False
    ndraws : int, optional
        Number of draws to use, by default 100
    curve_type : str, optional
        Type of curve to plot: "S", "M", "K", "harvest", or "success", by default "S"
    """
    if curve_type not in ["S", "M", "K", "harvest", "success"]:
        raise ValueError("Invalid curve type, must be one of S, M, K, harvest, or success")

    if prior:
        parameters = az.extract(idata, group="prior")
    else:
        parameters = az.extract(idata, group="posterior")

    # Create age grid (scaled)
    age_grid = np.linspace(0, 1, 100)
    
    # Create dataset with parameters (only include those that exist)
    S_grid = xr.Dataset(
        {
            "m": parameters["m"],
            "k": parameters["k"],
            "b": parameters["b"],
        }
    )
    
    # Add optional parameters if they exist
    optional_params = [
        "intercept_success", "intercept_mu", "b_groupsize_mu", 
        "eta_mu", "eta_success", "alpha_success"
    ]
    for param in optional_params:
        if param in parameters:
            S_grid[param] = parameters[param]
    
    # Add age dimension
    S_grid = S_grid.expand_dims({"age_scaled": age_grid})
    S_grid["age"] = S_grid.age_scaled * age_scale
    
    # Calculate skill curve
    S_grid["K_x"] = 1 - np.exp(-S_grid.k * S_grid.age_scaled)
    S_grid["M_x"] = np.exp(-S_grid.m * S_grid.age_scaled)
    S_grid["S_x"] = S_grid.M_x * S_grid.K_x**S_grid.b
    
    # Calculate mu for solo forager using new best-top-k formula
    # For solo forager (group size = 1), k=1:
    # g(1) = exp(intercept_μ + b_groupsize_mu * log(1)) = exp(intercept_μ)
    # top-1 average = S_x
    # μ = g(1) * S_x^η_mu = exp(intercept_μ) * S_x^η_mu
    if "intercept_mu" in S_grid.data_vars and "eta_mu" in S_grid.data_vars:
        # For solo forager: g(1) = exp(intercept_μ + b_groupsize_mu * log(1)) = exp(intercept_μ)
        S_grid["mu"] = np.exp(S_grid.intercept_mu) * S_grid.S_x**S_grid.eta_mu
    else:
        # Fallback if parameters not available
        S_grid["mu"] = xr.zeros_like(S_grid.S_x)
    
    # Calculate success probability (theta) using new formula
    if ("intercept_success" in S_grid.data_vars and "eta_success" in S_grid.data_vars 
        and "alpha_success" in S_grid.data_vars):
        S_grid["success_prob"] = 2 * (sp.expit(S_grid.S_x**S_grid.eta_success * S_grid.alpha_success) - 0.5)
    else:
        # Fallback if Bernoulli component not available
        S_grid["success_prob"] = xr.zeros_like(S_grid.S_x)

    if curve_type == "S":
        var = S_grid.S_x
        ylab = "S(x)"
        title = "skill"
    elif curve_type == "M":
        var = S_grid.M_x
        ylab = "M(x)"
        title = "senescence"
    elif curve_type == "K":
        var = S_grid.K_x
        ylab = "K(x)"
        title = "knowledge"
    elif curve_type == "harvest":
        var = S_grid.mu
        ylab = "harvest"
        title = "harvest"
    elif curve_type == "success":
        var = S_grid.success_prob
        ylab = "success probability"
        title = "success probability"
    
    # spaghetti plot
    for i in range(min(ndraws, len(var.sample))):
        plt.plot(S_grid.age, var.isel(sample=i).values, color="darkorange", alpha=0.05)

    plt.plot(S_grid.age, var.mean(dim="sample").values, color="darkorange", linewidth=2)

    plt.title(title)
    plt.xlabel("age")
    plt.ylabel(ylab)
    plt.show()


def plot_forager_contributions(
    marginal_contrib: xr.DataArray,
    data: xr.Dataset,
    sort_by_age: bool = True,
    figsize: tuple = (15, 20),
    nrows: int = 10,
    ncols: int = 5,
    ci_prob: float = 0.9
):
    """
    Plot the marginal contributions of foragers.
    
    Parameters:
    -----------
    marginal_contrib : xr.DataArray
        Marginal contributions data (from marginal_contributions function)
    data : xr.Dataset
        Original dataset with forager data (for ages and in_camp mask)
    sort_by_age : bool, default=True
        Whether to sort foragers by age.
    figsize : tuple, default=(15, 20)
        Figure size.
    nrows, ncols : int, default=10, 5
        Number of rows and columns in the subplot grid.
    ci_prob : float, default=0.9
        Credible interval level (e.g., 0.9 for 90% CI).
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Calculate the lower and upper quantiles for the credible interval
    alpha = (1 - ci_prob) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha
    
    # Calculate statistics from the full posterior
    mean_contributions = marginal_contrib.mean(dim=['chain', 'draw'])
    lower_ci = marginal_contrib.quantile(lower_quantile, dim=['chain', 'draw'])
    upper_ci = marginal_contrib.quantile(upper_quantile, dim=['chain', 'draw'])
    
    # Get ages and in_camp mask from the model data
    ages = data.age.values
    in_camp = data.in_camp.values
    
    # Create a list of forager indices and sort if needed
    forager_indices = range(len(data.coords['forager']))
    forager_info = list(zip(forager_indices, ages))
    
    if sort_by_age:
        forager_info.sort(key=lambda x: x[1])
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # Loop over foragers in the specified order
    for plot_idx, (forager_idx, age) in enumerate(forager_info):
        if plot_idx >= len(axes_flat):  # Safety check
            break
            
        # Get the forager ID
        forager_id = data.coords['forager'].values[forager_idx]
        
        # Get the data for this forager
        dates = marginal_contrib.coords['date'].values
        mean_values = mean_contributions.sel(forager=forager_id).values
        lower_values = lower_ci.sel(forager=forager_id).values
        upper_values = upper_ci.sel(forager=forager_id).values
        
        # Get the mask for this forager
        forager_mask = in_camp[forager_idx].astype(bool)
        
        # Apply the mask
        masked_dates = dates[forager_mask]
        masked_mean = mean_values[forager_mask]
        masked_lower = lower_values[forager_mask]
        masked_upper = upper_values[forager_mask]
        
        # Plot mean line (only for in_camp days)
        line, = axes_flat[plot_idx].plot(masked_dates, masked_mean)
        
        # Add shaded credible interval (only for in_camp days)
        axes_flat[plot_idx].fill_between(masked_dates, 
                                    masked_lower, 
                                    masked_upper, 
                                    alpha=0.3)
        
        # Set the title for each subplot including age
        axes_flat[plot_idx].set_title(f"Forager {forager_id} (Age: {age:.1f})")
        
        # Format date ticks if dates are datetime objects
        if len(masked_dates) > 0 and hasattr(masked_dates[0], 'strftime'):
            axes_flat[plot_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes_flat[plot_idx].tick_params(axis='x', rotation=45)

        # Add reference line at y=0
        axes_flat[plot_idx].axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
        
        # Add gridlines
        axes_flat[plot_idx].grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for idx in range(len(forager_info), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add common labels using figure-level commands
    ci_text = f"{int(ci_prob * 100)}% Credible Intervals"
    fig.suptitle(f"Marginal Contribution by Forager with {ci_text} (Sorted by Age)", fontsize=16)

    # Add common x and y labels
    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Contribution (expected kcal)', va='center', rotation='vertical', fontsize=14)

    # Adjust layout
    plt.tight_layout(rect=(0.05, 0.08, 0.95, 0.95))  # Leave space for labels and legend
    plt.subplots_adjust(top=0.92, bottom=0.12)  # Adjust top and bottom margins
    
    return fig, axes


def plot_shapley_contributions(
    shapley_results: Union[xr.Dataset, Dict[int, xr.DataArray]],
    data: xr.Dataset,
    group_indices: Optional[List[int]] = None,
    sort_by_age: bool = True,
    figsize: tuple = (15, 10),
    nrows: int = 5,
    ncols: int = 3,
    ci_prob: float = 0.9
):
    """
    Plot Shapley contributions for selected groups.
    
    Parameters:
    -----------
    shapley_results : Union[xr.Dataset, Dict[int, xr.DataArray]]
        Either a Dataset with 'shapley_contribution' variable (dimensions: group, forager),
        or a dictionary mapping group indices to Shapley contribution DataArrays
    data : xr.Dataset
        Original dataset with forager data (for ages)
    group_indices : List[int], optional
        List of group indices to plot. If None, plots all groups in shapley_results.
    sort_by_age : bool, default=True
        Whether to sort foragers by age within each group.
    figsize : tuple, default=(15, 10)
        Figure size.
    nrows, ncols : int, default=5, 3
        Number of rows and columns in the subplot grid.
    ci_prob : float, default=0.9
        Credible interval level (e.g., 0.9 for 90% CI).
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Convert to dict format for compatibility
    shapley_dict = _shapley_to_dict(shapley_results)
    
    # Calculate the lower and upper quantiles for the credible interval
    alpha = (1 - ci_prob) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha
    
    # Determine which groups to plot
    if group_indices is None:
        group_indices = list(shapley_dict.keys())
    
    # Limit to available groups
    group_indices = [g for g in group_indices if g in shapley_dict]
    
    if len(group_indices) == 0:
        print("No groups to plot")
        return None, None
    
    # Create figure and axes
    n_plots = min(len(group_indices), nrows * ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # Get ages for sorting
    ages = data.age.values
    
    # Loop over groups
    for plot_idx, group_idx in enumerate(group_indices[:n_plots]):
        shapley_data = shapley_dict[group_idx]
        
        # Get mean contributions (handle both old format with chain/draw and new format)
        if 'chain' in shapley_data.dims and 'draw' in shapley_data.dims:
            # Old format with posterior samples
            mean_contributions = shapley_data.mean(dim=['chain', 'draw'])
            lower_ci = shapley_data.quantile(lower_quantile, dim=['chain', 'draw'])
            upper_ci = shapley_data.quantile(upper_quantile, dim=['chain', 'draw'])
            has_ci = True
        else:
            # New format with posterior means
            mean_contributions = shapley_data
            has_ci = False
        
        # Get forager indices for this group
        forager_indices = shapley_data.coords['forager'].values
        
        # Create list of (forager_idx, age) tuples
        forager_info = [(int(fid), ages[int(fid)]) for fid in forager_indices]
        
        if sort_by_age:
            forager_info.sort(key=lambda x: x[1])
        
        # Extract values
        mean_values = [float(mean_contributions.sel(forager=fid).values) for fid, _ in forager_info]
        if has_ci:
            lower_values = [float(lower_ci.sel(forager=fid).values) for fid, _ in forager_info]
            upper_values = [float(upper_ci.sel(forager=fid).values) for fid, _ in forager_info]
        forager_labels = [f"Forager {fid}" for fid, _ in forager_info]
        
        # Plot
        ax = axes_flat[plot_idx]
        x_pos = np.arange(len(forager_labels))
        
        if has_ci:
            ax.barh(x_pos, mean_values, xerr=[np.array(mean_values) - np.array(lower_values),
                                               np.array(upper_values) - np.array(mean_values)],
                    alpha=0.7, color='steelblue', capsize=3)
        else:
            ax.barh(x_pos, mean_values, alpha=0.7, color='steelblue')
        ax.set_yticks(x_pos)
        ax.set_yticklabels(forager_labels, fontsize=8)
        ax.set_xlabel('Shapley Contribution (kcal)', fontsize=10)
        ax.set_title(f'Group {group_idx}', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


def plot_shapley_by_forager(
    shapley_results: Union[xr.Dataset, Dict[int, xr.DataArray]],
    data: xr.Dataset,
    sort_by_age: bool = True,
    figsize: tuple = (15, 20),
    nrows: int = 10,
    ncols: int = 5,
):
    """
    Plot Shapley contributions by forager in a grid layout, similar to plot_forager_contributions.
    
    Each subplot shows Shapley contributions for one forager across all groups they participated in,
    ordered by date (group_date).
    
    Parameters:
    -----------
    shapley_results : Union[xr.Dataset, Dict[int, xr.DataArray]]
        Either a Dataset with 'shapley_contribution' variable 
        (dimensions: sample, group, forager OR group, forager),
        or a dictionary mapping group indices to Shapley contribution DataArrays
    data : xr.Dataset
        Original dataset with forager data (for ages and group_date)
    sort_by_age : bool, default=True
        Whether to sort foragers by age.
    figsize : tuple, default=(15, 20)
        Figure size.
    nrows, ncols : int, default=10, 5
        Number of rows and columns in the subplot grid.
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Convert to Dataset format if needed
    if isinstance(shapley_results, xr.Dataset):
        shapley_var = shapley_results['shapley_contribution']
        # If sample dimension exists, take mean across samples first
        if 'sample' in shapley_var.dims:
            shapley_var = shapley_var.mean(dim='sample')
    else:
        # Convert dict to Dataset
        shapley_dict = shapley_results
        n_groups = len(data.coords['group'])
        n_foragers = len(data.coords['forager'])
        shapley_array = np.full((n_groups, n_foragers), np.nan, dtype=np.float64)
        
        for group_idx, group_data in shapley_dict.items():
            for forager_idx in group_data.coords['forager'].values:
                contrib_value = float(group_data.sel(forager=forager_idx).values)
                shapley_array[group_idx, forager_idx] = contrib_value
        
        shapley_var = xr.DataArray(
            shapley_array,
            dims=['group', 'forager'],
            coords={
                'group': data.coords['group'],
                'forager': data.coords['forager'],
            },
            name='shapley_contribution'
        )
    
    # Get ages and group_date
    ages = data.age.values
    group_dates = data.group_date.values
    
    # Create a list of forager indices and sort if needed
    forager_indices = range(len(data.coords['forager']))
    forager_info = list(zip(forager_indices, ages))
    
    if sort_by_age:
        forager_info.sort(key=lambda x: x[1])
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    # Loop over foragers
    for plot_idx, (forager_idx, age) in enumerate(forager_info):
        if plot_idx >= len(axes_flat):  # Safety check
            break
            
        # Get the forager ID for display
        forager_id = data.coords['forager'].values[forager_idx]
        
        # Collect Shapley contributions for this forager across all groups
        contributions = []
        dates = []
        group_ids = []
        
        # Get contributions from Dataset (use isel for positional indexing)
        forager_contribs = shapley_var.isel(forager=forager_idx)
        valid_mask = ~np.isnan(forager_contribs.values)
        
        if np.any(valid_mask):
            valid_groups = forager_contribs.group.values[valid_mask]
            valid_values = forager_contribs.values[valid_mask]
            
            for group_val, contrib_value in zip(valid_groups, valid_values):
                # Get group index
                if isinstance(group_val, (int, np.integer)):
                    group_idx = int(group_val)
                else:
                    group_idx = int(np.where(shapley_var.coords['group'].values == group_val)[0][0])
                
                contributions.append(float(contrib_value))
                dates.append(group_dates[group_idx])
                group_ids.append(group_idx)
        
        # Sort by date
        if len(dates) > 0:
            sorted_indices = np.argsort(dates)
            contributions = [contributions[i] for i in sorted_indices]
            dates = [dates[i] for i in sorted_indices]
            group_ids = [group_ids[i] for i in sorted_indices]
        
        # Plot
        ax = axes_flat[plot_idx]
        
        if len(contributions) > 0:
            ax.plot(dates, contributions, marker='o', markersize=4, linewidth=1.5, alpha=0.7)
            
            # Format date ticks if dates are datetime objects
            if hasattr(dates[0], 'strftime'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        else:
            # No contributions for this forager
            ax.text(0.5, 0.5, 'No groups', ha='center', va='center', transform=ax.transAxes)
        
        # Set the title for each subplot including age
        ax.set_title(f"Forager {forager_id} (Age: {age:.1f})")
        
        # # Add reference line at y=0
        # ax.axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Hide any unused subplots
    for idx in range(len(forager_info), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add common labels using figure-level commands
    fig.suptitle("Shapley Contribution by Forager Across Groups (Sorted by Age)", fontsize=16)
    
    # Add common x and y labels
    fig.text(0.5, 0.04, 'Date', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Shapley Contribution (kcal)', va='center', rotation='vertical', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=(0.05, 0.08, 0.95, 0.95))  # Leave space for labels
    plt.subplots_adjust(top=0.92, bottom=0.12)  # Adjust top and bottom margins
    
    return fig, axes


def plot_shapley_summary(
    shapley_results: Union[xr.Dataset, Dict[int, xr.DataArray]],
    data: xr.Dataset,
    group_indices: Optional[List[int]] = None,
    figsize: tuple = (12, 6),
    show_error_bars: bool = False,
):
    """
    Plot summary of Shapley contributions across groups.
    
    Shows total contributions summed across all groups for each forager.
    Error bars (if enabled) show variation across groups, not MCMC uncertainty.
    
    Parameters:
    -----------
    shapley_results : Union[xr.Dataset, Dict[int, xr.DataArray]]
        Either a Dataset with 'shapley_contribution' variable 
        (dimensions: sample, group, forager OR group, forager),
        or a dictionary mapping group indices to Shapley contribution DataArrays
    data : xr.Dataset
        Original dataset with forager data
    group_indices : List[int], optional
        List of group indices to include. If None, uses all groups.
    figsize : tuple, default=(12, 6)
        Figure size.
    show_error_bars : bool, default=False
        Whether to show error bars (variation across groups).
        Since we use posterior means, error bars represent variation across groups,
        not MCMC uncertainty.
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Convert to Dataset format if needed
    if isinstance(shapley_results, xr.Dataset):
        shapley_var = shapley_results['shapley_contribution']
        # If sample dimension exists, take mean across samples first
        if 'sample' in shapley_var.dims:
            shapley_var = shapley_var.mean(dim='sample')
    else:
        # Convert dict to Dataset
        shapley_dict = shapley_results
        n_groups = len(data.coords['group'])
        n_foragers = len(data.coords['forager'])
        shapley_array = np.full((n_groups, n_foragers), np.nan, dtype=np.float64)
        
        for group_idx, group_data in shapley_dict.items():
            for forager_idx in group_data.coords['forager'].values:
                contrib_value = float(group_data.sel(forager=forager_idx).values)
                shapley_array[group_idx, forager_idx] = contrib_value
        
        shapley_var = xr.DataArray(
            shapley_array,
            dims=['group', 'forager'],
            coords={
                'group': data.coords['group'],
                'forager': data.coords['forager'],
            },
            name='shapley_contribution'
        )
    
    # Determine which groups to include
    if group_indices is not None:
        shapley_var = shapley_var.sel(group=group_indices)
    
    # Aggregate contributions across groups by forager
    # Sum across groups (ignoring NaN)
    total_contributions = shapley_var.sum(dim='group', skipna=True)
    
    # Get ages for sorting
    ages = data.age.values
    forager_indices = range(len(data.coords['forager']))
    forager_info = [(fid, ages[fid]) for fid in forager_indices]
    forager_info.sort(key=lambda x: x[1])
    
    # Extract values (use isel for positional indexing)
    mean_values = [float(total_contributions.isel(forager=fid).values) if not np.isnan(total_contributions.isel(forager=fid).values) else 0.0 
                   for fid, _ in forager_info]
    forager_labels = [f"Forager {data.coords['forager'].values[fid]}" for fid, _ in forager_info]
    
    # Compute error bars if requested
    if show_error_bars:
        lower_values = []
        upper_values = []
        for fid, _ in forager_info:
            forager_contribs = shapley_var.isel(forager=fid)
            valid_values = forager_contribs.values[~np.isnan(forager_contribs.values)]
            if len(valid_values) > 1:
                alpha = 0.05  # 90% CI
                lower_values.append(np.quantile(valid_values, alpha))
                upper_values.append(np.quantile(valid_values, 1 - alpha))
            else:
                lower_values.append(mean_values[len(lower_values)])
                upper_values.append(mean_values[len(upper_values)])
    else:
        lower_values = mean_values
        upper_values = mean_values
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(forager_labels))
    
    if show_error_bars:
        ax.barh(x_pos, mean_values, xerr=[np.array(mean_values) - np.array(lower_values),
                                           np.array(upper_values) - np.array(mean_values)],
                alpha=0.7, color='steelblue', capsize=3)
    else:
        ax.barh(x_pos, mean_values, alpha=0.7, color='steelblue')
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(forager_labels, fontsize=10)
    ax.set_xlabel('Total Shapley Contribution (kcal)', fontsize=12)
    title = 'Shapley Contributions Across All Groups'
    if show_error_bars:
        title += ' (Error bars: variation across groups)'
    ax.set_title(title, fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

