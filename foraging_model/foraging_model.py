from functools import cache, cached_property
import xarray as xr
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .hurdle_gamma import hurdle_gamma_logp, hurdle_gamma_rng

class ForagingModel:
    def __init__(
            self,
            data: xr.Dataset,
            ) -> None:
        self.data = data

    @property
    def already_built(self) -> bool:
        return hasattr(self, "model")
    
    @property
    def already_rescaled(self) -> bool:
        return hasattr(self, "rescaled_predictive")
    
    def build_model(self):

        import numpy as np
        import pytensor.tensor as pt

        coords = self.data.coords
        coords = {key: value.to_numpy() for key, value in coords.items()}

        with pm.Model(coords=coords) as self.model:
            # Data
            kcal = pm.Data('kcal_scaled', self.data.kcal_scaled.values, dims="group")
            age = pm.Data('age', self.data.age_scaled.values, dims="forager")

            forager_ids = pm.Data("forager_ids", self.data.forager_ids.values, dims=("group", "forager_in_group"))
            # group size derived from forager_ids
            valid_foragers = forager_ids >= 0  # For skill calculation
            present_foragers = pt.or_(forager_ids >= 0, pt.eq(forager_ids, -99))  # For group size
            group_size = pm.Deterministic("group_size", pt.sum(present_foragers, axis=1), dims="group")
            max_groupsize = group_size.max()

            # mean non-zero kcal
            intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=0.5)
            # non-zero-return probability
            intercept_success = pm.Normal("intercept_success", mu=0, sigma=0.5)

            # gamma shape
            shape = pm.Gamma("shape", alpha=10, beta=10)

            # group size effect
            b_groupsize_mu = pm.Normal("b_groupsize_mu", mu=0.2, sigma=0.2)
            b_groupsize_success = pm.Normal("b_groupsize_success", mu=1, sigma=0.5)

            # --- skill curve ---
            m0 = pm.Normal("m0", mu=-1, sigma=1)
            k0 = pm.Normal('k0', mu=1, sigma=1)
            b0 = pm.Normal('b0', mu=0, sigma=1)

            m = pm.Deterministic("m", pt.exp(m0))
            k = pm.Deterministic("k", pt.exp(k0))
            b = pm.Deterministic("b", pt.exp(b0))

            M_x = pm.Deterministic("M", pt.exp(-m * age), dims="forager")
            K_x = pm.Deterministic("K", 1 - pt.exp(-k * age), dims="forager")
            S_x = pm.Deterministic("S", M_x * K_x**b, dims="forager")

            # For each group, get average skill of forager ids
            safe_forager_ids = pt.where(valid_foragers, forager_ids, 0)
            S_x_selected = S_x[safe_forager_ids]

            # mask for groups with no valid foragers
            # only relevant for interventions
            valid_groups = pt.sum(valid_foragers, axis=1) > 0
            #valid_idx = pt.as_tensor_variable(valid_groups)

            # avoid division by zero
            S_x_group = pm.Deterministic(
                "S_x_grouped",
                pt.where(
                    valid_groups,
                    pt.sum(S_x_selected * valid_foragers, axis=1) / pt.sum(valid_foragers, axis=1),
                    0  # when no valid foragers
                ),
                dims="group"
            )
            
            alpha_mu = pm.Deterministic("alpha_mu", pt.exp(intercept_mu + b_groupsize_mu*pt.log(group_size)), dims="group")
            alpha_success = pm.Deterministic("alpha_success", pt.exp(intercept_success + b_groupsize_success*pt.log(group_size)), dims="group")

            eta_mu0 = pm.Normal("eta_mu0", mu=0, sigma=1)
            eta_success0 = pm.Normal("eta_success0", mu=0, sigma=1)
            eta_mu = pm.Deterministic("eta_mu", pt.exp(eta_mu0))
            eta_success = pm.Deterministic("eta_success", pt.exp(eta_success0))

            # expected kcal
            mu = pm.Deterministic("mu", pt.where(valid_groups, S_x_group**eta_mu * alpha_mu, 0), dims="group")

            theta = pm.Deterministic("theta", 2*(pm.math.invlogit(pt.where(valid_groups, S_x_group**eta_success * alpha_success, 0)) - 0.5), dims="group")

            expected = pm.Deterministic("expected", pt.where(valid_groups, mu * theta, 0), dims="group")

            pm.CustomDist(
                "kcal",
                theta, # theta (binomial)
                shape, # alpha (gamma) - scalar, no indexing needed
                shape / mu, # beta (gamma)
                logp = hurdle_gamma_logp,
                random = hurdle_gamma_rng,
                observed = kcal,
                dims = ("group"),
                )

        return self
        
    def fit(self, **sample_params):
        if not self.already_built:
            self.build_model()

        if sample_params.get("nuts_sampler") == "nutpie":
            import nutpie

            compiled_model = nutpie.compile_pymc_model(
                    self.model
                )
            sample_params_copy = sample_params.copy()
            sample_params_copy.pop("nuts_sampler")
            #sample_params_copy.pop("var_names")
            self.idata = nutpie.sample(compiled_model, **sample_params_copy)
        else:
            with self.model:
                self.idata = pm.sample(**sample_params)

        with self.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(self.idata)
            self.idata.extend(prior)
            self.idata.extend(posterior_predictive)

        return self
    
    def rescale_predictive(self):
        if not self.already_rescaled:
            self.idata.posterior_predictive["kcal"] *= self.data.kcal_scale
            self.idata.prior_predictive["kcal"] *= self.data.kcal_scale
            self.idata.observed_data["kcal"] *= self.data.kcal_scale

        self.rescaled_predictive = True

        return self
    
    def plot_curve(
            self,
            prior: bool = False,
            ndraws: int = 100,
            type: str = "S",
            ) -> None:
        """Plot implied skill curves.
        
        Parameters
        ----------
        prior : bool, optional
            Whether to use prior or posterior samples, by default False
        ndraws : int, optional
            Number of draws to use, by default 100
            
        Returns
        -------
        xr.Dataset
            Dataset containing skill curves for each age value
        """
        import arviz as az
        import scipy.special as sp

        if not hasattr(self, "idata"):
            raise ValueError("Fit the model first")
        
        if type not in ["S", "M", "K", "harvest", "success"]:
            raise ValueError("Invalid curve type, must be one of S, M, or K")

        if prior:
            parameters = az.extract(self.idata, group="prior")
        else:
            parameters = az.extract(self.idata, group="posterior")

        # Create age grid
        age_grid = np.linspace(0, 1, 100)
        
        # Create dataset with parameters
        S_grid = xr.Dataset(
            {
                "m": parameters["m"],
                "k": parameters["k"],
                "b": parameters["b"],
                "intercept_success": parameters["intercept_success"],
                "eta_mu": parameters["eta_mu"],
                "eta_success": parameters["eta_success"],
            }
        )
        
        # Add age dimension
        S_grid = S_grid.expand_dims({"age_scaled": age_grid})
        S_grid["age"] = S_grid.age_scaled * self.data.age_scale
        
        # Calculate skill curve
        S_grid["K_x"] = 1 - np.exp(-S_grid.k * S_grid.age_scaled)
        S_grid["M_x"] = np.exp(-S_grid.m * S_grid.age_scaled)
        S_grid["S_x"] = S_grid.M_x * S_grid.K_x**S_grid.b
        S_grid["mu"] = S_grid.S_x**S_grid.eta_mu
        S_grid["theta"] = 2*(sp.expit(S_grid.S_x**S_grid.eta_success * np.exp(S_grid.intercept_success)) - 0.5)

        if type == "S":
            var = S_grid.S_x
            ylab = "S(x)"
            title = "skill"
        elif type == "M":
            var = S_grid.M_x
            ylab = "M(x)"
            title = "senescence"
        elif type == "K":
            var = S_grid.K_x
            ylab = "K(x)"
            title = "knowledge"
        elif type == "harvest":
            var = S_grid.mu
            ylab = "harvest"
            title = "harvest"
        elif type == "success":
            var = S_grid.theta
            ylab = "success"
            title = "success"
        
        # spaghetti plot
        for i in range(ndraws):
            plt.plot(S_grid.age, var.isel(sample=i).values, color="darkorange", alpha=0.05)

        plt.plot(S_grid.age, var.mean(dim="sample").values, color="darkorange", linewidth=2)

        plt.title(title)
        plt.xlabel("age")
        plt.ylabel(ylab)
        plt.show()

    @cache
    def marginal_contributions(
            self,
            group: str = "posterior",
            ) -> xr.Dataset:
        """
        Calculate the marginal contributions of each forager to the expected kcal on each day.
        """
        if group not in ["posterior", "prior"]:
            raise ValueError("Invalid group, must be one of posterior or prior")

        if group == "posterior":
            idata = self.idata.posterior
        else:
            idata = self.idata.prior

        expected = idata["expected"]
        forager_indices = range(len(self.data.coords["forager"]))
        forager_ids = self.data.coords["forager"].values

        forager_contributions = []

        for forager_idx in forager_indices:
            # Set this forager's ID to -1 (remove from groups)
            forager_ids_copy = self.idata["constant_data"]["forager_ids"].copy()
            forager_ids_copy = np.where(forager_ids_copy == forager_idx, -1, forager_ids_copy)
            intervention = {"forager_ids": forager_ids_copy}
            
            # Apply the intervention
            intervened_model = pm.do(self.model, intervention)
            
            # Sample posterior predictive
            with intervened_model:
                if group == "posterior":
                    idata_intervened = pm.sample_posterior_predictive(
                        self.idata,
                        var_names=["expected"]
                    )
                else:
                    idata_intervened = pm.sample_prior_predictive(
                        var_names=["expected"]
                    )
            
            # Calculate the difference
            if group == "posterior":
                diff = expected - idata_intervened.posterior_predictive["expected"]
            else:
                diff = expected - idata_intervened.prior["expected"]
            diff *= self.data.kcal_scale
            
            # Add the date coordinate
            diff = diff.assign_coords(date=("group", self.data.group_date.values))
            
            # Group by date while preserving chains and draws
            diff_by_date = diff.groupby('date').sum()
            
            # Keep only the dimensions we need
            reduced_dims = [dim for dim in diff_by_date.dims if dim not in ['date', 'chain', 'draw']]
            if reduced_dims:
                diff_by_date = diff_by_date.sum(dim=reduced_dims)
            
            # Store this forager's contribution
            forager_contributions.append(diff_by_date)
    
        # Combine all forager contributions into a single DataArray
        # by stacking them along a new 'forager' dimension
        combined = xr.concat(forager_contributions, dim=pd.Index(forager_ids, name='forager'))
        
        return combined.rename('marginal_contribution')
    
    
    def plot_forager_contributions(
        self,
        sort_by_age=True,
        figsize=(15, 20),
        nrows=10,
        ncols=5,
        group="posterior",
        ci_prob=0.9
    ):
        """
        Plot the marginal contributions of foragers.
        
        Parameters:
        -----------
        sort_by_age : bool, default=True
            Whether to sort foragers by age.
        figsize : tuple, default=(15, 20)
            Figure size.
        nrows, ncols : int, default=10, 5
            Number of rows and columns in the subplot grid.
        group : str, default="posterior"
            The inference group to use, either "posterior" or "prior".
        ci_level : float, default=0.9
            Credible interval level (e.g., 0.9 for 90% CI).
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        # Get the cached contributions
        marginal_contrib = self.marginal_contributions(group=group)
        
        # Calculate the lower and upper quantiles for the credible interval
        alpha = (1 - ci_prob) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha
        
        # Calculate statistics from the full posterior
        mean_contributions = marginal_contrib.mean(dim=['chain', 'draw'])
        lower_ci = marginal_contrib.quantile(lower_quantile, dim=['chain', 'draw'])
        upper_ci = marginal_contrib.quantile(upper_quantile, dim=['chain', 'draw'])
        
        # Get ages and in_camp mask from the model data
        ages = self.data.age.values
        in_camp = self.data.in_camp.values
        
        # Create a list of forager indices and sort if needed
        forager_indices = range(len(self.data.coords['forager']))
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
            forager_id = self.data.coords['forager'].values[forager_idx]
            
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
            if hasattr(dates[0], 'strftime'):
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
        plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.95])  # Leave space for labels and legend
        plt.subplots_adjust(top=0.92, bottom=0.12)  # Adjust top and bottom margins
        
        return fig, axes
        


