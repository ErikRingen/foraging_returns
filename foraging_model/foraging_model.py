import xarray as xr
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
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
            group_size = pm.Data('group_size', self.data.group_size.values, dims="group")
            max_groupsize = group_size.max()

            forager_ids = pm.Data("forager_ids", self.data.forager_ids.values, dims=("group", "forager_in_group"))

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
            valid_foragers = forager_ids >= 0
            # Replace -1 with 0 for safe indexing, then mask out after
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
            mu = pm.Deterministic("mu", S_x_group**eta_mu * alpha_mu, dims="group")
            theta = pm.Deterministic("theta", 2*(pm.math.invlogit(S_x_group**eta_success * alpha_success) - 0.5), dims="group")

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


