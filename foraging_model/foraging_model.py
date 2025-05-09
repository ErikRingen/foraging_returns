import xarray as xr
import pymc as pm
import numpy as np

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
        
            max_groupsize = self.data.group_size.values.max()
            kcal_scaled = pm.Data('kcal_scaled', self.data.kcal_scaled.values, dims="group")
            group_size = pm.Data('group_size', self.data.group_size.values, dims="group")

            # non-zero-return probability
            theta = pm.Beta("theta", alpha=6, beta=1)

            # mean non-zero kcal
            intercept = pm.Normal("intercept", mu=0, sigma=0.5)

            # gamma shape
            shape = pm.HalfNormal("shape", sigma=1)

            # group size total effect
            b_groupsize = pm.Normal("b_groupsize", mu=0, sigma=0.1)
            # dirichlet decomposition
            s_groupsize_raw = pm.Dirichlet("s_groupsize_raw", a=pt.ones(max_groupsize-1) * 2.0)
            s_groupsize = pt.concatenate([pt.zeros(1), s_groupsize_raw])
            s_groupsize_cumsum = pt.cumsum(s_groupsize)

            # Calculate effect for each group size
            size_idx = group_size - 1
            groupsize_effect = pm.Deterministic("groupsize_effect", b_groupsize * s_groupsize_cumsum[size_idx], dims="group")

            # expected kcal
            mu = pm.Deterministic("mu", pt.exp(intercept + groupsize_effect))

            pm.CustomDist(
                "kcal",
                theta, # theta (binomial)
                shape, # alpha (gamma)
                shape / mu, # beta (gamma)
                logp = hurdle_gamma_logp,
                random = hurdle_gamma_rng,
                observed = kcal_scaled,
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

