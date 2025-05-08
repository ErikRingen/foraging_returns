import xarray as xr
import pymc as pm


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

        coords = self.data.coords
        coords = {key: value.to_numpy() for key, value in coords.items()}

        with pm.Model(coords=coords) as self.model:

            # non-zero-return probability
            psi = pm.Beta("psi", alpha=6, beta=1)

            # mean non-zero kcal
            mu = pm.HalfNormal("mu", sigma=2)

            sigma = pm.HalfNormal("sigma", sigma=2)

            pm.HurdleGamma(
                "kcal",
                psi = psi,
                mu = mu,
                sigma = sigma,
                observed = self.data.kcal_scaled,
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

