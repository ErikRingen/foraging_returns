import xarray as xr
import pymc as pm
import numpy as np
import pandas as pd
import nutpie
import pytensor.tensor as pt


class ForagingModel:
    def __init__(
            self,
            data: xr.Dataset,
            target_scaling: str | None = None,
            age_scaling: str | None = None,
            ) -> None:
        """
        Initialize ForagingModel.
        
        Parameters
        ----------
        data : xr.Dataset
            Dataset with unscaled data (from ForagingData.to_dataset())
        target_scaling : str | None, optional
            Scaling method for target variable ('mean', 'max', or None)
        age_scaling : str | None, optional
            Scaling method for age variable ('mean', 'max', or None)
        """
        self.data = data
        self.target_scaling = target_scaling
        self.age_scaling = age_scaling
        
        # Validate scaling options
        if self.target_scaling not in ['mean', 'max', None]:
            raise ValueError(f"Target scaling must be 'mean', 'max', or None, got {self.target_scaling}")
        if self.age_scaling not in ['mean', 'max', None]:
            raise ValueError(f"Age scaling must be 'mean', 'max', or None, got {self.age_scaling}")
        
        # Scale factors will be stored here after build_model()
        self.kcal_scale = None
        self.age_scale = None

    @property
    def already_built(self) -> bool:
        return hasattr(self, "model")
    
    
    def build_model(self):
        # Extract coordinates - convert to numpy arrays for nutpie compatibility
        coords = {}
        for key, value in self.data.coords.items():
            if hasattr(value, 'to_numpy'):
                try:
                    coord_values = value.to_numpy()
                    # Convert datetime64 and non-numeric to numeric indices for nutpie
                    if pd.api.types.is_datetime64_any_dtype(coord_values.dtype):
                        coords[key] = np.arange(len(coord_values)).tolist()
                    elif coord_values.dtype == object or (hasattr(coord_values.dtype, 'kind') and coord_values.dtype.kind in ['U', 'S', 'O']):
                        coords[key] = np.arange(len(coord_values)).tolist()
                    else:
                        coords[key] = coord_values.tolist()
                except (ValueError, TypeError):
                    coords[key] = list(range(len(value))) if hasattr(value, '__len__') else [0]
            else:
                coords[key] = value

        # Compute scale factors from data (before model)
        # These will be used to create scaled versions inside the model
        if self.target_scaling == 'mean':
            target_col = self.data['kcal']
            non_zero = target_col.where(target_col != 0)
            self.kcal_scale = float(non_zero.mean().values)
        elif self.target_scaling == 'max':
            self.kcal_scale = float(self.data['kcal'].max().values)
        else:
            self.kcal_scale = 1.0
        
        if self.age_scaling == 'mean':
            age_col = self.data['age']
            non_zero = age_col.where(age_col != 0)
            self.age_scale = float(non_zero.mean().values)
        elif self.age_scaling == 'max':
            self.age_scale = float(self.data['age'].max().values)
        else:
            self.age_scale = 1.0

        with pm.Model(coords=coords) as self.model:
            # Data (unscaled)
            kcal_raw = pm.Data('kcal_raw', self.data['kcal'].values, dims="group")
            age_raw = pm.Data('age_raw', self.data['age'].values, dims="forager")
            
            # Create scaled versions as deterministic transformations
            if self.kcal_scale != 1.0:
                kcal = pm.Deterministic('kcal_scaled', kcal_raw / self.kcal_scale, dims="group")
            else:
                kcal = pm.Deterministic('kcal_scaled', kcal_raw, dims="group")
            
            if self.age_scale != 1.0:
                age = pm.Deterministic('age_scaled', age_raw / self.age_scale, dims="forager")
            else:
                age = pm.Deterministic('age_scaled', age_raw, dims="forager")
            forager_ids = pm.Data("forager_ids", self.data.forager_ids.values, dims=("group", "forager_in_group"))
            
            # Forager-daily success data (Bernoulli component)
            # Long format: only eligible (forager, date) observations
            forager_success = pm.Data('forager_success', self.data.forager_success.values, dims="forager_date")
            forager_idx = pm.Data('forager_idx', self.data.forager_idx.values, dims="forager_date")

            # Group size derived from forager_ids
            valid_foragers = forager_ids >= 0
            present_foragers = pt.or_(forager_ids >= 0, pt.eq(forager_ids, -99))
            group_size = pm.Deterministic("group_size", pt.sum(present_foragers, axis=1), dims="group")

            # Scalar distributions
            intercept_mu = pm.Normal("intercept_mu", mu=0, sigma=0.5)
            shape = pm.Gamma("shape", alpha=10, beta=10)
            b_groupsize_mu = pm.Normal("b_groupsize_mu", mu=0.2, sigma=0.2)

            # Skill curve parameters (scalar)
            m0 = pm.Normal("m0", mu=-1, sigma=1)
            k0 = pm.Normal('k0', mu=1, sigma=1)
            b0 = pm.Normal('b0', mu=0, sigma=1)

            # Skill curve
            m = pm.Deterministic("m", pt.exp(m0))
            k = pm.Deterministic("k", pt.exp(k0))
            b = pm.Deterministic("b", pt.exp(b0))

            M_x = pm.Deterministic("M", pt.exp(-m * age), dims="forager")
            K_x = pm.Deterministic("K", 1 - pt.exp(-k * age), dims="forager")
            S_x = pm.Deterministic("S", M_x * K_x**b, dims="forager")

            # Skill elasticities
            eta_success0 = pm.Normal("eta_success0", mu=0, sigma=1)
            eta_success = pm.Deterministic("eta_success", pt.exp(eta_success0))

            eta_mu0 = pm.Normal("eta_mu0", mu=0, sigma=1)
            eta_mu = pm.Deterministic("eta_mu", pt.exp(eta_mu0))

            # Alphas
            intercept_success = pm.Normal("intercept_success", mu=0, sigma=0.5)

            alpha_success = pm.Deterministic("alpha_success", pt.exp(intercept_success))

            # Bernoulli component: Forager-daily success
            theta = pm.Deterministic("theta", 2*(pt.sigmoid(S_x[forager_idx]**eta_success * alpha_success) - 0.5), dims="forager_date")
            
            pm.Bernoulli("non_zero_prod", p=theta, observed=forager_success, dims="forager_date")

            # Best-top-k group skill calculation (optimized)
            # For each group, compute: μ(S) = max_{1 ≤ k ≤ |S|} g(k) * (avg of top k skills)^η
            # where g(k) = exp(intercept_μ + b_{groupsize,μ} log k)
            
            # Extract skills for valid foragers in each group
            safe_forager_ids = pt.where(valid_foragers, forager_ids, 0)
            S_x_selected = S_x[safe_forager_ids]  # Shape: (group, forager_in_group)
            
            # Mask for groups with no valid foragers
            valid_groups = pt.sum(valid_foragers, axis=1) > 0
            
            # Get max group size and number of groups (static values)
            max_group_size = int(self.data.forager_ids.shape[1])
            n_groups = len(self.data.coords['group'])
            
            # Sort skills descending for each group (optimized: single sort operation)
            # Use negative values for ascending sort, then negate back
            S_x_neg = -S_x_selected
            S_x_padded = pt.where(valid_foragers, S_x_neg, 1e10)
            sort_indices = pt.argsort(S_x_padded, axis=1)
            row_indices = pt.arange(n_groups)[:, None]
            S_x_sorted = -S_x_padded[row_indices, sort_indices]  # Sorted descending
            
            # Get group sizes once
            group_sizes_int = pt.sum(valid_foragers, axis=1)  # Keep as int for efficiency
            group_sizes_float = pt.cast(group_sizes_int, dtype='float64')
            
            # Mask for valid entries after sorting (only compute for valid positions)
            valid_mask = pt.cast(
                pt.arange(max_group_size, dtype='int32')[None, :] < pt.cast(group_sizes_int, dtype='int32')[:, None],
                dtype='float64'
            )
            
            # Cumulative sum of sorted skills (cast once, mask efficiently)
            cumsum = pt.cumsum(pt.cast(S_x_sorted, dtype='float64') * valid_mask, axis=1)
            
            # Create k values once: 1, 2, ..., max_group_size
            k_values = pt.arange(1, max_group_size + 1, dtype='float64')
            
            # Compute g(k) = exp(intercept_μ + b_{groupsize,μ} log k) once
            log_k = pt.log(k_values)
            g_k = pt.exp(intercept_mu + b_groupsize_mu * log_k)  # Shape: (max_group_size,)
            
            # Compute top-k averages efficiently: cumsum[:, k-1] / k
            # Use broadcasting instead of tiling large matrices
            k_matrix = pt.cast(pt.arange(max_group_size, dtype='int32') + 1, dtype='float64')[None, :]  # (1, max_group_size)
            group_sizes_expanded = group_sizes_float[:, None]  # (n_groups, 1)
            
            # Only compute averages where k <= group_size
            valid_k_mask = k_matrix <= group_sizes_expanded
            top_k_averages = pt.where(
                valid_k_mask,
                cumsum / k_matrix,  # Broadcasting: cumsum (n_groups, max_group_size) / k_matrix (1, max_group_size)
                0.0
            )
            
            # Compute values efficiently using broadcasting
            # Mask invalid k values before computing to avoid unnecessary operations
            top_k_averages_masked = top_k_averages * valid_k_mask  # Zero out invalid k
            top_k_averages_safe = pt.maximum(top_k_averages_masked, 1e-10)  # Faster than where
            g_k_expanded = g_k[None, :]  # (1, max_group_size) - broadcast instead of tile
            values_by_k = g_k_expanded * (top_k_averages_safe ** eta_mu)  # Broadcasting
            
            # Find maximum over k for each group (invalid k already masked to 0)
            mu_best_topk = pt.max(values_by_k, axis=1)  # Shape: (group,)
            
            # Set to 0 for invalid groups
            mu_best_topk_safe = pt.where(valid_groups, mu_best_topk, 0.0)
            
            # Expected returns using best-top-k
            mu = pm.Deterministic("mu", mu_best_topk_safe, dims="group")

            # Gamma component: Group-trip returns
            rate = pm.Deterministic("rate", shape / mu, dims="group")
            pm.Gamma("kcal", alpha=shape, beta=rate, observed=kcal, dims="group")

        return self
        
    def fit(
        self,
        tune: int = 150,
        draws: int = 250,
        chains: int = 4,
        random_seed: int | None = None,
        **kwargs
    ):
        """
        Fit the model using nutpie sampler.
        
        Parameters
        ----------
        tune : int, default=150
            Number of tuning steps
        draws : int, default=250
            Number of posterior samples per chain
        chains : int, default=4
            Number of chains
        random_seed : int | None, default=None
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to nutpie.sample()
        
        Returns
        -------
        InferenceData
            Returns InferenceData object containing the posterior samples
        """
        if not self.already_built:
            self.build_model()
        
        compiled_model = nutpie.compile_pymc_model(self.model)
        
        # Prepare sample parameters (remove nuts_sampler if present, as we always use nutpie)
        sample_kwargs = {
            "tune": tune,
            "draws": draws,
            "chains": chains,
            **kwargs
        }
        if random_seed is not None:
            sample_kwargs["seed"] = random_seed
        
        self.idata = nutpie.sample(compiled_model, **sample_kwargs)

        return self.idata
    
    def rescale_predictive(self):
        """Rescale predictive samples back to original scale."""
        if self.kcal_scale is None:
            raise ValueError("Model must be built before rescaling. Call build_model() first.")
        
        if not hasattr(self, "idata"):
            raise ValueError("Model must be fitted before rescaling. Call fit() first.")
        
        self.idata.posterior_predictive["kcal"] *= self.kcal_scale
        self.idata.prior_predictive["kcal"] *= self.kcal_scale
        if "kcal" in self.idata.observed_data:
            self.idata.observed_data["kcal"] *= self.kcal_scale

        return self
        


