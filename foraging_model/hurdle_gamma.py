import numpy as np
import pytensor.tensor as pt
import pymc as pm
from typing import Optional, Tuple

def hurdle_gamma_logp(
    value: pt.TensorVariable,
    theta: pt.TensorVariable,
    alpha: pt.TensorVariable,
    beta: pt.TensorVariable,
) -> pt.TensorVariable:
    return pt.switch(
        value > 0,
        pt.log(theta) + pm.logp(pm.Gamma.dist(alpha=alpha, beta=beta), value=value),
        pt.math.log1p(-theta)
    )
    
def hurdle_gamma_rng(
        *dist_params,
        rng: Optional[np.random.Generator] = None,
        size : Optional[Tuple[int]]=None
) -> np.ndarray | float:  
    """Random number generator for hurdle gamma distribution"""
    if rng is None:
        rng = np.random.default_rng()
    theta, alpha, beta = dist_params
    is_nonzero = rng.binomial(1, theta, size=size)
    gamma_samples = rng.gamma(shape=alpha, scale=1/beta, size=size)
    result = is_nonzero * gamma_samples
        
    return result
