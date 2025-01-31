from typing import List, Union, Tuple, Literal
import jax.numpy as jnp
import jax.random as jr
import warnings
from .utils import list_array_scaled, dirichlet_expectation

""" Functions for setting up Dirichlet priors

__author__: Lancelot Da Costa
"""



def dirichlet_prior(template: List[jnp.ndarray],
                   init: Literal["uniform", "like", "random"] = "uniform",
                   scale: float = 1.0,
                   learning_enabled: bool = True,
                   key: jr.PRNGKey = None) -> Tuple[Union[List[jnp.ndarray], None], List[jnp.ndarray]]:
    """Initialize Dirichlet parameters using template shapes and return expectations.

    Args:
        template: List of arrays used to determine Dirichlet parameter shapes
        init: Method to use for initialization:
            - "uniform": uniform concentration parameters
            - "like": scale the template directly
            - "random": random uniform values
        scale: Scaling factor for parameters (default=1.0)
        learning_enabled: Whether to return parameters (True) or None (False)
        key: JAX random key for random init

    Returns:
        Tuple containing:
        - Parameters if learning_enabled else None
        - Expected values of Dirichlet distribution if learning_enabled else template
    """
    if not learning_enabled:
        return None, template #TODO: make sure template is a list of categorical distributions, by checking non-positive normalised entries

    if init == "uniform":
        concentration = _dirichlet_uniform(template, scale)   
    elif init == "like":
        concentration = _dirichlet_like(template, scale)
    elif init == "random":
        concentration = _dirichlet_random(template, scale, key)
    else:
        raise ValueError(f"Unknown initialization method: {init}. Must be one of: uniform, like, random")
    
    return concentration, [dirichlet_expectation(arr) for arr in concentration]


def _dirichlet_uniform(template: List[jnp.ndarray], scale: float = 1.0) -> List[jnp.ndarray]:
    """Initialize uniform Dirichlet parameters using template shapes and alpha_i = scale for all i.

    Args:
        template: List of arrays used to determine shapes
        scale: Value for uniform parameters (default=1.0)

    Returns:
        List of uniform Dirichlet concentration parameters scaled by scale
    """
    shapes = [arr.shape for arr in template]
    
    # Create scaled uniform priors (these are the concentration parameters of the Dirichlet distribution)
    return list_array_scaled(shapes, scale)


def _dirichlet_like(template: List[jnp.ndarray], scale: float = 1.0) -> List[jnp.ndarray]:
    """Initialize Dirichlet parameters by scaling the template.
    
    Args:
        template: List of arrays to scale (must be strictly positive)
        scale: Scaling factor (default=1.0)
    
    Returns:
        List of scaled template arrays
    """
    # Check that template has strictly positive entries
    non_positive = []
    for i, arr in enumerate(template):
        if (arr <= 0.0).any():
            non_positive.append(i)
    
    if non_positive:
        raise ValueError(f"Arrays at indices {non_positive} contain non-positive entries")
    
    # Scale the template to get concentration parameters
    return [scale * jnp.array(arr) for arr in template]


def _dirichlet_random(template: List[jnp.ndarray], scale: float = 1.0, key: jr.PRNGKey = None) -> List[jnp.ndarray]:
    """Initialize random Dirichlet parameters using iid uniform distributions on interval [0, scale].

    Args:
        template: List of arrays used to determine shapes
        scale: Scaling factor (default=1.0)
        key: JAX random key (required)

    Returns:
        List of scaled random parameters
    """
    if key is None:
        raise ValueError("Random key must be provided")

    shapes = [arr.shape for arr in template] # Get shapes from template
    keys = jr.split(key, len(shapes)) # Generate a random key for each shape
    
    return [scale * jr.uniform(k, shape=shape) for k, shape in zip(keys, shapes)]
