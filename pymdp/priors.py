from typing import List, Union, Tuple, Literal
import jax.numpy as jnp
import jax.random as jr
import warnings
from .utils import list_array_scaled


def dirichlet_prior(template_categorical: List[jnp.ndarray],
                   init: Literal["uniform", "like", "random"] = "uniform",
                   scale: float = 1.0,
                   learning_enabled: bool = True,
                   key: jr.PRNGKey = None) -> Union[List[jnp.ndarray], None]:
    """Initialize Dirichlet concentration parameters using the specified init method or return None.

    Args:
        template_categorical: List of categorical distributions used as templates
        init: Method to use for initialization. One of:
               - "uniform": uniform concentration parameters
               - "like": scale the template directly
               - "random": random uniform values
        scale: Scale factor for the concentration parameters (default=1.0)
        learning_enabled: Whether to initialize parameters (True) or return None (False)
        key: JAX random key (required if init="random")

    Returns:
        List of Dirichlet concentration parameters if learning_enabled is True, None otherwise
    """
    if not learning_enabled:
        return None

    if init == "uniform":
        return _dirichlet_uniform(template_categorical, scale)
    elif init == "like":
        return _dirichlet_like(template_categorical, scale)
    elif init == "random":
        return _dirichlet_random(template_categorical, scale, key)
    else:
        raise ValueError(f"Unknown initialization method: {init}. Must be one of: uniform, like, random")


def _dirichlet_uniform(template_categorical: List[jnp.ndarray], scale: float = 1.0) -> List[jnp.ndarray]:
    """Initialize uniform Dirichlet concentration parameters.

    Initializes uniform Dirichlet concentration parameters alpha_i = scale for all i. 
    Note that the corresponding Dirichlet distribution has template_categorical as its expectation.

    Args:
        template_categorical: List of categorical distributions used to determine the shape
        scale: Concentration parameter for the uniform Dirichlet distribution (default=1.0)

    Returns:
        List of Dirichlet concentration parameters
    """
    # Get shapes from template
    shapes = [arr.shape for arr in template_categorical]
    
    # Create scaled uniform priors (these are the concentration parameters of the Dirichlet distribution)
    return list_array_scaled(shapes, scale)


def _dirichlet_like(template_categorical: List[jnp.ndarray], scale: float = 1.0) -> List[jnp.ndarray]:
    """Initialize Dirichlet concentration parameters by scaling the template.

    Initializes Dirichlet concentration parameters by scaling the template,
    i.e. alpha = scale * template_categorical. Note that the corresponding Dirichlet distribution
    has template_categorical as its expectation.

    Args:
        template_categorical: List of categorical distributions used as the template
        scale: Scale factor for the concentration parameters (default=1.0)

    Returns:
        List of Dirichlet concentration parameters
    """
    # Check that template is normalized
    non_normalized = []
    for i, arr in enumerate(template_categorical):
        if not jnp.allclose(arr.sum(axis=-1), 1.0):
            non_normalized.append(i)
    
    # Print indices at which it is not normalized
    if non_normalized:
        warnings.warn(f"Template categorical distributions at indices {non_normalized} are not normalized")
    
    # Scale the template to get concentration parameters
    return [scale * jnp.array(arr) for arr in template_categorical]


def _dirichlet_random(template_categorical: List[jnp.ndarray], scale: float = 1.0, key: jr.PRNGKey = None) -> List[jnp.ndarray]:
    """Initialize Dirichlet concentration parameters with random uniform values.

    Initializes Dirichlet concentration parameters by sampling uniform random values and scaling them. 
    Note that unlike _dirichlet_like and _dirichlet_uniform, the corresponding Dirichlet distribution 
    will NOT have template_categorical as its expectation.

    Args:
        template_categorical: List of categorical distributions used to determine shapes
        scale: Scale factor for the concentration parameters (default=1.0)
        key: JAX random key for generating random values (required)

    Returns:
        List of Dirichlet concentration parameters
    """
    if key is None:
        raise ValueError("Random key must be provided")

    # Get shapes from template
    shapes = [arr.shape for arr in template_categorical]
    
    # Generate a random key for each shape
    keys = jr.split(key, len(shapes))
    
    # Create random uniform parameters for each shape
    return [scale * jr.uniform(k, shape=shape) for k, shape in zip(keys, shapes)]
