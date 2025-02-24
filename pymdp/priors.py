from typing import List, Union, Tuple, Literal
import jax.numpy as jnp
import jax.random as jr
import warnings
from .utils import list_array_scaled
from .maths import dirichlet_expectation

""" Functions for setting up Dirichlet and categorical priors

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


def check_consistency(param, prior, name):
    """Check consistency between a parameter and its prior.
    
    If prior exists, check that param is the expectation of prior.
    If not, update param to be the expectation.
    
    Parameters
    ----------
    param : List[jnp.ndarray]
        List of parameter matrices
    prior : List[jnp.ndarray] or None
        List of prior matrices, if None no check is performed
    name : str
        Name of parameter for print message
        
    Returns
    -------
    List[jnp.ndarray]
        Updated parameter matrices
    """
    if prior is not None:
        expected = [dirichlet_expectation(arr) for arr in prior]
        for i, (p, exp_p) in enumerate(zip(param, expected)):
            if not jnp.allclose(p, exp_p):
                print(f"{name}[{i}] updated to match expectation of p{name}[{i}]")
                param[i] = exp_p
    return param


def create_uniform_A(num_batches: int, num_obs: List[int], num_states: List[int], A_dependencies: List[List[int]]) -> List[jnp.ndarray]:
    """Create uniform base tensors for observation (A) matrices.
    
    Parameters
    ----------
    num_batches : int
        Number of parallel batches
    num_obs : List[int]
        Number of observations for each modality
    num_states : List[int]
        Number of states for each factor
    A_dependencies : List[List[int]]
        Dependencies between observation modalities and state factors
        
    Returns
    -------
    List[jnp.ndarray]
        List of uniform A matrices for each modality
    """
    A_base = []
    for i in range(len(num_obs)):
        # Get shape based on dependencies
        shape = [num_batches, num_obs[i]]
        for state_idx in A_dependencies[i]:
            shape.append(num_states[state_idx])
        A_base.append(
            jnp.ones(shape, dtype=jnp.float32) / num_obs[i]
        )
    return A_base


def create_uniform_B(num_batches: int, num_states: List[int], num_actions: List[int], B_dependencies: List[List[int]]) -> List[jnp.ndarray]:
    """Create uniform base tensors for transition (B) matrices.
    
    Parameters
    ----------
    num_batches : int
        Number of parallel batches
    num_states : List[int]
        Number of states for each factor
    num_actions : List[int]
        Number of actions for each factor
    B_dependencies : List[List[int]]
        Dependencies between state factors
        
    Returns
    -------
    List[jnp.ndarray]
        List of uniform B matrices for each factor
    """
    B_base = []
    for i in range(len(num_states)):
        # Get shape based on dependencies
        shape = [num_batches, num_states[i]]
        for state_idx in B_dependencies[i]:
            shape.append(num_states[state_idx])
        shape.append(num_actions[i])
        B_base.append(
            jnp.ones(shape, dtype=jnp.float32) / num_states[i]
        )
    return B_base


def create_uniform_D(num_batches: int, num_states: List[int]) -> List[jnp.ndarray]:
    """Create uniform base tensors for initial state (D) distributions.
    
    Parameters
    ----------
    num_batches : int
        Number of parallel batches
    num_states : List[int]
        Number of states for each factor
        
    Returns
    -------
    List[jnp.ndarray]
        List of uniform D matrices for each factor
    """
    return [
        jnp.ones(
            (num_batches, num_states[i]), 
            dtype=jnp.float32
        ) / num_states[i] for i in range(len(num_states))
    ]
