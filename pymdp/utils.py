#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import jax.random as jr

import io
import matplotlib.pyplot as plt

from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    Tuple,
)

Tensor = Any  # maybe jnp.ndarray, but typing seems not to be well defined for jax
Vector = List[Tensor]
Shape = Sequence[int]
ShapeList = list[Shape]


def norm_dist(dist: Tensor) -> Tensor:
    """Normalizes a Categorical probability distribution"""
    return dist / dist.sum(0)


def list_array_uniform(shape_list: ShapeList) -> Vector:
    """
    Creates a list of jax arrays representing uniform Categorical
    distributions with shapes given by shape_list[i]. The shapes (elements of shape_list)
    can either be tuples or lists.
    """
    arr = []
    for shape in shape_list:
        arr.append(norm_dist(jnp.ones(shape)))
    return arr


def list_array_zeros(shape_list: ShapeList) -> Vector:
    """
    Creates a list of 1-D jax arrays filled with zeros, with shapes given by shape_list[i]
    """
    arr = []
    for shape in shape_list:
        arr.append(jnp.zeros(shape))
    return arr


def list_array_scaled(shape_list: ShapeList, scale: float = 1.0) -> Vector:
    """
    Creates a list of 1-D jax arrays filled with scale, with shapes given by shape_list[i]
    """
    arr = []
    for shape in shape_list:
        arr.append(scale * jnp.ones(shape))

    return arr


def get_combination_index(x, dims):
    """
    Find the index of an array of categorical values in an array of categorical dimensions

    Parameters
    ----------
    x: ``numpy.ndarray`` or ``jax.Array`` of shape `(batch_size, act_dims)`
        ``numpy.ndarray`` or ``jax.Array`` of categorical values to be converted into combination index
    dims: ``list`` of ``int``
        ``list`` of ``int`` of categorical dimensions used for conversion

    Returns
    ----------
    index: ``np.ndarray`` or `jax.Array` of shape `(batch_size)`
        ``np.ndarray`` or `jax.Array` index of the combination
    """
    assert isinstance(x, jax.Array) or isinstance(x, np.ndarray)
    assert x.shape[-1] == len(dims)

    index = 0
    product = 1
    for i in reversed(range(len(dims))):
        index += x[..., i] * product
        product *= dims[i]
    return index


def index_to_combination(index, dims):
    """
    Convert the combination index according to an array of categorical dimensions back to an array of categorical values

    Parameters
    ----------
    index: ``np.ndarray`` or `jax.Array` of shape `(batch_size)`
        ``np.ndarray`` or `jax.Array` index of the combination
    dims: ``list`` of ``int``
        ``list`` of ``int`` of categorical dimensions used for conversion

    Returns
    ----------
    x: ``numpy.ndarray`` or ``jax.Array`` of shape `(batch_size, act_dims)`
        ``numpy.ndarray`` or ``jax.Array`` of categorical values to be converted into combination index
    """
    x = []
    for base in reversed(dims):
        x.append(index % base)
        index = index // base

    x = np.flip(np.stack(x, axis=-1), axis=-1)
    return x


def fig2img(fig):
    """
    Utility function that converts a matplotlib figure to a numpy array
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, facecolor="white", format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im[:, :, :3]


def dirichlet_uniform(template_categorical: List[jnp.ndarray], scale: float = 1.0, learning_enabled: bool = True) -> Union[List[jnp.ndarray], None]:
    """Initialize uniform Dirichlet concentration parameters or return None.

    If learning is enabled, initializes uniform Dirichlet concentration parameters 
    alpha_i = scale for all i. Note that the corresponding Dirichlet distribution has template_categorical
    as its expectation.

    Args:
        template_categorical: List of categorical distributions used to determine the shape
        scale: Concentration parameter for the uniform Dirichlet distribution (default=1.0)
        learning_enabled: Whether to initialize concentration parameters (True) or return None (False)

    Returns:
        List of Dirichlet concentration parameters if learning_enabled is True, None otherwise

    #TODO: Potential todo: support per-factor scales
    """
    if not learning_enabled:
        return None

    # Get shapes from template
    shapes = [arr.shape for arr in template_categorical]
    
    # Create scaled uniform priors (these are the concentration parameters of the Dirichlet distribution)
    return list_array_scaled(shapes, scale)


def dirichlet_like(template_categorical: List[jnp.ndarray], scale: float = 1.0, learning_enabled: bool = True) -> Union[List[jnp.ndarray], None]:
    """Initialize Dirichlet concentration parameters by scaling the template or return None.

    If learning is enabled, initializes Dirichlet concentration parameters by scaling the template,
    i.e. alpha = scale * template_categorical. Note that the corresponding Dirichlet distribution
    has template_categorical as its expectation.

    Args:
        template_categorical: List of categorical distributions used as the template
        scale: Scale factor for the concentration parameters (default=1.0)
        learning_enabled: Whether to initialize concentration parameters (True) or return None (False)

    Returns:
        List of Dirichlet concentration parameters if learning_enabled is True, None otherwise

    #TODO: Potential todo: support per-factor scales
    """
    if not learning_enabled:
        return None

    # Check that template is normalized
    non_normalized = []
    for i, arr in enumerate(template_categorical):
        if not jnp.allclose(arr.sum(axis=-1), 1.0):
            non_normalized.append(i)
    
    # Print indices at which it is not normalized
    if non_normalized:
        import warnings
        warnings.warn(f"Template categorical distributions at indices {non_normalized} are not normalized")
    
    # Scale the template to get concentration parameters
    return [scale * jnp.array(arr) for arr in template_categorical]


def dirichlet_random(template_categorical: List[jnp.ndarray], 
                    scale: float = 1.0, 
                    learning_enabled: bool = True,
                    key: jr.PRNGKey = None) -> Union[List[jnp.ndarray], None]:
    """Initialize Dirichlet concentration parameters with random uniform values or return None.

    If learning is enabled, initializes Dirichlet concentration parameters by sampling uniform
    random values and scaling them. Note that unlike dirichlet_like and dirichlet_uniform,
    the corresponding Dirichlet distribution will NOT have template_categorical as its expectation.

    Args:
        template_categorical: List of categorical distributions used to determine shapes
        scale: Scale factor for the concentration parameters (default=1.0)
        learning_enabled: Whether to initialize concentration parameters (True) or return None (False)
        key: JAX random key for generating random values (required if learning_enabled=True)

    Returns:
        List of Dirichlet concentration parameters if learning_enabled is True, None otherwise

    #TODO: Potential todo: support per-factor scales
    """
    if not learning_enabled:
        return None

    if key is None:
        raise ValueError("Random key must be provided when learning is enabled")

    # Get shapes from template
    shapes = [arr.shape for arr in template_categorical]
    
    # Generate a random key for each shape
    keys = jr.split(key, len(shapes))
    
    # Create random uniform parameters for each shape
    return [scale * jr.uniform(k, shape=shape) for k, shape in zip(keys, shapes)]
