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


def flatten_multi_trial_tensor(tensor, multi_trials=True):
    """
    Helper function to flatten multi-trial tensor data into a single time series.
    
    For multi-trial data, tensors have shape [num_trials, timesteps_per_trial, ...].
    This function reshapes the tensor to [num_trials*timesteps_per_trial, ...],
    effectively treating the entire multi-trial history as one continuous timeline.
    
    Parameters
    ----------
    tensor : ndarray or jax.Array
        Multi-dimensional array with shape [trials, timesteps, ...] 
        
    Returns
    -------
    ndarray or jax.Array
        Flattened array with shape [trials*timesteps, ...]
    """
    if multi_trials:
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    else:
        return tensor


def are_equal_dicts_jnp_arrays(dict1, dict2, verbose=True, atol=0, rtol=0):
    """
    Compares two dictionaries of jnp arrays for equality.
    
    This function checks if two dictionaries have the same keys, and for each key,
    checks if the corresponding arrays have the same shape and content.
    
    Parameters
    ----------
    dict1 : dict
        First dictionary to compare. Values should be jnp arrays.
    dict2 : dict
        Second dictionary to compare. Values should be jnp arrays.
    verbose : bool, optional
        Whether to print detailed error messages. Default is True.
    tol : float, optional
        Tolerance for floating point comparison. Default is 0.
    Returns
    -------
    bool
        True if dictionaries have identical keys and array values, False otherwise.
    
    Examples
    --------
    >>> d1 = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5])}
    >>> d2 = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5])}
    >>> are_equal_dicts_jnp_arrays(d1, d2)
    True
    
    >>> d3 = {'a': jnp.array([1, 2, 4]), 'b': jnp.array([4, 5])}
    >>> are_equal_dicts_jnp_arrays(d1, d3)
    Contents for key 'a' don't match
    False
    """
    # Check if they have the same keys
    if dict1.keys() != dict2.keys():
        if verbose:
            missing_in_1 = set(dict2.keys()) - set(dict1.keys())
            missing_in_2 = set(dict1.keys()) - set(dict2.keys())
            print(f"Keys don't match: missing in dict1 {missing_in_1}, missing in dict2 {missing_in_2}")
        return False
    
    # Compare each key-value pair individually
    for key in dict1:
        val1, val2 = dict1[key], dict2[key]

        if not isinstance(val1, jnp.ndarray) or not isinstance(val2, jnp.ndarray):
            print(f"Key '{key}' has non-jnp array values: {type(val1)} and {type(val2)}")
        else:
            # Check shapes
            if val1.shape != val2.shape:
                if verbose:
                    print(f"Shapes for key '{key}' don't match: {val1.shape} vs {val2.shape}")
                return False
        
            # Check contents
            if not jnp.allclose(val1, val2, atol=atol, rtol=rtol):
                if verbose:
                    print(f"Contents for key '{key}' don't match")
                return False

    
    return True
