from jax.lax import scan
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple, List, Union, Dict, Any
from jax import tree_util, nn, jit, vmap, lax
from jax.scipy.special import xlogy
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import ArrayLike
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse
from pymdp.utils import flatten_multi_trial_tensor_list

MINVAL = jnp.finfo(float).eps

def stable_xlogx(x):
    return xlogy(x, jnp.clip(x, MINVAL))

def stable_entropy(x):
    return - stable_xlogx(x).sum()

def stable_cross_entropy(x, y):
    return - xlogy(x, jnp.clip(y, MINVAL)).sum()

def log_stable(x):
    return jnp.log(jnp.clip(x, min=MINVAL))


@multimethod
@partial(jit, static_argnames=["keep_dims"])
def factor_dot(M: ArrayLike, xs: list[ArrayLike], keep_dims: Optional[tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.
    Parameters
    ----------
    - `qs` [list of 1D numpy.ndarray] - list of jnp.ndarrays

    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return factor_dot_flex(M, xs, dims, keep_dims=keep_dims)


@multimethod
def factor_dot(M: JAXSparse, xs: List[ArrayLike], keep_dims: Optional[Tuple[int]] = None):
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return spm_dot_sparse(M, xs, dims, keep_dims=keep_dims)


def spm_dot_sparse(
    X: JAXSparse, x: List[ArrayLike], dims: Optional[List[Tuple[int]]], keep_dims: Optional[List[Tuple[int]]]
):
    if dims is None:
        dims = (jnp.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    dims = jnp.array(dims).flatten()

    if keep_dims is not None:
        for d in keep_dims:
            if d in dims:
                dims = jnp.delete(dims, jnp.argwhere(dims == d))

    for d in range(len(x)):
        s = jnp.ones(jnp.ndim(X), dtype=int)
        s = s.at[dims[d]].set(jnp.shape(x[d])[0])
        X = X * x[d].reshape(tuple(s))

    sparse_sum = sparse.sparsify(jnp.sum)
    Y = sparse_sum(X, axis=tuple(dims))
    return Y


@partial(jit, static_argnames=["dims", "keep_dims"])
def factor_dot_flex(M, xs, dims: List[Tuple[int]], keep_dims: Optional[Tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.

    Parameters
    ----------
    - `M` [numpy.ndarray] - tensor
    - 'xs' [list of numpyr.ndarray] - list of tensors
    - 'dims' [list of tuples] - list of dimensions of xs tensors in tensor M
    - 'keep_dims' [tuple] - tuple of integers denoting dimesions to keep
    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    all_dims = tuple(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    args += [keep_dims]
    return contract(*args, backend="jax")


def get_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Return observation likelihood for a single observation modality m
    
    Parameters
    ----------
    o_m : Array
        If distr_obs=True: distribution over observations, shape (num_obs)
        If distr_obs=False: observation index 
    A_m : Array
        Likelihood mapping, shape (batch, num_obs, num_states)
    distr_obs : bool
        Whether observations are distributions (True) or indices (False)
    
    Returns
    -------
    likelihood : Array
        Likelihood of observation under each hidden state, shape (batch, num_states)
    """
    if distr_obs:
        #TODO: check if this is correct in the batched version
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        # Index observation dimension while preserving batch
        likelihood = A_m[:, o_m]

    return likelihood

def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Compute observation log-likelihood for a single modality"""
    return log_stable(get_likelihood_single_modality(o_m, A_m, distr_obs=distr_obs))


def compute_log_likelihood(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities"""
    result = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    log_likelihood = jnp.sum(jnp.stack(result), axis=0) #this line sums all elements of result (list of arrays) ie sums log likelihoods across all modalities

    return log_likelihood


def compute_log_likelihood_per_modality(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities, and return them per modality"""
    ll_all = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)

    return ll_all


def compute_accuracy(qs, obs, A, distr_obs=True):
    """Compute the accuracy portion of the variational free energy (expected log likelihood under the variational posterior)
    distr_obs : boolean, True if the observations are a distribution (eg one hot vector), False if they are the observation index"""

    log_likelihood = compute_log_likelihood(obs, A, distr_obs=distr_obs)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()

def compute_accuracy_with_A_dependencies(qs: List[jnp.ndarray], 
    obs: List[Union[int, jnp.ndarray]], 
    A: List[jnp.ndarray], 
    A_deps: List[List[int]], 
    distr_obs: bool = True
) -> float:
    """Compute the accuracy portion of the variational free energy under (possibly) non-full A dependencies"""

    ll_per_modality = compute_log_likelihood_per_modality(obs, A, distr_obs=distr_obs)

    for mod_idx in range(len(A)): #for each modality
        dep = A_deps[mod_idx][0] #select first dependency
        x = qs[dep]
        for dep in A_deps[mod_idx][1:]: #for each additional dependency
            x = jnp.expand_dims(x, -1) * qs[dep]
        ll_per_modality[mod_idx] = ll_per_modality[mod_idx] * x #multiply log likelihood by the probability of the latent states it depends on

    accuracy = sum(jnp.sum(arr) for arr in ll_per_modality) #sums accross modalities
    return accuracy


def compute_complexity(qs, prior):
    """
    Computes the complexity term of the variational free energy:
    Takes in list of qs and list of (empirical) priors, returns sum of KL(q||p)
    """
    complexity = 0.0
    for q, p in zip(qs, prior):
        H_q = stable_entropy(q)
        H_qp = stable_cross_entropy(q, p)
        complexity += -H_q + H_qp
    return complexity


def compute_free_energy(qs, prior, obs, A, A_deps=None, distr_obs=True):
    """
    Calculate variational free energy by breaking its computation down into three steps:
    1. computation of the complexity term: -H[Q(s)] + H_{Q(s)}[-lnP(s)]
    2. computation of the accuracy term: E_{Q(s)}[lnP(o|s)]
    Then return 1. minus 2.
    distr_obs : boolean, True if the observations are a distribution (eg one hot vector), False if they are the observation index
    """
    if A_deps is None:
        accuracy = compute_accuracy(qs, obs, A, distr_obs=distr_obs)
    else:
        accuracy = compute_accuracy_with_A_dependencies(qs, obs, A, A_deps, distr_obs=distr_obs)
    vfe = compute_complexity(qs, prior) - accuracy
    return vfe


def compute_prediction_errors(info):
    """
    Compute various prediction error metrics from rollout info
    Designed to work with output of rollout function under fpi inference algorithm
    """
    from pymdp.envs.rollout import flatten_multi_trial_info, is_multi_trial

    #Flatten the rollout info if multi-trial
    is_multi, _ = is_multi_trial(info)
    if is_multi: flat_info = flatten_multi_trial_info(info)
    else: flat_info = info

    # Get variables from rollout info
    observations = flat_info["observation"]  #list of arrays (one per modality) shape: (T+1, batch_size, obs_dim)
    beliefs = flat_info["qs"]  # list of arrays (one per factor) shape: (T+1, batch_size, 1, num_states)
    empirical_priors = flat_info["empirical_prior"]  # list of arrays (one per factor) shape: (T+1, batch_size, num_states)

    # Get A matrix history and dependencies (flatten if necessary; recall these are not flattened yet)
    A_hist = flatten_multi_trial_tensor_list(info["agent"].A, is_multi) # list of arrays (one per modality) shape: (T+1, batch_size, num_obs, num_states)
    A_deps = info["agent"].A_dependencies # list of lists (one per modality)

    # Initialize array to store free energy for each timestep
    num_timesteps = observations[0].shape[0]

    # Define the scan function that extracts prediction error statistics for one timestep
    def scan_fn(carry, t):
        # Get current variables
        prior_t = [p[t] for p in empirical_priors]  # Current prior (list of arrays)
        obs_t = [jnp.array(o[t].squeeze(), dtype=jnp.int32) for o in observations]  # Current observation (list of arrays)
        qs_t = [q[t] for q in beliefs]  # Current beliefs (list of arrays)
        A_t = [A_hist_mod[t] for A_hist_mod in A_hist] # Current A matrix (list of arrays)
        
        # Compute prediction error and components
        pe_t = compute_free_energy(qs_t, prior_t, obs_t, A_t, A_deps, distr_obs=False)
        negacc_t = -compute_accuracy_with_A_dependencies(qs_t, obs_t, A_t, A_deps, distr_obs=False)
        comp_t = compute_complexity(qs_t, prior_t)

        # For multi-factor environments, compute mean of L2 norms across all factors
        factor_l2_complexity = jnp.stack([jnp.linalg.norm(q[0,0,:] - p[0,:]) for q, p in zip(qs_t, prior_t)])
        comp_l2_t = jnp.mean(jnp.array(factor_l2_complexity))
        # Original single-factor implementation:
        # comp_l2_t = jnp.linalg.norm(qs_t[0][0,0,:]- prior_t[0][0,:])

        # Return results for this timestep
        return carry, (pe_t, negacc_t, comp_t, comp_l2_t)

    # Run scan over all timesteps
    # We use a dummy carry value (None) since we're not accumulating anything across steps
    _, (pe, negacc, comp, comp_l2) = scan(scan_fn, None, jnp.arange(num_timesteps))
    
    # Compute accumulated prediction errors
    pe_accumulated = jnp.cumsum(pe)
    
    return {
        'pred_error': pe,
        'neg_accuracy': negacc,
        'complexity': comp,
        'complexity_l2': comp_l2,
        'pe_accumulated': pe_accumulated
    }

def compute_preferences(info):
    """This is for single trial data"""

    # Number of modalities
    num_modalities = len(info['observation'])
    
    # Single-trial data: observations shape is (num_timesteps, batch_size, 1)
    num_timesteps = info['observation'][0].shape[0]
    batch_size = info['observation'][0].shape[1]

    # Initialize results with appropriate shapes
    modality_preferences = [jnp.zeros((num_timesteps, batch_size)) for _ in range(num_modalities)]

    for m in range(num_modalities):
        # Get observations for this modality
        obs_m = info['observation'][m]  # Shape: (num_timesteps, batch_size, 1)
        
        # Process each timestep and batch element
        for t in range(num_timesteps):
            for b in range(batch_size):
                # Get observation index
                obs_idx = int(obs_m[t, b, 0])
                
                # Get preferences for this modality
                C_value = float(info['agent'].C[m][t, b, obs_idx])
                
                # Store preference
                modality_preferences[m] = modality_preferences[m].at[t, b].set(C_value)

    
    # Accumulate modality preferences into combined preferences (sum because preferences are in log space)
    combined_preferences = sum(modality_preferences)
    
    # Compute cumulative preferences
    cumulative_preferences = jnp.cumsum(combined_preferences, axis=0)

    # Return results
    return {
        "modality_preferences": modality_preferences,
        "combined_preferences": combined_preferences,
        "cumulative_preferences": cumulative_preferences
    }


def compute_preferences_multitrial(info):
    """Compute preferences for multi-trial data using lax.scan. Returns results with an added trial dimension."""

    from pymdp.envs.rollout import is_multi_trial

    _, num_trials = is_multi_trial(info)
    num_modalities = len(info['observation'])

    # Define scan function over trials
    def scan_fn(carry, trial_idx):
        # Extract single-trial info
        trial_observations = [info['observation'][m][trial_idx] for m in range(num_modalities)]
        trial_C = [info['agent'].C[m][trial_idx] for m in range(num_modalities)]
        trial_prefs = _compute_preferences(trial_observations, trial_C)
        return carry, trial_prefs

    # Scan over all trials
    _, (modality_all, combined_all, cumulative_all) = scan(scan_fn, None, jnp.arange(num_trials))

    # Return with trial dimension as leading axis
    return {
        "modality_preferences": modality_all,         # shape: (num_trials, num_modalities, T, batch)
        "combined_preferences": combined_all,         # shape: (num_trials, T, batch)
        "cumulative_preferences": cumulative_all      # shape: (num_trials, T, batch)
    }

def _compute_preferences(observations, C):
    """This is for single trial data"""

    # Number of modalities
    num_modalities = len(observations)
    
    # Single-trial data: observations shape is (num_timesteps, batch_size, 1)
    num_timesteps = observations[0].shape[0]
    batch_size = observations[0].shape[1]

    # Initialize results with appropriate shapes
    modality_preferences = [jnp.zeros((num_timesteps, batch_size)) for _ in range(num_modalities)]

    for m in range(num_modalities):
        # Get observations for this modality
        obs_m = observations[m]  # Shape: (num_timesteps, batch_size, 1)
        
        # Process each timestep and batch element
        for t in range(num_timesteps):
            for b in range(batch_size):
                # Get observation index
                obs_idx = obs_m[t, b, 0].astype(int)
                
                # Get preferences for this modality
                C_value = C[m][t, b, obs_idx].astype(float)
                
                # Store preference
                modality_preferences[m] = modality_preferences[m].at[t, b].set(C_value)

    
    # Accumulate modality preferences into combined preferences (sum because preferences are in log space)
    combined_preferences = sum(modality_preferences)
    
    # Compute cumulative preferences
    cumulative_preferences = jnp.cumsum(combined_preferences, axis=0)

    # Return results
    return (modality_preferences, combined_preferences, cumulative_preferences)

def multidimensional_outer(arrs):
    """Compute the outer product of a list of arrays by iteratively expanding the first array and multiplying it with the next array"""

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x


def spm_wnorm(A):
    """
    Returns Expectation of logarithm of Dirichlet parameters over a set of
    Categorical distributions, stored in the columns of A.
    """
    norm = 1. / A.sum(axis=0)
    avg = 1. / (A + MINVAL)
    wA = norm - avg
    return wA


def dirichlet_expected_value(dir_arr):
    """
    Returns Expectation of Dirichlet parameters over a set of
    Categorical distributions, stored in the columns of A.
    """
    dir_arr = jnp.clip(dir_arr, min=MINVAL)
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=0, keepdims=True))
    return expected_val


def dirichlet_expectation(arr: jnp.ndarray) -> jnp.ndarray:
    """Normalize Dirichlet parameters to get expected probabilities."""
    return arr / arr.sum(axis=1, keepdims=True)


def smooth_data(data, window_size: int = None, copy: bool = True):
    """
    Apply moving average smoothing to dictionary of numerical data.
    
    This utility function handles dictionaries of data by applying convolution-based 
    smoothing to any 1D numerical arrays found within the dictionary.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary of data to smooth
    window_size : int, optional
        Size of the smoothing window. If None or <= 1, original data is returned.
    copy : bool, default=True
        Whether to create a copy of the input data or modify in-place.
        
    Returns
    -------
    Dict[str, Any]
        Smoothed data dictionary with same structure as input
    """
    # Return original data if no smoothing requested
    if window_size is None or window_size <= 1:
        return data.copy() if copy else data
    
    # Create smoothing kernel
    kernel = jnp.ones(window_size) / window_size
    
    # Create a copy if requested
    result = data.copy() if copy else data
    
    # Apply smoothing to each entry in the dictionary
    for key in result.keys():
        # Only smooth 1D arrays of numerical data
        if isinstance(result[key], (jnp.ndarray, jnp.ndarray)) and result[key].ndim == 1:
            result[key] = jnp.convolve(result[key], kernel, mode='same')
    
    return result


if __name__ == "__main__":
    obs = [0, 1, 2]
    obs_vec = [nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)

    print(res)
