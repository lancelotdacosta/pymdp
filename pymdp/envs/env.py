from typing import Optional, List, Dict
from jaxtyping import Array, PRNGKeyArray
from functools import partial

from equinox import Module, field, tree_at
from jax import vmap, random as jr, tree_util as jtu
import jax.numpy as jnp

def _float_to_int_index(x):
    # converting float to integer for array indexing while preserving the og data structure for gradient computation    
    return jnp.asarray(x, jnp.int32)

def select_probs(positions, matrix, dependency_list, actions=None):
    # creating integer indices from float state positions for the positions specified in dependency_list
    index_args = tuple(_float_to_int_index(p) for i, p in enumerate(positions) 
                      if i in dependency_list)  #TODO: implement B_action dependencies
    if actions is not None:
        index_args += (_float_to_int_index(actions),)
    return matrix[..., *index_args]


def cat_sample(key, p):
    a = jnp.arange(p.shape[-1], dtype=jnp.float32)
    if p.ndim > 1:
        choice = lambda key, p: jr.choice(key, a, p=p)
        keys = jr.split(key, len(p))
        # print(keys.shape)
        return vmap(choice)(keys, p)

    return jr.choice(key, a, p=p)


class Env(Module):
    params: Dict
    state: List[Array]
    current_obs: List[Array]
    dependencies: Dict = field(static=True)
    labels: Dict = field(static=True)

    def __init__(self, params: Dict, dependencies: Dict, labels: Dict = None):
        self.params = params
        self.dependencies = dependencies

        # Initialize state and observation arrays
        self.state = jtu.tree_map(lambda x: jnp.zeros([x.shape[0]]), self.params["D"])
        self.current_obs = jtu.tree_map(lambda x: jnp.zeros([x.shape[0], x.shape[1]]), self.params["A"])

        # If labels not provided, create default ones
        if labels is None:
            self.labels = self._initialize_default_labels()
        else:
            self.labels = labels

    def _initialize_default_labels(self):
        """Initialize default labels based on tensor shapes.
        
        This method creates default labels for state factors, observation modalities,
        and control factors based on the shapes of A, B, and D tensors.
        
        Returns
        -------
        Dict
            Dictionary containing default labels for all components
        """
        # Get number of state factors from D tensors
        num_state_factors = len(self.params["D"])
        
        # Get number of observation modalities from A tensors
        num_obs_modalities = len(self.params["A"])
        
        # For control factors, we need to infer from B's shapes
        # Each B[f] has shape (batch_size, num_states[f], *other_factors, num_controls)
        # We need to extract the control dimension size for each state factor
        control_dims = []
        for f, b_f in enumerate(self.params["B"]):
            # The last dimension is the control dimension for this factor
            # If B has shape (batch, s_f, s_0, ..., s_{f-1}, s_{f+1}, ..., s_{n-1}, a_f)
            # Then control_dim = b_f.shape[-1]
            control_dims.append(b_f.shape[-1])
        
        # Create default labels dictionary
        labels = {
            "state_factors": {},
            "observation_modalities": {},
            "control_factors": {}
        }
        
        # Infer number of states for each factor from D tensors
        num_states = [d.shape[-1] for d in self.params["D"]]
        
        # Infer number of observations for each modality from A tensors
        num_obs = [a.shape[1] for a in self.params["A"]]
        
        # Add state factor labels
        for f in range(num_state_factors):
            factor_name = f"Factor{f}"
            factor_states = [f"state{f}_{s}" for s in range(num_states[f])]
            labels["state_factors"][factor_name] = factor_states
        
        # Add observation modality labels
        for m in range(num_obs_modalities):
            modality_name = f"Modality{m}"
            modality_obs = [f"obs{m}_{o}" for o in range(num_obs[m])]
            labels["observation_modalities"][modality_name] = modality_obs
        
        # Add control factor labels
        for c in range(len(control_dims)):
            control_name = f"Control{c}"
            control_actions = [f"action{c}_{a}" for a in range(control_dims[c])]
            labels["control_factors"][control_name] = control_actions
        
        return labels

    @vmap
    def reset(self, key: PRNGKeyArray, state: Optional[List[Array]] = None):
        if state is None:
            probs = self.params["D"]
            keys = list(jr.split(key, len(probs) + 1))
            key = keys[0]
            state = jtu.tree_map(cat_sample, keys[1:], probs)

        env = tree_at(lambda x: x.state, self, state)

        new_obs = self._sample_obs(key, state)
        env = tree_at(lambda x: x.current_obs, env, new_obs)
        return new_obs, env

    def render(self, mode="human"):
        """

        Returns
        ----
        if mode == "human":
            returns None, renders the environment using MPL inside the function
        elif mode == "rgb_array":
            A (H, W, 3) uint8 jax.numpy array, with values between 0 and 255
        """
        pass

    @vmap 
    def step(self, rng_key: PRNGKeyArray, actions: Optional[Array] = None):
        """Execute one time step within the environment.

        This function implements the core POMDP dynamics by:
        1. Transitioning to a new state based on the current state and action (if provided)
        2. Generating new observations based on the new state

        The state transition uses the B (transition) matrix and B dependencies from the environment's parameters,
        while observations are generated using the A (observation) matrix.

        Args:
            rng_key (PRNGKeyArray): JAX random key for stochastic operations
            actions (Optional[Array], optional): List of actions to take. If None, state remains unchanged.
                Each action corresponds to a control factor in the environment's state space.

        Returns:
            Tuple[List[Array], Env]: A tuple containing:
                - new_obs: List of new observations, one for each observation modality
                - env: Updated environment instance with new state and observations
        """
        # return a list of random observations and states
        key_state, key_obs = jr.split(rng_key)
        state = self.state
        if actions is not None:
            actions = list(actions)
            _select_probs = partial(select_probs, state)
            state_probs = jtu.tree_map(_select_probs, self.params["B"], self.dependencies["B"], actions)

            keys = list(jr.split(key_state, len(state_probs)))
            new_state = jtu.tree_map(cat_sample, keys, state_probs)
        else:
            new_state = state

        new_obs = self._sample_obs(key_obs, new_state)

        env = tree_at(lambda x: (x.state), self, new_state)
        env = tree_at(lambda x: x.current_obs, env, new_obs)
        return new_obs, env

    def _sample_obs(self, key, state):
        _select_probs = partial(select_probs, state)
        obs_probs = jtu.tree_map(_select_probs, self.params["A"], self.dependencies["A"])

        keys = list(jr.split(key, len(obs_probs)))
        new_obs = jtu.tree_map(cat_sample, keys, obs_probs)
        new_obs = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), new_obs)
        return new_obs

    def get_labels(self):
        """Get the labels dictionary for this environment.
        
        Returns
        -------
        Dict
            Dictionary containing labels for state factors, observation modalities,
            and control factors
        """
        return self.labels