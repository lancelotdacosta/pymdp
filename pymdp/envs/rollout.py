from typing import Dict, Tuple
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax
from pymdp.utils import flatten_multi_trial_tensor, flatten_multi_trial_tensor_list
import warnings
import equinox as eqx

from pymdp.agent import Agent
from pymdp.envs.env import Env

#TODO: introduced many changes to rollout beyond v1alpha branch including a reordering of inferences. May need to put old rollout back with a legacy flag to ensure backward compatibility.
def rollout(agent: Agent, env: Env, num_timesteps: int, rng_key: jr.PRNGKey, policy_search=None) -> Tuple[Dict, Dict, Env]:
    """
    Rollout an agent in an environment for a number of timesteps following the active inference cycle.

    Parameters
    ----------
    agent: active inference agent 
    env: environment that can step forward and return observations
    num_timesteps: how many timesteps to simulate
    rng_key: random key for sampling
    policy_search: optional custom policy inference function such as sophisticated inference

    Inner workings
    ----------
    Initialization (t=0):
    1. Initialize policy distribution and action (zeroed - just for shape matching)
    2. Initialize prior beliefs (using agent.D)
    3. Get initial observation from environment reset
    4. Compute initial posterior beliefs after seeing initial observation

    For each timestep t (1 to T), the active inference cycle proceeds as:
    1. Policy inference: compute policy distribution using expected free energy
    2. Action selection: sample action from policy distribution
    3. Prediction: compute empirical prior for next state using action and current beliefs
    4. Environment interaction: execute action and get new observation
    5. State inference: update beliefs using new observation and prediction using variational inference
    6. (Optional) Parameter learning: update A, B, and/or D if learning is enabled

    Returns
    ----------
    last: ``dict``
        Dictionary from the last timestep containing final action, observation, beliefs, etc.
    info: ``dict``
        Dictionary containing information about the rollout with arrays of shape (T+1, batch_size, ...):
        - qpi[t]: policy distribution at time t based on beliefs[t-1]
        - action[t]: action chosen at time t
        - empirical_prior[t]: predicted next state after taking action[t]
        - observation[t]: result of taking action[t]
        - qs[t]: posterior beliefs after seeing observation[t]
    env: ``Env``
        Environment state after the rollout
    """

    # get the batch_size of the agent
    batch_size = agent.batch_size

    # default policy search just uses standard active inference policy selection
    if policy_search is None:
        def default_policy_search(agent, qs, rng_key):
            qpi, _ = agent.infer_policies(qs)
            return qpi, None
        policy_search = default_policy_search
    elif not callable(policy_search):
        raise TypeError("policy_search must be callable or None")

    # get initial policy and action distribution - unused and meaningless - just for shape matching
    D_reshaped = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D) #reshaping to match the shape of qs
    qpi_0, _ = agent.infer_policies(D_reshaped)
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    action_0 = agent.sample_action(qpi_0, rng_key=keys[1:])
    action_0 *= 0 # zero out initial action as no action taken yet

    # get initial prior belief using D
    p0 = agent.D
   
    # initialise first observation from environment
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    observation_0, env = env.reset(keys[1:])

    # compute and store posterior state beliefs after initial observation (used for D learning)
    qs_0 = agent.infer_states(
        observations=observation_0,
        empirical_prior=p0, 
    )

    # set up initial state to carry through timesteps
    initial_carry = {
        "qs": qs_0,
        "action_t": action_0,
        "observation_t": observation_0,
        "empirical_prior": p0,
        "env": env,
        "agent": agent,
        "rng_key": rng_key,
        "qs_0": qs_0
    }

    def step_fn(carry, x):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action_t = carry["action_t"]
        observation_t = carry["observation_t"]
        qs_prev = carry["qs"]
        empirical_prior = carry["empirical_prior"]
        env = carry["env"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]
        qs_0 = carry["qs_0"]

        # compute policy posterior
        rng_key, key = jr.split(rng_key)
        qpi, _ = policy_search(agent, qs_prev, key) # compute policy posterior using EFE - uses C to consider preferred outcomes

        # sample action from policy distribution
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_t = agent.sample_action(qpi, rng_key=keys[1:])

        # update empirical prior about next state
        empirical_prior, _ = agent.update_empirical_prior(action_t, qs_prev) # return empirical_prior. The empirical prior is D for mmp, vmp and it is the last posterior times transition matrix given the last action for fpi, ovf.  

        # step environment forward with chosen action and get new observation
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_t, env = env.step(rng_key=keys[1:], actions=action_t) 

        # perform state inference using variational inference (FPI) - uses A matrix to map between hidden states and observations
        qs = agent.infer_states(
            observations=observation_t, # This is observation_0 in first step
            empirical_prior=empirical_prior, # This is agent.D in first step
        )
        
        # --- Online learning update (A and/or B and/or D) ---
        if agent.learning_mode == "online":
            agent = _update_agent_parameters(agent, qs, qs_prev, observation_t, action_t, qs_0)

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key
        carry = {
            "action_t": action_t,
            "observation_t": observation_t,
            "qs": qs,
            "empirical_prior": empirical_prior,
            "env": env,
            "agent": agent,
            "rng_key": rng_key,
            "qs_0": qs_0
        }
        info = {
            "qpi": qpi,
            "qs": qs,  # Store full belief state
            "env": env,
            "agent": agent,
            "observation": observation_t,
            "action": action_t,
            "empirical_prior": empirical_prior  # Store prior for free energy computation
        }

        return carry, info

    # run the active inference loop for num_timesteps using jax.lax.scan (jax version of for loop)
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps))

    # prepare initial info to concatenate with trajectory
    initial_info = {
        "action": jnp.expand_dims(action_0, 0),
        "observation": [jnp.expand_dims(o, 0) for o in observation_0],  
        "qs": jtu.tree_map(lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim))), qs_0), 
        "qpi": jnp.expand_dims(qpi_0, 0),  
        "env": env,
        "agent": agent,
        "empirical_prior": p0  # Initial prior is just D
    }

    # combine initial info with trajectory info
    info = jtu.tree_map(_concat_or_pass, initial_info, info) #TODO: there is a bug for batch_size > 1

    # ------------------------------------------------------------------
    # Offline learning update (single batch update after rollout)
    # ------------------------------------------------------------------
    if agent.learning_mode == "offline":
        # Perform a one-shot parameter update after the full trajectory.
        # In addition to the final updated agent, we also build a *history*
        # agent whose parameter tensors have a leading time dimension so that
        # they remain compatible with analysis utilities that expect that
        # shape (consistent with the online-learning case).
        agent_updated, agent_history = _offline_parameter_learning(agent, info)

        # Store objects in the appropriate output containers
        last["agent"] = agent_updated      # final agent after learning
        info["agent"] = agent_history      # time-stacked parameters for analysis

    return last, info, env


def _update_agent_parameters(agent, qs, qs_prev, observation_t, action_t, qs_0):
    """
    Update agent parameters based on current beliefs and observations.
    
    This helper function handles the parameter updating logic for active inference agents,
    preparing the appropriate belief and action formats for the different learning cases.
    
    Parameters
    ----------
    agent : Agent
        The agent whose parameters should be updated
    qs : list of arrays
        Current posterior beliefs about hidden states
    qs_prev : list of arrays
        Previous posterior beliefs about hidden states
    observation_t : list of arrays
        Current observation across modalities
    action_t : array
        Current action
    qs_0 : list of arrays, optional
        Initial posterior beliefs (used for D learning)
        
    Returns
    -------
    Agent
        Updated agent with new parameters
    """
    if agent.learn_A or agent.learn_B or agent.learn_D:
        if agent.learn_B:
            # stacking beliefs for B learning
            beliefs_B = jtu.tree_map(lambda x, y: jnp.concatenate([x,y], axis=1), qs_prev, qs)
            # reshaping action to match the stacked beliefs
            action_B = jnp.expand_dims(action_t, 1)  # adding time dimension
        else:
            beliefs_B = None
            action_B = action_t
        
        # Update parameters
        agent = agent.infer_parameters(
            qs, 
            observation_t, 
            action_B if agent.learn_B else action_t,
            beliefs_B=beliefs_B,
            beliefs_D=qs_0
        )

    return agent


def _offline_parameter_learning(agent: Agent, info: Dict):
    """Update Dirichlet parameters once at the end of a rollout using the full
    history contained in *info*.

    The function reuses the existing ``_update_agent_parameters`` helper for each
    timestep, but **does not** touch the agent during the actual rollout. It is
    called only when ``agent.learning_mode == 'offline'``.
    """
    # Extract tensors (each includes t=0)
    actions = info["action"]  # shape (T+1, batch, control_factors)
    observations = info["observation"]  # list[modalities] each (T+1, batch, ...)
    qs_list = info["qs"]  # list[factors] each (T+1, batch, ...)

    # Initial posterior (t=0) used for D-learning
    qs_0 = [q[0] for q in qs_list]

    num_steps = actions.shape[0]  # includes t = 0

    # ------------------------------------------------------------------
    # 1) Run the batched parameter-learning update to get the FINAL agent
    # ------------------------------------------------------------------

    def step_fn(carry_agent, t):
        qs_prev = [q[t-1] for q in qs_list]
        qs_curr = [q[t]   for q in qs_list]
        obs_t   = [o[t]   for o in observations]
        act_t   = actions[t]

        carry_agent = _update_agent_parameters(
            carry_agent, qs_curr, qs_prev, obs_t, act_t, qs_0
        )
        return carry_agent, carry_agent  # second output gathers full agent per step

    # Run scan and collect per-timestep agents (shape: (num_steps-1, ...))
    agent_final, agents_hist = jax.lax.scan(step_fn, agent, jnp.arange(1, num_steps)) #Learn by absorbing all data throughout the trial. 
    # agent_final, agents_hist = jax.lax.scan(step_fn, agent, jnp.arange(num_steps-1, num_steps)) #Learn by absorbing only the last time step (This might help in the absence of smoothing, because beliefs are more refined at the last time step). 

    # Prepend the *pre-learning* agent (t=0) to obtain a length-num_steps history
    def prepend(init_leaf, hist_leaf):
        # Only stack along time for array leaves produced by scan
        if isinstance(hist_leaf, jnp.ndarray):
            return jnp.concatenate([jnp.expand_dims(init_leaf, 0), hist_leaf], axis=0)
        else:
            # Non-array leaves (e.g. strings, bools) – keep init version; broadcast not needed
            return init_leaf

    agent_history = jtu.tree_map(prepend, agent, agents_hist)

    return agent_final, agent_history


def multi_trial_rollout(agent: Agent, env: Env, num_timesteps: int, num_trials: int, rng_key: jr.PRNGKey):
    """
    Run multiple trials of rollout allowing learning across trials.
    
    This function preserves the agent's learned parameters across trials,
    but resets the environment at the beginning of each trial.
    
    Parameters
    ----------
    agent : Agent
        the active inference agent
    env : Env
        the environment
    num_timesteps : int
        number of timesteps to run in each trial
    num_trials : int
        number of trials to run
    rng_key : PRNGKey
        random key for the simulation
        
    Returns
    -------
    last_carry : dict 
        dictionary containing the final state of the simulation
    all_info : dict
        nested dictionary containing the trajectory information from all trials
    env : Env
        the final state of the environment
    """
    
    # Define function for a single trial
    def rollout_trial(carry, _):
        agent, rng_key = carry #env is carried over but actually reset within rollout at each trial
        
        # Split key for this trial
        rng_key, rollout_key = jr.split(rng_key)
        
        # Run a single rollout
        last, info, _ = rollout(agent, env, num_timesteps, rollout_key)
        
        # Get updated agent for next trial
        agent = last["agent"]
        
        # Return updated state and info for this trial
        return (agent, rng_key), info
    
    # Initialize carry state for scan
    init_carry = (agent, rng_key)
    
    # Run all trials using scan
    (final_agent, final_key), all_trial_info = jax.lax.scan(
        rollout_trial,
        init_carry,
        jnp.arange(num_trials)
    )

    return final_agent, final_key, all_trial_info

def is_multi_trial(info):
    """Check if info contains data from multiple trials
    
    This function examines the shape of the action array to determine if the info
    dictionary contains data from multiple trials or just a single trial.
    
    In pymdp, action arrays have the following dimension structure:
    - For single-trial data (from rollout function):
      shape = (timesteps, batch_size, control_factors)
      Example: (3, 1, 2) means 3 timesteps, batch size of 1, and 2 control factors
    
    - For multi-trial data (from multi_trial_rollout function):
      shape = (num_trials, timesteps, batch_size, control_factors)
      Example: (5, 3, 1, 2) means 5 trials, 3 timesteps per trial, batch size of 1,
      and 2 control factors
    
    The key insight is that JAX's lax.scan automatically adds a leading dimension
    for the trial number when used in multi_trial_rollout.
    
    Returns
    -------
    is_multi : bool
        True if info contains multiple trials, False otherwise
    num_trials : int or None
        Number of trials if multi-trial, None otherwise
    """
    if 'action' in info:
        # If there's an extra leading dimension for trials
        if len(info['action'].shape) == 4:
            num_trials = info['action'].shape[0]
            return True, num_trials
        elif len(info['action'].shape) == 3:
            return False, None
        else:
            raise ValueError("Unexpected shape of action array")
    else:
        raise ValueError("Unsupported: Action key not found in info")

def get_info_trial(combined_info, trial_idx, verbose=True):
    """
    Extract a single trial's information from a multi-trial rollout.
    
    Parameters
    ----------
    info : dict
        Dictionary containing rollout information with keys:
        - 'observation': List of observation arrays from environment
        - 'qs': List of belief arrays for each state factor
        - 'qpi': Policy distributions
        - 'action': Selected actions
        - 'empirical_prior': Prior beliefs before observations
        
        Note on 'agent' and 'env' keys: There's a nuanced behavior with these objects:
        1. The agent/env objects themselves are the final ones from the last trial, due to how
           JAX's lax.scan handles custom Python objects that aren't registered with its pytree system.
        2. However, the tensors within agent and env objects (e.g.,A, B, D matrices) DO have a trial dimension
           and contain the full history across trials. These can be accessed as:
           - agent.A[modality_idx][trial_idx, timestep, batch_idx, ...]
           - agent.B[factor_idx][trial_idx, timestep, batch_idx, ...]
           - agent.D[factor_idx][trial_idx, timestep, batch_idx, ...]
           the same goes with all other
        
        This mixed behavior occurs because JAX automatically adds a scan dimension to arrays,
        but can't do the same with custom objects (like the agent container itself).
        
    trial_idx : int
        Index of the trial to extract
    verbose : bool, default=True
        Whether to print warnings about agent/env state mismatch
    
    Returns
    -------
    trial_info : dict
        Dictionary containing information for the specified trial
    """
    # Checks and warnings
    is_multi, num_trials = is_multi_trial(combined_info)
    if not is_multi:
        raise ValueError("Input info is not multi-trial, cannot extract trial data.")
    elif verbose and trial_idx < num_trials - 1:
        warnings.warn(f"WARNING: Extracting data for trial {trial_idx}. However, 'agent' and 'env' in the result "
                      f"will be from the final trial ({num_trials-1}), not trial {trial_idx}. "
                      f"See get_info_trial documentation for details.")
    
    if trial_idx not in range(num_trials):
        raise ValueError(f"Trial index {trial_idx} is out of bounds. Valid indices are 0 to {num_trials-1}.")
    
    # Extraction of trial data
    info_trial = {key: None for key in combined_info.keys()}
    for key in combined_info.keys():
        if key in ['action','qpi']: #these fields are jnp.ndarray with an extra dimension upfront for num_trials
            info_trial[key] = combined_info[key][trial_idx]
        elif key in ['empirical_prior', 'observation', 'qs']: #these fields are lists of jnp.ndarray per factor/modality, where each jnp.array has an extra dimension upfront for num_trials
            info_trial[key] = [combined_info[key][f][trial_idx] for f in range(len(combined_info[key]))] #here we loop over factors/modalities 
        elif key in ['agent', 'env']: #these fields are just the final objects from the last trial, but see this function's documentation for details
            info_trial[key] = combined_info[key]  
        else:
            raise ValueError(f"Key {key} not recognized in info dictionary.")
    return info_trial

def flatten_multi_trial_info(info):
    """
    Flatten multi-trial info data into a single continuous time series.
    
    This function converts data from a multi-trial rollout (with separate
    trials) into a format where all trials are concatenated into one continuous
    sequence. This is useful for analysis that treats the entire learning
    experience as a single time series.
    
    The function handles the different data types in the info dictionary:
    - For arrays like 'action' and 'qpi', it flattens the trial dimension
    - For lists of arrays like 'observation', it flattens each array separately
    - For custom Python objects ('agent', 'env'), it keeps the objects themselves.
      Note: Due to the immutability of agent and environment objects, their internal
      parameters (A, B, D) are not flattened directly.
    
    Parameters
    ----------
    info : dict
        Information dictionary from a rollout, potentially containing
        multi-trial data where the first dimension of arrays corresponds
        to different trials
    
    Returns
    -------
    dict
        Flattened information dictionary where all trial data is concatenated
        into a single time sequence. If the input was not multi-trial,
        the original dictionary is returned unchanged.
        
    Notes
    -----
    This is useful for visualizations and analyses that want to view
    learning as a continuous process rather than separate trials.
    The agent and environment objects from the original info are kept as-is
    (without attempting to flatten their internal parameters) due to their
    immutable nature.
    """
    is_multi, _ = is_multi_trial(info)

    if not is_multi:
        return info
    else:
        flat_info = {key: None for key in info.keys()}
        for key in info.keys():
            if key in ['action','qpi']: #these fields are jnp.ndarray with an extra dimension upfront for num_trials
                flat_info[key] = flatten_multi_trial_tensor(info[key])
            elif key in ['empirical_prior', 'observation', 'qs']:
                flat_info[key] = flatten_multi_trial_tensor_list(info[key])
            elif key in ['agent', 'env']:
                # Keep the original objects - they are immutable and cannot be directly modified
                flat_info[key] = info[key]    
                # NOTE: If you need to access flattened parameters, you can do it like this:
                # flat_A = [flatten_multi_trial_tensor(info[key].A[m]) for m in range(len(info[key].A))]
                # But we cannot modify the agent object directly due to immutability
            else:
                raise ValueError(f"Key {key} not recognized in info dictionary.")
        return flat_info


def counterfactual_rollout(agent, obs_sequence, action_sequence):
    """
    Perform a counterfactual rollout using a (counterfactual) agent with assumed action and observation sequences.
    
    Counterfactual rollouts are used to evaluate how well another model explains or predicts a
    the agent-environment interaction (assumed sequence of observations and actions that were taken). 
    This is useful for model comparison, where different agent models (with different internal structures or parameters)
    can be evaluated on the same observation-action sequence to determine which model better explains the data.
    
    The function simulates belief updating and learning for an agent using the provided observation
    and action sequences. It processes these sequences step by step, updating beliefs and model
    parameters (if learning is enabled) at each timestep.
    
    Parameters
    ----------
    agent : Agent
        The agent model to use for the counterfactual rollout. This agent's parameters and structure
        will be used to process the observation and action sequences.
    obs_sequence : list of arrays
        Sequence of observations from a rollout. Each item in the list corresponds to one modality,
        and contains observations across all timesteps for that modality.
    action_sequence : array
        Sequence of actions from the rollout. Shape should be (num_timesteps,).
    
    Returns
    -------
    last_carry : dict
        The final state of the agent after processing all timesteps, including:
        - "qs": posterior beliefs at the final timestep
        - "empirical_prior": empirical prior at the final timestep
        - "agent": the agent with updated parameters (if learning is enabled)
        - additional state information
    info : dict
        Information about the entire rollout across all timesteps, including:
        - "observation": sequence of observations
        - "action": sequence of actions
        - "qs": posterior beliefs at each timestep
        - "empirical_prior": empirical priors at each timestep
        - "agent": the agent state at each timestep
    
    Notes
    -----
    This function can be used to:
    1. Compare different models by measuring prediction errors downstream
    2. Evaluate how well another model generalizes to given observations and actions
    3. Test hypotheses about model structure and parameter settings
    4. Analyze belief updating and learning in different agent architectures
    
    Counterfactual rollouts differ from standard rollouts in that the agent doesn't generate actions
    through its policy or interact with an environment - instead it processes pre-recorded observation
    and action sequences.
    """

    # get the batch_size of the agent
    num_timesteps = len(action_sequence)

    # get initial policy and action distribution - unused and meaningless - just for shape matching
    action_0 = action_sequence[0] # zeroth action of action sequence

    # get initial prior belief using D
    p0 = agent.D
   
    # initialise first observation from environment
    observation_0 = [o[0] for o in obs_sequence]

    # compute and store posterior state beliefs after initial observation (used for D learning)
    qs_0 = agent.infer_states(
        observations=observation_0,
        empirical_prior=p0, 
    )

    # set up initial state to carry through timesteps
    initial_carry = {
        "qs": qs_0,
        "actions": action_sequence,
        "observations": obs_sequence,
        "empirical_prior": p0,
        "agent": agent,
        "qs_0": qs_0
    }

    def step_fn(carry, t):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        qs_prev = carry["qs"]
        empirical_prior = carry["empirical_prior"]
        action_sequence = carry["actions"]
        obs_sequence = carry["observations"]
        agent = carry["agent"]
        qs_0 = carry["qs_0"]

        # Get current action
        action_t = action_sequence[t]

        # update empirical prior about next state
        empirical_prior, _ = agent.update_empirical_prior(action_t, qs_prev) # return empirical_prior. The empirical prior is D for mmp, vmp and it is the last posterior times transition matrix given the last action for fpi, ovf.  

        # get new observation
        observation_t = [o[t] for o in obs_sequence]

        # perform state inference using variational inference (FPI) - uses A matrix to map between hidden states and observations
        qs = agent.infer_states(
            observations=observation_t, # This is observation_0 in first step
            empirical_prior=empirical_prior, # This is agent.D in first step
        )
        
        # Learning parameters: A and/or B and/or D
        agent = _update_agent_parameters(agent, qs, qs_prev, observation_t, action_t, qs_0)

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key
        carry = {
            "qs": qs,
            "actions": action_sequence,
            "observations": obs_sequence,
            "empirical_prior": empirical_prior,
            "agent": agent,
            "qs_0": qs_0
        }
        info = {
            "qs": qs,  # Store full belief state
            "agent": agent,
            "observation": observation_t,
            "action": action_t,
            "empirical_prior": empirical_prior  # Store prior for free energy computation
        }

        return carry, info

    # run the active inference loop for num_timesteps using jax.lax.scan (jax version of for loop)
    last_carry, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(1,num_timesteps))

    # prepare initial info to concatenate with trajectory
    initial_info = {
        "action": action_0,
        "observation": observation_0,  
        "qs": qs_0, 
        "agent": agent,
        "empirical_prior": p0  # Initial prior is just D
    }

    # combine initial info with trajectory info
    info = jtu.tree_map(_concat_or_pass, initial_info, info) #TODO: there is a bug for batch_size > 1

    return last_carry, info


def _concat_or_pass(init, steps):
    # helper function to concatenate initial state with trajectory by dealing with different shapes and data types
    if isinstance(init, list):
        return [jnp.concatenate([i, s], axis=0) for i, s in zip(init, steps)]
    elif isinstance(init, jnp.ndarray):
        if init.ndim < steps.ndim:
            init = jnp.expand_dims(init, 0)
        elif init.shape[1:] != steps.shape[1:]: 
            init = jnp.transpose(init, (1, 0) + tuple(range(2, init.ndim)))
        return jnp.concatenate([init, steps], axis=0)
    return steps

# EXAMPLE TESTS FOR COUNTERFACTUAL ROLLOUT IN SIMPLEST DEMO THAT COULD BE USED LATER FOR A UNIT TEST FILE

# # Running tests for counterfactual rollout:
# # Test if counterfactual observations match the original sequence
# for i, (orig_obs, cf_obs) in enumerate(zip(obs_sequence, info_counterfactual['observation'])):
#     assert jnp.allclose(orig_obs, cf_obs), f"Observation {i} values don't match"
# print("✓ Observations match the original sequence")

# # Test if counterfactual actions match the original sequence
# assert jnp.allclose(info_counterfactual['action'], action_sequence), "Counterfactual actions do not match the original sequence"
# print("✓ Actions match the original sequence")

# print("\nCounterfactual rollout successfully reproduced the original observation and action sequences.")

# # Additional tests for counterfactual rollout
# print("\n--- Additional tests for counterfactual rollout ---")

# # Test 1: Check if the counterfactual has all expected fields for compute_prediction_errors
# print("Test 1: Checking if counterfactual info has all fields needed for prediction error analysis...")
# required_fields = ["observation", "action", "qs", "empirical_prior", "agent"]
# for field in required_fields:
#     assert field in info_counterfactual, f"Required field '{field}' missing from counterfactual info"
# print("✓ All required fields present")

# # Test 2: Check if beliefs are being updated properly during the rollout
# print("Test 2: Checking if beliefs are updated properly during rollout...")
# for i, qs_factor in enumerate(info_counterfactual['qs']):
#     # Check if beliefs have expected shape (time, batch_size, ...)
#     assert qs_factor.ndim >= 3, f"Belief shape for factor {i} is incorrect: {qs_factor.shape}"
    
#     # Check if we have correct number of timesteps
#     assert qs_factor.shape[0] == len(action_sequence), f"Expected {len(action_sequence)} timesteps but got {qs_factor.shape[0]}"

# print("✓ Beliefs seem to be updated properly")

# # Test 3: Check if empirical priors are being properly updated
# print("Test 3: Checking if empirical priors are updated properly...")
# for i, prior_factor in enumerate(info_counterfactual['empirical_prior']):
#     # Check shape
#     assert prior_factor.ndim >= 2, f"Empirical prior shape for factor {i} is incorrect: {prior_factor.shape}"
    
#     # Check if we have correct number of timesteps
#     assert prior_factor.shape[0] == len(action_sequence), f"Expected {len(action_sequence)} timesteps but got {prior_factor.shape[0]}"
# print("✓ Empirical priors seem to be updated properly")

# print("\nAll additional tests passed! Counterfactual rollout implementation is robust.")