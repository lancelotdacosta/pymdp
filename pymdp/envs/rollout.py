from typing import Dict, Tuple
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax

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
        
        # Learning parameters: A and/or B and/or D
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

    return last, info, env


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


def counterfactual_rollout(agent, obs_sequence, action_sequence):

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
