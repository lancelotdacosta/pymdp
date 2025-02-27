"""
Counterfactual rollout functionality for pymdp.

This module provides tools for performing counterfactual rollouts based on
sequences of observations and actions from a previous rollout.

__author__: Lancelot Da Costa
"""

from typing import Dict, List, Optional, Tuple, Any
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp.agent import Agent

def counterfactual_rollout(
    agent: Agent, 
    obs_sequence: List[List[jnp.ndarray]], 
    action_sequence: jnp.ndarray, 
    rng_key: Optional[jr.PRNGKey] = None,
    reset_agent: bool = True,
    save_belief_states: bool = True,
    save_parameters: bool = True
) -> Tuple[List[jnp.ndarray], Dict[str, Any]]:
    """
    Roll out an agent on a given observation/action sequence without environment interaction.
    
    Parameters
    ----------
    agent : Agent
        The agent to use for counterfactual inference
    obs_sequence : List[List[jnp.ndarray]]
        Sequence of observations from original rollout
    action_sequence : jnp.ndarray
        Sequence of actions from original rollout
    rng_key : jax.random.PRNGKey, optional
        Random key for stochastic operations
    reset_agent : bool, default=True
        Whether to reset the agent's state before rollout
    save_belief_states : bool, default=True
        Whether to save agent's belief states during rollout
    save_parameters : bool, default=True
        Whether to save agent's parameters during rollout
        
    Returns
    -------
    final_belief : List[jnp.ndarray]
        Final belief state of the agent
    info : dict
        Dictionary containing saved agent states and statistics
    """
    # Initialize random key if not provided
    if rng_key is None:
        rng_key = jr.PRNGKey(0)
    
    # Get basic dimensions
    batch_size = agent.batch_size
    num_timesteps = len(obs_sequence) - 1  # First observation is initial
    
    # Get initial observation
    observation_0 = [o[0] for o in obs_sequence]
    
    # Get initial action (zeroed since no action taken yet)
    action_0 = jnp.zeros_like(action_sequence[0])
    
    # Reset agent if requested
    if reset_agent:
        # get initial prior belief using D
        p0 = agent.D
        
        # compute initial posterior after seeing first observation
        qs_0 = agent.infer_states(
            observations=observation_0,
            empirical_prior=p0, 
        )
    else:
        # Use agent's current state
        p0 = agent.empirical_prior if hasattr(agent, 'empirical_prior') else agent.D
        qs_0 = agent.qs if hasattr(agent, 'qs') else agent.infer_states(
            observations=observation_0,
            empirical_prior=p0
        )
    
    # Initialize storage for tracking
    info = {
        "action": [action_0],
        "observation": [[jnp.expand_dims(o, 0) for o in observation_0]],
        "qs": [qs_0],
        "empirical_prior": [p0],
        "agent": [agent]
    }
    
    # Current belief state 
    qs_prev = qs_0
    
    # Iterate through timesteps
    for t in range(num_timesteps):
        # Get action and observation for this timestep
        action_t = action_sequence[t]
        observation_t = [o[t+1] for o in obs_sequence]  # t+1 because first obs is initial state
        
        # Update empirical prior about next state based on current belief and action
        empirical_prior, _ = agent.update_empirical_prior(action_t, qs_prev)
        
        # Perform state inference using the observation
        qs = agent.infer_states(
            observations=observation_t,
            empirical_prior=empirical_prior,
        )
        
        # Perform parameter learning if enabled
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
        
        # Save data for this timestep
        info["action"].append(action_t)
        info["observation"].append([jnp.expand_dims(o, 0) for o in observation_t])
        info["qs"].append(qs)
        info["empirical_prior"].append(empirical_prior)
        info["agent"].append(agent)
        
        # Update beliefs for next iteration
        qs_prev = qs
    
    # Convert lists to arrays for consistency with normal rollout
    # Convert info structure to match rollout's output
    # (sequences of T+1 steps with batch_size at each step)
    
    # Reshape observations to have time as first dimension
    obs_reshaped = []
    for m in range(len(info["observation"][0])):
        obs_m = [info["observation"][t][m] for t in range(len(info["observation"]))]
        obs_reshaped.append(jnp.concatenate(obs_m, axis=0))
    info["observation"] = obs_reshaped
    
    # Reshape actions to have time as first dimension
    info["action"] = jnp.stack(info["action"], axis=0)
    
    # Reshape beliefs to have time as first dimension
    qs_reshaped = []
    for f in range(len(info["qs"][0])):
        qs_f = [info["qs"][t][f] for t in range(len(info["qs"]))]
        # Need to reshape to match rollout output which has shape (T+1, batch_size, 1, num_states)
        # First ensure qs is 3D (batch_size, 1, num_states)
        qs_f_shaped = [jnp.reshape(q, (batch_size, 1, -1)) if q.ndim < 3 else q for q in qs_f]
        qs_reshaped.append(jnp.stack(qs_f_shaped, axis=0))
    info["qs"] = qs_reshaped
    
    # Reshape empirical priors to have time as first dimension
    prior_reshaped = []
    for f in range(len(info["empirical_prior"][0])):
        prior_f = [info["empirical_prior"][t][f] for t in range(len(info["empirical_prior"]))]
        prior_reshaped.append(jnp.stack(prior_f, axis=0))
    info["empirical_prior"] = prior_reshaped
    
    return qs_prev, info
