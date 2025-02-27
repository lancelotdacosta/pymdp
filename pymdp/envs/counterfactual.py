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
import jax.lax as jl

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
    
    # Define the step function for jax.lax.scan
    def step_fn(carry, t_idx):
        # Unpack the carried state
        qs_prev, agent, rng_key = carry
        
        # Get action and observation for this timestep
        action_t = action_sequence[t_idx]
        observation_t = [o[t_idx+1] for o in obs_sequence]  # t_idx+1 because first obs is initial state
        
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
        
        # Create info dict for this timestep
        step_info = {
            "action": action_t,
            "observation": [jnp.expand_dims(o, 0) for o in observation_t],
            "qs": qs,
            "empirical_prior": empirical_prior,
            "agent": agent
        }
        
        # Return updated state and info
        return (qs, agent, rng_key), step_info
    
    # Initial state to carry through iterations
    initial_carry = (qs_prev, agent, rng_key)
    
    # Run the counterfactual inference loop using jax.lax.scan
    (final_qs, final_agent, final_rng_key), step_info = jl.scan(
        step_fn, initial_carry, jnp.arange(num_timesteps)
    )
    
    # Create complete info structure
    # Initial info to concatenate with trajectory
    initial_info = {
        "action": jnp.expand_dims(action_0, 0),
        "observation": [jnp.expand_dims(o, 0) for o in observation_0],  
        "qs": jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs_0),
        "empirical_prior": jtu.tree_map(lambda x: jnp.expand_dims(x, 0), p0),
        "agent": final_agent  # Just keep the final agent state
    }
    
    # Helper function to concatenate initial state with trajectory
    def concat_or_pass(init, steps):
        if isinstance(init, list) and isinstance(steps, list):
            return [jnp.concatenate([i, s], axis=0) for i, s in zip(init, steps)]
        elif isinstance(init, jnp.ndarray) and isinstance(steps, jnp.ndarray):
            if init.ndim < steps.ndim:
                init = jnp.expand_dims(init, 0)
            return jnp.concatenate([init, steps], axis=0)
        else:
            # For non-array types like the agent object
            return steps
    
    # Combine initial info with trajectory
    info = jtu.tree_map(concat_or_pass, initial_info, step_info)
    
    # Rename observation to match rollout function's output
    info["observations"] = info.pop("observation")
    
    return final_qs, info
