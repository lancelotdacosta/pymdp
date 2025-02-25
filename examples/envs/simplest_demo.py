#!/usr/bin/env python
# coding: utf-8

# # Simplest Environment Demo
# 
# In this script, we demonstrate a simple active inference agent in JAX solving the simplest possible environment using the `jax-pymdp` library.
# 
# The simplest environment has:
# - Two states (locations): left (0) and right (1)
# - Two observations: left (0) and right (1)
# - Two actions: go left (0) and go right (1)
# 
# The environment is fully observed (the observation likelihood matrix A is the identity matrix) and deterministic 
# (actions always lead to their corresponding states).

# ### Imports
#
# First, import `pymdp` and the modules we'll need.

# %% Importing necessary libraries
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, render_rollout, print_parameter_learning
from pymdp.envs import rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import plot_prediction_errors, plot_model_comparison
import matplotlib.pyplot as plt


# if __name__ == "__main__":
key_idx = 0 # Initialize master random key index at the start


# %% ### 1. Basic Demo
#
# This demo shows how to use the simplest environment with an active inference agent.
# The environment consists of two states (left and right) and two actions (stay and move).
# The agent can observe which state it is in perfectly.
#
# First, we'll create an instance of the simplest environment

# Set up random key
key = jr.PRNGKey(key_idx)

# Set up batch size
batch_size = 1

# Initialize environment
env = SimplestEnv(batch_size=batch_size)

# Initialize agent's learning config
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

# Initialise POMDP model from environment and learning config
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=100               #can play with this
)

# Update initial beliefs (D): Equal probability for all states
model = model.set_uniform_D()

# Set up preference (C) matrix
# C = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 1].set(1.0)]  # The agent prefers to be in the right state (state 1)
C = [jnp.zeros((batch_size, model.structure.num_obs[0]), dtype=jnp.float32)]  # All states equally preferred

# Initialize the agent based on model and other parameters
agent = Agent.from_model(
    model=model,
    C=C,
    policy_len=1,            # Plan one step ahead
    inference_algo="fpi",
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Print rollout and visualize results
plot_beliefs(info, agent)
render_rollout(env, info)  # Optionally: render_rollout(env, info, save_gif=True, filename="figures/simplest.gif")
print_rollout(info)

# %% ### 2. Parameter (A, B) Learning Demo
#
# Here we demonstrate how the agent can learn the observation (A) and transition (B) tensors through experience.

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

# Initialize POMDP model with learning config
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=100               #can play with this
)

# Set uniform initial beliefs
model = model.set_uniform_D()

# Initialize agent with parameter learning
agent = Agent.from_model(
    model=model,
    C=C,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Analyze and visualize results
print("\nRollout with A, B learning:")
print_rollout(info)
print_parameter_learning(info, learning_config)
if learning_config.learn_A:
    plot_A_learning(agent, info, env)

# Note: Joint A, B learning works well with random initialization, but not with strictly uniform initialization
# This is expected as uniform initialization provides no initial structure to learn from. Later could try noisy uniform initialization

# In[9]:


# %% ### 3. Initial State Distribution (D) Learning Demo
#
# Here we demonstrate learning of the initial state distribution (D). Note that D learning
# is limited by the fact that only the initial state belief (qs_0) is used to update D,
# and there is no retrospective updating of this belief for now (i.e. no smoothing).

# Enable D learning only
learning_config = LearningConfig(learn_D=True, learn_A=False, learn_B=False)

# Initialize POMDP model with D learning
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=5               #can play with this
)

# Initialize agent with D learning
agent = Agent.from_model(
    model=model,
    C=C,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Analyze and visualize results
print("\nRollout with D learning:")
print_rollout(info)

if learning_config.learn_D:
    print('\nParameter D learning:')
    for t in range(model.structure.T+1):
        print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])


# %% ### 4. Joint A, B, D Parameter Learning Demo
#
# Finally, we demonstrate learning of all parameters (A, B, D) simultaneously.
# This combines the previous learning scenarios into a full model learning task.

# Reinitialize random key
key = jr.PRNGKey(key_idx)

# Enable all parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Initialize POMDP model with all learning enabled
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=100               #can play with this
)

# Initialize agent with all parameter learning
agent = Agent.from_model(
    model=model,
    C=C,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Analyze and visualize results
pe_analysis = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis)

print("\nRollout with all parameter learning:")
print_rollout(info)
print_parameter_learning(info, learning_config)

if learning_config.learn_A:
    plot_A_learning(agent, info, env)

# Note: Joint learning works well for A, B, and D, but D learning remains limited by the
# lack of retrospective updating (i.e. smoothing) of initial state beliefs

# %% #Let's investigate active inference and learning under a mispecified generative model.
# Here we will investigate joint A, B, D learning and prediction error accumulation for a one layer, n latent state POMDP in the simplest environment.

# Reinitialize random key for fair comparison with the previous simulation
key = jr.PRNGKey(key_idx)

# Get structure from environment
env_structure = env.get_structure()

# Modify structure number of latent states
misspecified_num_states = 3
misspecified_structure = env_structure.modify(num_states = misspecified_num_states, T = model.structure.T) 

# Enable learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Initialize misspecified model
misspecified_model, key = POMDPModel.from_structure(
    structure=misspecified_structure,
    learning=learning_config,
    init="random",
    scale=1.0,
    key=key
)

# Initialize agent from misspecified model
agent = Agent.from_model(
    model=misspecified_model,
    C=C,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with misspecified model
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=misspecified_model.structure.T, rng_key=rollout_key)

# Analyse rollout and learning
# Print rollout
print("\nRollout with parameter learning:")
# print_rollout(info) #TODO: adapt to misspecified structure: num_states =! 2

# Print parameter learning
print_parameter_learning(info, learning_config)

# Compute and plot prediction errors
pe_analysis_misspecified = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis_misspecified)

#Compare well-specified vs misspecified model metrics
plot_model_comparison(pe_analysis, pe_analysis_misspecified, 
                     labels=('Well-specified', 'Misspecified'))

# This is great. 
# We now have modular code that can be used to do Bayesian model comparison of one layer pomdps
# in any environment where we can do without retrospective inference (ie smoothing)
# where it is ok to learn parameters at every timestep (and without smoothing)
# and where the standard fpi algorithm is enough.
# %%
