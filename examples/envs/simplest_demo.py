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

# In[1]:

# importing necessary libraries
import jax.numpy as jnp
from jax import random as jr
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, render_rollout, print_parameter_learning
from pymdp.envs import rollout
from pymdp.agent import Agent
from pymdp.priors import dirichlet_prior
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import plot_prediction_errors, plot_model_comparison
from pymdp.models.pomdp import POMDPConfig, POMDPStructure
from pymdp.learning import LearningConfig
import matplotlib.pyplot as plt


# if __name__ == "__main__":
key = jr.PRNGKey(2)  # Initialize master random key at the start

# ### 1. Initialize environment and get its parameters
#
# First, we'll create an instance of the simplest environment and get its observation (A) and transition (B) tensors.

# In[2]: Set up agent and run simulation
batch_size = 1

# Initialize environment and get config
env = SimplestEnv(batch_size=batch_size)
config = POMDPConfig.from_env(env, learning=LearningConfig(learn_A=False, learn_B=False))
structure = config.structure

# Initialize agent's generative model using environment parameters
A_gm = [a.copy() for a in env.params["A"]]
B_gm = [b.copy() for b in env.params["B"]]

# Set up preference (C) matrix
# The agent prefers to be in the right state (state 1)
# C = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 1].set(1.0)]  # Prefer right state
C = [jnp.zeros((batch_size, structure.num_obs[0]), dtype=jnp.float32)]  # All states equally preferred

# Set up initial beliefs (D)
# Start with certainty about being in the left state (matching the environment's initial state)
# D_gm = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 0].set(1.0)]  # Certain about starting in left state
D_gm = [jnp.ones((batch_size, structure.num_states[0]), dtype=jnp.float32) *0.5]  # Equal probability for all states

# Initialize the agent
agent = Agent(
    A=A_gm,
    B=B_gm,
    C=C,
    D=D_gm,
    policy_len=1,            # Plan one step ahead
    A_dependencies=structure.A_dependencies,
    B_dependencies=structure.B_dependencies,
    inference_algo="fpi",
    apply_batch=False,
    learn_A=config.learning.learn_A,
    learn_B=config.learning.learn_B
)

# Run simulation
key, rollout_key = jr.split(key)  # Split key for rollout
T = 1  # Number of timesteps to rollout
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# In[5]:
# Print rollout and visualize results
plot_beliefs(info, agent)
render_rollout(env, info)  # Optionally: render_rollout(env, info, save_gif=True, filename="figures/simplest.gif")
print_rollout(info)

# In[6]:

# ### 5. Parameter Learning Demo
#
# Now we'll demonstrate how the agent can learn the observation (A) and transition (B) tensors.

# Update config to enable A, B parameter learning
config = config.update_learning(learn_A=True, learn_B=True)

# Set up random priors over A and B
key, key_A = jr.split(key)
key, key_B = jr.split(key)
pA, A_gm = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=config.learning.learn_A, key=key_A)
pB, B_gm = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=config.learning.learn_B, key=key_B)

# In[6]:
# Initialize agent with parameter learning enabled
agent = Agent(
    A=A_gm,
             B=B_gm,
             C=C,
    D=D_gm,
             pA=pA,  # Prior over A
             pB=pB,  # Prior over B
    A_dependencies=config.structure.A_dependencies,
    B_dependencies=config.structure.B_dependencies,
    learn_A=config.learning.learn_A,
    learn_B=config.learning.learn_B,
             apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
T = 1  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# In[7]:
# Print rollout and learning results
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info, learn_A=config.learning.learn_A, learn_B=config.learning.learn_B)

# Visualize A learning
if config.learning.learn_A:
    plot_A_learning(agent, info, env)

# Results:
# Joint A, B learning works under random initialization, not under strictly uniform initialization (as expected). Later could try noisy uniform initialization

# In[9]:
# ### 6. Initial State distribution (D) Learning Demo
#
# Now we'll demonstrate learning of the initial state distribution (D).

# Update config to enable D learning
config = config.update_learning(learn_D=True,learn_A=False,learn_B=False)

# Set up random prior over D
key, key_D = jr.split(key)
pD, D_gm = dirichlet_prior(D_gm, init="like", scale=1.0, learning_enabled=config.learning.learn_D, key=key_D)

# %%
# Initialize agent with D learning enabled
agent = Agent(
    A=env.params["A"],  # Use true A
    B=env.params["B"],  # Use true B
    C=C,
    D=D_gm,
    pD=pD,
    A_dependencies=structure.A_dependencies,
    B_dependencies=structure.B_dependencies,
    learn_A=config.learning.learn_A,
    learn_B=config.learning.learn_B,
    learn_D=config.learning.learn_D,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with D learning
key, rollout_key = jr.split(key)
T = 1  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# Print rollout and learning results
print("\nRollout with D learning:")
print_rollout(info)

# Print and visualize D learning
if config.learning.learn_D:
    print('\n Parameter D learning:\n')  # True initial state distribution
    # print('\n Initial D matrix:\n', jnp.array(info["agent"].D[0])[0])  # True initial state distribution
    # print('\n Final learned D matrix:\n', jnp.array(info["agent"].D[0])[-1])  # Learned initial state distribution
    for t in range(T+1):
        print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])

# Results:
# The agent accumulates Dirichlet parameters as expected so D learning works.
# The only limitation is that there is no smoothing so that qs_0 
# (which is the only data that is used to update beliefs about D) stays constant over time.
# This is because there is no smoothing

# %% #Let's investigate joint A, B, D learning.

# Enable learning of all parameters
config = config.update_learning(learn_A=True, learn_B=True, learn_D=True)

# Set up random priors over A, B, and D
key, key_A = jr.split(key)
key, key_B = jr.split(key)
key, key_D = jr.split(key)
pA, A_gm = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=config.learning.learn_A, key=key_A)
pB, B_gm = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=config.learning.learn_B, key=key_B)
pD, D_gm = dirichlet_prior(D_gm, init="random", scale=1.0, learning_enabled=config.learning.learn_D, key=key_D)

# %%
# Initialize agent with parameter learning enabled
agent = Agent(
    A=A_gm,
    B=B_gm,
    C=C,
    D=D_gm,
    pA=pA,
    pB=pB,
    pD=pD,
    A_dependencies=config.structure.A_dependencies,
    B_dependencies=config.structure.B_dependencies,
    learn_A=config.learning.learn_A,
    learn_B=config.learning.learn_B,
    learn_D=config.learning.learn_D,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
T = 100  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

#%% Compute prediction errors
pe_analysis = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis)

# %%
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info, 
    learn_A=config.learning.learn_A,
    learn_B=config.learning.learn_B,
    learn_D=config.learning.learn_D
)

# Visualize A learning
if config.learning.learn_A:
    plot_A_learning(agent, info, env)

# Results:
# Joint A, B, D learning works as best it can under random initialization. The only thing is that the agent does not learn D well because qs_0 is really imprecise (and is not updated retrospectively because there is no smoothing) and that is the only thing the agent uses to learn D.

# %% #Let's investigate active inference and learning under a mispecified generative model.
# Here we will investigate joint A, B, D learning and prediction error accumulation for a one layer, n latent state POMDP in the simplest environment.

# Create new POMDP structure with different number of states
num_states = 5  # Can fiddle with this
misspecified_config = POMDPConfig(
    structure=POMDPStructure(
        num_obs=[2],           # Number of observations
        num_states=[num_states],  # Number of hidden states
        num_actions=[2],       # Number of actions (as a list)
        num_modalities=1,      # Number of observation modalities
        num_factors=1,         # Number of state factors
        num_batches=batch_size,  # Number of batches
        T=T                    # Number of timesteps
    ),
    learning=LearningConfig(learn_A=True, learn_B=True, learn_D=True)
)

# Create uniform tensors with new dimensions
A_gm = [jnp.ones((batch_size, misspecified_config.structure.num_obs[0], num_states), dtype=jnp.float32) / misspecified_config.structure.num_obs[0]]
B_gm = [jnp.ones((batch_size, num_states, num_states, misspecified_config.structure.num_actions[0]), dtype=jnp.float32) / num_states]
D_gm = [jnp.ones((batch_size, num_states), dtype=jnp.float32) / num_states]

# Set up random priors over A, B, and D using the misspecified tensors
key, key_A = jr.split(key)
key, key_B = jr.split(key)
key, key_D = jr.split(key)
pA, A_gm = dirichlet_prior(A_gm, init="random", scale=1.0, learning_enabled=misspecified_config.learning.learn_A, key=key_A)
pB, B_gm = dirichlet_prior(B_gm, init="random", scale=1.0, learning_enabled=misspecified_config.learning.learn_B, key=key_B)
pD, D_gm = dirichlet_prior(D_gm, init="random", scale=1.0, learning_enabled=misspecified_config.learning.learn_D, key=key_D)

# Initialize misspecified agent
agent = Agent(
    A=A_gm,
    B=B_gm,
    C=C,
    D=D_gm,
    pA=pA,
    pB=pB,
    pD=pD,
    A_dependencies=misspecified_config.structure.A_dependencies,
    B_dependencies=misspecified_config.structure.B_dependencies,
    learn_A=misspecified_config.learning.learn_A,
    learn_B=misspecified_config.learning.learn_B,
    learn_D=misspecified_config.learning.learn_D,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=misspecified_config.structure.T, rng_key=rollout_key)

# %% Analyse rollout and learning
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info,
    learn_A=misspecified_config.learning.learn_A,
    learn_B=misspecified_config.learning.learn_B,
    learn_D=misspecified_config.learning.learn_D
)

# %%
# Compute and plot prediction errors
pe_analysis_misspecified = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis_misspecified)

#%% Compare well-specified vs misspecified model metrics
plot_model_comparison(pe_analysis, pe_analysis_misspecified, 
                     labels=('Well-specified', 'Misspecified'))

# This is great. 
# We now have modular code that can be used to do Bayesian model comparison of one layer pomdps
# in any environment where we can do without retrospective inference (ie smoothing)
# where it is ok to learn parameters at every timestep (and without smoothing)
# and where the standard fpi algorithm is enough.
#%%