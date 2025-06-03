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
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, print_parameter_learning
from pymdp.envs.rollout import rollout, counterfactual_rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import render_rollout, plot_prediction_errors, plot_model_comparison

# if __name__ == "__main__":
key_idx = 1 # Initialize master random key index at the start

#%% Initialise environment
batch_size = 1
env = SimplestEnv(batch_size=batch_size)

# %% ### 1. Basic Demo
#
# This demo shows how to use the simplest environment with an active inference agent.
# The environment consists of two states (left and right) and two actions (stay and move).
# The agent can observe which state it is in perfectly.
#
# First, we'll create an instance of the simplest environment

# Set up random key
key = jr.PRNGKey(key_idx)

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

# %% ### 5. Model Comparison: Well-Specified vs Misspecified Model
#
# Finally, we compare learning performance between well-specified and misspecified models.
# A misspecified model has a different structure than the environment - in this case,
# we use more latent states than actually exist (eg. 3 vs 2).
#
# This allows us to:
# 1. Study how agents learn with incorrect assumptions about their environment
# 2. Compare prediction errors between well-specified and misspecified models
# 3. Demonstrate Bayesian model comparison in active inference

# Reinitialize random key for fair comparison
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create both models with same initialization conditions
true_structure = env.get_structure().modify(T=100)
misspecified_structure = true_structure.modify(num_states=3)

key_well_specified = jr.PRNGKey(key_idx)  # Same seed for both models
well_specified_model, _ = POMDPModel.from_structure(true_structure, learning_config, "random", 1.0, key_well_specified)
key_misspecified = jr.PRNGKey(key_idx)
misspecified_model, _ = POMDPModel.from_structure(misspecified_structure, learning_config, "random", 1.0, key_misspecified)

# Create agents and run side by side rollouts with same seed
agents = [
    Agent.from_model(model=well_specified_model, C=C, apply_batch=False, action_selection="stochastic"),
    Agent.from_model(model=misspecified_model, C=C, apply_batch=False, action_selection="stochastic")
]

pe_analyses = []
keys = [jr.PRNGKey(key_idx), jr.PRNGKey(key_idx)]
models = [well_specified_model, misspecified_model]
for agent, model, key in zip(agents, models, keys):
    key, rollout_key = jr.split(key)
    _, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)
    pe_analyses.append(compute_prediction_errors(info))

# Store the last rollout info for next section (misspecified model)
misspecified_rollout_info = info

#Optional, print parameter learning
# print_parameter_learning(info, learning_config)

# Compare models
plot_model_comparison(pe_analyses, labels=('Well-specified (2 states)', 'Misspecified (3 states)'))

# Note: This demo shows how we can perform Bayesian model comparison for one-layer POMDPs
# in environments where:
# 1. We can learn effectively without retrospective inference (no smoothing required)
# 2. Learning can be performed at every timestep
# 3. Standard fixed-point iteration is sufficient for inference
print("\nModel comparison complete. Well-specified model should generally show lower prediction errors.")

# %% ### 6. Counterfactual Experiment
#
# This demonstrates how to perform a counterfactual rollout with a different model structure,
# allowing us to compare which model better explains the observed data.
#
# A counterfactual rollout differs from a regular rollout in that:
# - The observations and actions are FIXED (taken from a previous rollout)
# - The agent doesn't choose actions or generate new observations
# - The agent only performs inference (belief updating) given the fixed observations
# - This lets us ask: "How well would this model have explained the same data?"
#
# In this experiment, we take the observation-action sequence from the misspecified model's
# rollout and replay it through the well-specified model to see which model better explains
# the data (lower prediction error = better explanation).

# Extract observation and action sequences from the misspecified model rollout
obs_sequence = misspecified_rollout_info['observation']
action_sequence = misspecified_rollout_info['action']

# Create an agent with the well-specified structure (2 states) for counterfactual analysis
counterfactual_agent = Agent.from_model(
    model=well_specified_model,
    C=C,
    policy_len=1,
    inference_algo="fpi",
    apply_batch=False,
    action_selection="stochastic"
)

# Perform the counterfactual rollout: replay the misspecified model's obs-action sequence
# through the well-specified model. The agent will only perform inference (no action selection).
_, info_counterfactual = counterfactual_rollout(
    counterfactual_agent,
    obs_sequence,
    action_sequence)

# Compute prediction errors for the counterfactual
pe_analysis_counterfactual = compute_prediction_errors(info_counterfactual)

# Plot prediction errors for the counterfactual
plot_prediction_errors(pe_analysis_counterfactual, title="Counterfactual Model (True 2-state Structure)")

# Compare all three: well-specified rollout vs misspecified rollout vs counterfactual rollout
plot_model_comparison([pe_analyses[0], pe_analyses[1], pe_analysis_counterfactual],
                     labels=('Well-specified', 'Misspecified', 'Counterfactual'), alpha=0.7, lw=1)

print("Counterfactual analysis complete. Compare the plots to see which model better explains the data.")
# %%
