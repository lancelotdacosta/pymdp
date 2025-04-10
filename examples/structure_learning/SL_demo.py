# This is a demo of structure learning on the TMaze environment
# The code should be as modular as possible so that it is straightforward to change the environment
# The demo goes in steps: structure learning is only toward the end

# %% Importing necessary libraries
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.env_factory import make, EnvType
from pymdp.envs.simplest import SimplestEnv, plot_A_learning
from pymdp.envs.simplest import print_rollout as legacy_print_rollout
from pymdp.envs.simplest import plot_beliefs as legacy_plot_beliefs
from pymdp.envs.simplest import print_parameter_learning as legacy_print_parameter_learning
from pymdp.envs.rollout import rollout, counterfactual_rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import print_rollout, render_rollout, plot_beliefs, plot_preferences, analyze_rollout, print_parameter_learning
from pymdp.analysis import plot_prediction_errors, plot_model_comparison, plot_parameter_learning
import matplotlib.pyplot as plt

# if __name__ == "__main__":
key_idx = 0 # Initialize master random key index at the start

#%% Initialise environment

# Set up batch size
batch_size = 1

# Initialize environment
env = make(
    EnvType.T_MAZE, 
    batch_size=batch_size
)

#%% ### 1. Basic Demo

# Set up random key
key = jr.PRNGKey(key_idx)

# Initialize agent's learning config
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 10},
    agent_params={"action_selection": "stochastic"}
)

# Run simulation
key, rollout_key = jr.split(key)
_, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# # Analyze rollout: print and visualize results
# analyze_rollout(info, agent, env, render=True, plot=True, print=True)

# %% ### 2. Parameter (A, B) Learning Demo
#
# Here we demonstrate how the agent can learn the observation (A) and transition (B) tensors through experience.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 100},
    agent_params={"action_selection": "stochastic"}
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
_, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# # Analyze and visualize results
# analyze_rollout(info, agent, env, render=True, plot=True, print=True)
# print_parameter_learning(info, agent, learning_config, env, verbose=False)
# #agent seems to be learning B matrix right under top left reward but not under top right reward. Need to investigate this
# plot_parameter_learning(info, learning_config, env)

#%% ### 3. Initial State Distribution (D) Learning Demo
#
# Enable D learning only

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable D learning only
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=True)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 100},
    agent_params={"action_selection": "stochastic"}
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
_, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# # Analyze and visualize results
# analyze_rollout(info, agent, env, render=True, plot=True, print=True)
# print_parameter_learning(info, learning_config, env, verbose=False)
# plot_parameter_learning(info, learning_config, env)

#%% ### 4. Joint A, B, D Parameter Learning Demo
#
# Finally, we demonstrate learning of all parameters (A, B, D) simultaneously.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable all parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 1000},
    agent_params=env.get_default_agent_params()
)

# Run simulation and collect results
key, rollout_key = jr.split(key)
_, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Analyze and visualize results
# analyze_rollout(info, agent, env, render=True, plot=True, print=True)
# print_parameter_learning(info, agent, learning_config, env, verbose=False)
# plot_parameter_learning(info, learning_config, env)
pe_analysis = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis)

# %% ### 5. Model Comparison: Well-Specified vs Misspecified Model
#
# Finally, we compare learning performance between well-specified and misspecified models.
# A misspecified model has a different structure than the environment
#
# This allows us to:
# 1. Study how agents learn with incorrect assumptions about their environment
# 2. Compare prediction errors between well-specified and misspecified models
# 3. Demonstrate Bayesian model comparison in active inference

# Reinitialize random key for fair comparison
key = jr.PRNGKey(key_idx)

# Create misspecified model with more states than the environment
true_structure = env.get_structure()

misspecified_num_states = [5, 2]
misspecified_structure = true_structure.modify(
    num_states=misspecified_num_states,
    T=model.structure.T
)

# Enable all parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Initialize misspecified model and agent
misspecified_model, key = POMDPModel.from_structure(
    structure=misspecified_structure,
    learning=learning_config,
    key=key
)

agent = Agent.from_model(
    model=misspecified_model,
    C=env.get_default_C(), #works for misspecified model as it is a preference over observations, not states
    **env.get_default_agent_params()
)

# Run simulation with misspecified model
key, rollout_key = jr.split(key)
_, info, _ = rollout(agent, env, num_timesteps=misspecified_model.structure.T, rng_key=rollout_key)

# Analyze and visualize results
# plot_preferences(agent, env)
# render_rollout(env, info) #takes time!
# plot_beliefs(info, env) #BUG
# print_rollout(info, env) #BUG
# print_parameter_learning(info, learning_config, env, verbose=False)
# plot_parameter_learning(info, learning_config, env) #BUG: but this makes no sense to plot as we cannot compare it to the well-specified model


# %%
