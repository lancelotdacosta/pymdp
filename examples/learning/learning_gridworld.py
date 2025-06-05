# This is a demo of learning on the gridworld environment
# The code should be as modular as possible so that it is straightforward to change the environment
# The demo goes in steps: structure learning is only toward the end

# %% Importing necessary libraries
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.env_factory import make, EnvType
from pymdp.envs.rollout import rollout, counterfactual_rollout, multi_trial_rollout, get_info_trial
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors, compute_preferences
from pymdp.analysis import print_rollout, print_initial_state, render_rollout, plot_beliefs, plot_agent_preferences, print_parameter_learning, plot_rollout_preferences
from pymdp.analysis import plot_prediction_errors, plot_model_comparison, plot_parameter_learning
import matplotlib.pyplot as plt

# if __name__ == "__main__":
key_idx = 0 # Initialize master random key index at the start

# Initialise environment

# Set up batch size
batch_size = 1

# Initialize environment
env = make(
    EnvType.GRIDWORLD, 
    batch_size=batch_size,
    rows=2,
    cols=2
)

workspace_agent_params = env.get_default_agent_params()
workspace_agent_params.update({
    "action_selection": "stochastic",
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "learning_mode": "offline",
    "inference_algo": "fpi"
})

# ### 1. Basic Demo. 

# #Demo of active Inference with the perfect model. 

# # Set up random key
# key = jr.PRNGKey(key_idx)

# # Initialize agent's learning config
# learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

# # Create agent directly from environment with environment config C matrices
# agent, model, key = Agent.from_env(
#     env=env,
#     learning_config=learning_config,
#     key=key,
#     model_params={"T": 10},
#     agent_params=workspace_agent_params
# )

# ### Run simulation
# num_trials = 1 # Number of trials to run
# _, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

# ### Analysis of simulation results
# #print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# render_rollout(env, get_info_trial(combined_info, trial_idx=0, verbose=False), fps=2)
# # print and plot parameter learning
# # plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
# print_parameter_learning(combined_info, learning_config, env)
# #compute and plot prediction errors
# pe_analysis = compute_prediction_errors(combined_info)
# plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials,trial_lines=False)
# #compute and plot preferences for multiple trials
# preferences= compute_preferences(combined_info)
# plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

# %% ### 2. Parameter (A and B) Learning Demo
#
# Here we demonstrate how the agent can learn the likelihood (A) and transition (B) tensors through experience.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 1000},
    agent_params=workspace_agent_params,
    #uniform_D=True
)

### Run simulation
num_trials = 10 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
# #print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
render_rollout(env, get_info_trial(combined_info, trial_idx=0, verbose=False), fps=1)
# # print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=True)
print_parameter_learning(combined_info, learning_config, env)
# #compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=20, num_trials=num_trials,trial_lines=True)
# #compute and plot preferences for multiple trials
# preferences= compute_preferences(combined_info)
# plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

#%%