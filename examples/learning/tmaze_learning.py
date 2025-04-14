# This is a demo of learning onthe TMaze environment
# The code should be as modular as possible so that it is straightforward to change the environment
#
# Authors: Lancelot da Costa


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
from pymdp.envs.rollout import rollout, counterfactual_rollout, multi_trial_rollout, is_multi_trial,get_info_trial
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import print_rollout, print_initial_state, render_rollout, plot_beliefs, plot_preferences, analyze_rollout, print_parameter_learning
from pymdp.analysis import plot_prediction_errors, plot_model_comparison, plot_parameter_learning, print_experiment_setup
import matplotlib.pyplot as plt
from copy import deepcopy

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

# %% ### 2b. Parameter (B) Learning Demo
#
# Here we demonstrate how the agent can learn the transition (B) tensor through experience.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 2},
    agent_params={"action_selection": "stochastic"},
    #uniform_D=True
)
agent2 = deepcopy(agent)
#%%

key = jr.PRNGKey(key_idx)
# Run simulation with multiple trials
# Checked that this works! :)
num_trials = 5  # Number of trials to run
all_info = []
for trial in range(num_trials):
    print(f"\n--- Trial {trial+1}/{num_trials} ---")
    key, rollout_key = jr.split(key)
    last, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)
    print_initial_state(info)
    all_info.append(info)
    agent = last["agent"] # save agent for next trial. Don't need to do this for the environment since this is reset in the rollout function anyway.

#%%

key = jr.PRNGKey(key_idx)
# Use the multi_trial_rollout function for efficient multi-trial learning
key, rollout_key = jr.split(key)
last, combined_info = multi_trial_rollout(agent2, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=rollout_key)

#%% ================ANALYSIS WHICH IS WORKING NOW IN THE MULTI-TRIAL ROLLOUT================
print_experiment_setup(combined_info)
#%%
print_rollout(combined_info)

#%% ================ANALYSIS REMAINING================
# plot_parameter_learning(combined_info, learning_config, env)
print_parameter_learning(combined_info, learning_config, env, verbose=False)

#%%
# print_rollout(all_info[0], env), print_rollout(all_info[1], env)
print_parameter_learning(all_info[0], learning_config, env, verbose=False)
print_parameter_learning(all_info[1], learning_config, env, verbose=False)

#%%
# Analysis after all trials are done
pe_analysis = compute_prediction_errors(info)  # Analyze the final trial

# # Analyze and visualize results
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None)

#%%
plot_parameter_learning(info, learning_config, env)
#agent seems to be learning B matrix right under top left reward but not under top right reward. Need to investigate this
print_parameter_learning(info, learning_config, env, verbose=True)

#%%
plot_preferences(agent, env)
render_rollout(env, info, fps=10)
plot_beliefs(info, env)


#%% For just A learning complexity is infinite

# prior_t = [p[1] for p in info["empirical_prior"]]  # Current prior (list of arrays)
# qs_t = [q[1] for q in info["qs"]]
# action_t = info["action"][1,]
