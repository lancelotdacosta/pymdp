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
from pymdp.envs.rollout import rollout, counterfactual_rollout, multi_trial_rollout, is_multi_trial,get_info_trial, flatten_multi_trial_info
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors, compute_preferences
from pymdp.analysis import print_rollout, print_initial_state, render_rollout, plot_beliefs, plot_agent_preferences, print_parameter_learning, plot_rollout_preferences
from pymdp.analysis import plot_prediction_errors, plot_model_comparison, plot_parameter_learning, print_experiment_setup

# if __name__ == "__main__":
key_idx = 1 # Initialize master random key index at the start

#%% Initialise environment

# Set up environment parameters
batch_size = 1
reward_condition = None # 0 is reward in left arm, 1 is reward in right arm, None is random allocation
reward_probability = 1.0 # 100% chance of reward in the correct arm
punishment_probability = 1.0 # 100% chance of punishment in the other arm
cue_validity = 1.0 # 100% valid cues
dependent_outcomes = False # if True, punishment occurs as a function of reward probability (i.e., if reward probability is 0.8, then 20% punishment). If False, punishment occurs with set probability (i.e., 20% no outcome and punishment will only occur in the other (non-rewarding) arm)


# Initialize environment
env = make(
    EnvType.T_MAZE,
    batch_size=batch_size,
    reward_probability=reward_probability,
    punishment_probability=punishment_probability,
    cue_validity=cue_validity,
    reward_condition=reward_condition,
    dependent_outcomes=dependent_outcomes
)

#%% ### 2a. Parameter (A) Learning Demo

# Here we demonstrate how the agent can learn the observation (A) tensor through experience.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=False, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 10},
    agent_params={"action_selection": "stochastic", 
    "policy_len": 4, 
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "use_utility": False,
    "learning_mode": "online"},
    uniform_D=False
)

# Run simulation with multiple trials

num_trials = 2000 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

#print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# #print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
# print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=100, num_trials=num_trials,trial_lines=False)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

#%% ### 2B. Parameter (B) Learning Demo
#
# Here we demonstrate how the agent can learn the transition (B) tensor through experience.

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=False, learn_B=True, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 100},
    agent_params={"action_selection": "stochastic", 
    "policy_len": 4,
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "use_utility": False,
    "learning_mode": "offline"},
    uniform_D=False
)

# key, rollout_key = jr.split(key)
# _, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)
# print_rollout(info, batch_idx=0)
# print_parameter_learning(info, learning_config, env)

# Run simulation with multiple trials
num_trials = 2000 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

#print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
#print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=100, num_trials=num_trials)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)


#%% ### 2C. Parameter (A&B) Learning Demo
# NOT YET WORKING FINE!!!
# Here we demonstrate how the agent can learn the transition and likelihood (A&B) tensors through experience.

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
    agent_params={"action_selection": "stochastic", 
    "policy_len": 4,
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "use_utility": False,
    "learning_mode": "offline"},
    uniform_D=False
)

# Run simulation with multiple trials
num_trials = 2000 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

#print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
#print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env)
# print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=100, num_trials=num_trials)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

#%% ================FURTHER POSSIBLE ANALYSIS================
print_experiment_setup(combined_info)
print_rollout(combined_info)
print_parameter_learning(combined_info, learning_config, verbose=False)
plot_parameter_learning(combined_info, learning_config, env)
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='log', smoothing=None, num_trials=num_trials)
plot_agent_preferences(agent, env)
# plot_model_comparison
# print_initial_state
# initial_state
render_rollout(env, info, fps=10)
plot_beliefs(info, env)

