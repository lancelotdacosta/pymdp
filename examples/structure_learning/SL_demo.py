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
from pymdp.envs.rollout import rollout, counterfactual_rollout, multi_trial_rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors, compute_preferences
from pymdp.analysis import print_rollout, print_initial_state, render_rollout, plot_beliefs, plot_agent_preferences, print_parameter_learning, plot_rollout_preferences
from pymdp.analysis import plot_prediction_errors, plot_model_comparison, plot_parameter_learning
import matplotlib.pyplot as plt

# if __name__ == "__main__":
key_idx = 0 # Initialize master random key index at the start

#%% Initialise environment

# Set up batch size
batch_size = 1

# Initialize environment
env = make(
    EnvType.SIMPLEST, 
    batch_size=batch_size
)

#%% ### 1. Basic Demo. 

#Demo of active Inference with the perfect model. 

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

### Run simulation
num_trials = 1 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
#print last trial of rollout
print_initial_state(combined_info, trial_idx= num_trials - 1)
print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# print and plot parameter learning
# plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials,trial_lines=False)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

# %% ### 2. Parameter (A and B) Learning Demo
#
# Here we demonstrate how the agent can learn the likelihood (A) and transition (B) tensors through experience.

# Set up random key
key = jr.PRNGKey(1)

# Enable A, B parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 100},
    agent_params={"action_selection": "stochastic",
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "learning_mode": "online"},
    #uniform_D=True
)

### Run simulation
num_trials = 1 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
#print last trial of rollout
print_initial_state(combined_info, trial_idx= num_trials - 1)
print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials,trial_lines=False)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

#%% ### 3. Initial State Distribution (D) Learning Demo
#
# Here we demonstrate learning of the initial state distribution (D). Note that D learning
# is limited by the fact that only the initial state belief (qs_0) is used to update D,
# and there is no retrospective updating of this belief for now (i.e. no smoothing).

# Set up random key
key = jr.PRNGKey(key_idx)

# Enable D learning only
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=True)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 10},
    agent_params={"action_selection": "stochastic"}
)

### Run simulation
num_trials = 1 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
#print last trial of rollout
print_initial_state(combined_info, trial_idx= num_trials - 1)
print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials,trial_lines=False)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

#%% ### 4. Joint A, B, D Parameter Learning Demo
#
# Finally, we demonstrate learning of all parameters (A, B, D) simultaneously.

# Set up random key
key = jr.PRNGKey(1)

# Enable all parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create agent directly from environment with environment config C matrices
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 400},
    agent_params={"action_selection": "stochastic",
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "learning_mode": "online"}
)

### Run simulation
num_trials = 1 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
#print last trial of rollout
print_initial_state(combined_info, trial_idx= num_trials - 1)
print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
# print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env, trial_lines=False)
print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials,trial_lines=False)
#compute and plot preferences for multiple trials
preferences= compute_preferences(combined_info)
plot_rollout_preferences(preferences, "cumulative_preferences", batch_idx=0, title="Cumulative preferences", zoom=False)

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
# render_rollout(env, info)
# plot_beliefs(info, env) #BUG
# print_rollout(info, env) #BUG
# print_parameter_learning(info, learning_config, env, verbose=False)
# plot_parameter_learning(info, learning_config, env) #BUG: but this makes no sense to plot as we cannot compare it to the well-specified model

# pe_analysis_misspecified = compute_prediction_errors(info)
# plot_prediction_errors(pe_analysis_misspecified)

# plot_model_comparison((pe_analysis, pe_analysis_misspecified), 
#                      labels=('Well-specified', 'Misspecified'))
# %% TEST: Bayesian model comparison with different models

# Create misspecified model with more states than the environment
true_structure = env.get_structure()

pe_analysis_misspecified = []

labels = list(range(1,8))

for num_states in labels:
    print(f"Testing model with {num_states} location states")
    misspecified_num_states = [num_states, 2]
    misspecified_structure = true_structure.modify(
        num_states=misspecified_num_states,
        T=model.structure.T
    )

    # Enable all parameter learning
    learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

    # Initialize misspecified model and agent
    key = jr.PRNGKey(key_idx)
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

    key = jr.PRNGKey(key_idx)
    # Run simulation with misspecified model
    key, rollout_key = jr.split(key)
    _, info, _ = rollout(agent, env, num_timesteps=misspecified_model.structure.T, rng_key=rollout_key)

    pe_analysis_misspecified.append(compute_prediction_errors(info))

plot_model_comparison(pe_analysis_misspecified,labels=labels, yscale='log', smoothing=10)
# %%
