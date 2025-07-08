# This is a demo of structure learning on the a modular environment
# The code should be as modular as possible so that it is straightforward to change the environment
# The demo goes in steps: structure learning is only toward the end

# %% Importing necessary libraries
# TODO get_ipython().run_line_magic('load_ext', 'autoreload')
# TODO get_ipython().run_line_magic('autoreload', '2')
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
key_idx = 1 # Initialize master random key index at the start

#%% Initialise environment

# Set up batch size
batch_size = 1

# Initialize environment
env = make(
    EnvType.SIMPLEST, 
    batch_size=batch_size
)

workspace_agent_params = env.get_default_agent_params()
workspace_agent_params.update({
    "action_selection": "stochastic",
    "use_param_info_gain": True,
    "use_states_info_gain": True,
    "learning_mode": "online"
})

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
    agent_params=workspace_agent_params
)

### Run simulation
num_trials = 1 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

### Analysis of simulation results
#print last trial of rollout
print_initial_state(combined_info, trial_idx= num_trials - 1)
print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
render_rollout(env, get_info_trial(combined_info, trial_idx=0, verbose=False), fps=2)
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
    agent_params=workspace_agent_params,
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
    agent_params=workspace_agent_params
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
    model_params={"T": 100},
    agent_params=workspace_agent_params
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
# A misspecified model has a different structure than the environment - in this case,
# we use more latent states than actually exist (eg. 3 vs 2).
#
# This allows us to:
# 1. Study how agents learn with incorrect assumptions about their environment
# 2. Compare prediction errors between well-specified and misspecified models
# 3. Demonstrate Bayesian model comparison in active inference

# Learning config
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create both models with same initialization seed
true_structure = env.get_structure().modify(T=model.structure.T)
misspecified_structure = true_structure.modify(num_states=3)

init_key = jr.PRNGKey(key_idx)  # Same seed for both models
well_specified_model, _ = POMDPModel.from_structure(true_structure, learning_config, "random", 1.0, init_key)
misspecified_model, _ = POMDPModel.from_structure(misspecified_structure, learning_config, "random", 1.0, init_key)

# Create agents and run side-by-side rollouts with same seed
agents = [
    Agent.from_model(model=well_specified_model, C=env.get_default_C(), **workspace_agent_params),
    Agent.from_model(model=misspecified_model, C=env.get_default_C(), **workspace_agent_params)
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

# Compare models
plot_model_comparison(pe_analyses, labels=('Well-specified (2 states)', 'Misspecified (3 states)'))

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

# Create an agent with the well-specified structure for counterfactual analysis
counterfactual_agent = Agent.from_model(
    model=well_specified_model,
    C=env.get_default_C(),
    **workspace_agent_params
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
plot_prediction_errors(pe_analysis_counterfactual, title="Counterfactual Model (True Structure)")

# Compare all three: well-specified rollout vs misspecified rollout vs counterfactual rollout
plot_model_comparison([pe_analyses[0], pe_analyses[1], pe_analysis_counterfactual],
                     labels=('Well-specified', 'Misspecified', 'Counterfactual'), alpha=0.7, lw=1)

print("Counterfactual analysis complete. Compare the plots to see which model better explains the data.")

# %%

