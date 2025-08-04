# This is a demo of learning on the TMaze environment
# The code should be as modular as possible so that it is straightforward to change the environment
#
# Authors: Lancelot Da Costa


# %% Importing necessary libraries
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass
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
key_idx = 4 # Initialize master random key index at the start

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

# Enable A parameter learning
learning_config = LearningConfig(learn_A=True, learn_B=False, learn_D=False)

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
num_trials = 100 # Number of trials to run
_, key, combined_info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

#print last trial of rollout
# print_initial_state(combined_info, trial_idx= num_trials - 1)
# print_rollout(combined_info, batch_idx=0, trials=num_trials - 1)
#print and plot parameter learning
plot_parameter_learning(combined_info, learning_config, env)
# print_parameter_learning(combined_info, learning_config, env)
#compute and plot prediction errors
pe_analysis = compute_prediction_errors(combined_info)
plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials)
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


#%% This is the old code for the tmaze learning demo from the pymdp repo (branch v1alpha)

#!/usr/bin/env python
# coding: utf-8

# T-Maze Joint A and B Learning Demo
# This script demonstrates joint learning of observation model (A) and transition model (B) 
# with random initialization (no prior knowledge of environment structure)

#%% importing necessary libraries
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import mediapy
import os

from jax import random as jr
from pymdp.envs import TMaze, rollout
from pymdp.agent import Agent
from PIL import Image


# Environment setup for joint A and B learning
batch_size = 1 # number of environments to run in parallel
reward_condition = None # 0 is reward in left arm, 1 is reward in right arm, None is random allocation
reward_probability = 1.0 # 100% chance of reward in the correct arm
punishment_probability = 1.0 # 100% chance of punishment in the other arm
cue_validity = 1.0 # 100% valid cues
dependent_outcomes = False

# initialising the environment
env = TMaze( 
    batch_size=batch_size, 
    reward_probability=reward_probability,     
    punishment_probability=punishment_probability, 
    cue_validity=cue_validity,          
    reward_condition=reward_condition, 
    dependent_outcomes=dependent_outcomes
)


# Creating C and D tensors needed for the agent
# Get environment structure for tensor shapes
A_env = [jnp.array(a, dtype=jnp.float32) for a in env.params["A"]]  
B_env = [jnp.array(b, dtype=jnp.float32) for b in env.params["B"]]

# creating C tensors (preferences) - only care about outcomes
C = [jnp.zeros((batch_size, a.shape[1]), dtype=jnp.float32) for a in A_env] 
C[1] = C[1].at[:,1].set(2.0)    # prefer reward
C[1] = C[1].at[:,2].set(-3.0)   # avoid punishment

# creating D tensors (initial beliefs)
D = []
# D[0]: location - agent starts in centre
D_loc = jnp.zeros((batch_size, B_env[0].shape[1]), dtype=jnp.float32) 
D_loc = D_loc.at[0,0].set(1.0)  # set centre location to 1.0
D.append(D_loc)

# D[1]: reward location - uniform distribution (no prior knowledge)
D_reward = jnp.ones((batch_size, B_env[1].shape[1]), dtype=jnp.float32) 
D_reward = D_reward / jnp.sum(D_reward, axis=1, keepdims=True)
D.append(D_reward)


# Random initialization of both A and B tensors (complete ignorance)
key = jr.PRNGKey(24)

# Randomly initialize A tensors (observation model)
A = [jnp.array(a, dtype=jnp.float32) for a in env.params["A"]]
A_dependencies = env.dependencies["A"] 

for i in [1, 2]:  # randomize outcome (i=1) and cue (i=2) observation mappings
    key, subkey = jr.split(key)
    A[i] = jr.uniform(subkey, shape=A[i].shape) # random values between 0 and 1
    A[i] = A[i] / jnp.sum(A[i], axis=1, keepdims=True) # normalize
pA = A

# Randomly initialize B tensors (transition model)  
B = [jnp.array(b, dtype=jnp.float32) for b in env.params["B"]]
B_dependencies = env.dependencies["B"]

# Randomize location transitions
key, subkey = jr.split(key)
B[0] = jr.uniform(subkey, shape=B[0].shape)
B[0] = B[0] / jnp.sum(B[0], axis=1, keepdims=True)

# Randomize reward location transitions
key, subkey = jr.split(key)
B[1] = jr.uniform(subkey, shape=B[1].shape)
B[1] = B[1] / jnp.sum(B[1], axis=1, keepdims=True)
pB = B

# Initialize agent with joint A and B learning enabled
agent = Agent(
    A, B, C, D, 
    pA=pA,
    pB=pB,
    policy_len=5,
    A_dependencies=A_dependencies, 
    B_dependencies=B_dependencies,
    apply_batch=False, 
    learn_A=True,    # Enable A learning
    learn_B=True,    # Enable B learning
    gamma=0.1,
    action_selection="stochastic"
)

# Run simulation with extended timeline for learning
key = jr.PRNGKey(0)
T = 500 # Extended timesteps for learning
_, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)

# Print learning results - A tensors
print("=== A TENSOR LEARNING RESULTS ===")
print("Environment's A tensor (locations)")
print(env.params["A"][0])
print("Agent's A tensor at t=0 (locations)")
print(info["agent"].A[0][0])
print(f"Agent's A tensor at t={T} (locations)")
print(info["agent"].A[0][-1])
print('Difference between agent and environment A tensors (locations)')
print(np.round(jnp.abs(info["agent"].A[0][-1]-env.params["A"][0]), decimals=2))

print("\nEnvironment's A tensor (outcomes)")
print(env.params["A"][1])
print("Agent's A tensor at t=0 (outcomes)")
print(info["agent"].A[1][0])
print(f"Agent's A tensor at t={T} (outcomes)")
print(info["agent"].A[1][-1])
print('Difference between agent and environment A tensors (outcomes)')
print(np.round(jnp.abs(info["agent"].A[1][-1]-env.params["A"][1]), decimals=2))

print("\nEnvironment's A tensor (cues)")
print(env.params["A"][2])
print("Agent's A tensor at t=0 (cues)")
print(info["agent"].A[2][0])
print(f"Agent's A tensor at t={T} (cues)")
print(info["agent"].A[2][-1])
print('Difference between agent and environment A tensors (cues)')
print(np.round(jnp.abs(info["agent"].A[2][-1]-env.params["A"][2]), decimals=2))

# Print learning results - B tensors
print("\n=== B TENSOR LEARNING RESULTS ===")
print("Environment's B tensor (locations)")
print(env.params["B"][0])
print("Agent's B tensor at t=0 (locations)")
print(info["agent"].B[0][0])
print(f"Agent's B tensor at t={T} (locations)")
print(info["agent"].B[0][-1])
print('Difference between agent and environment B tensors (locations)')
print(np.round(jnp.abs(info["agent"].B[0][-1]-env.params["B"][0]), decimals=2))

print("\nEnvironment's B tensor (reward)")
print(env.params["B"][1])
print("Agent's B tensor at t=0 (reward)")
print(info["agent"].B[1][0])
print(f"Agent's B tensor at t={T} (reward)")
print(info["agent"].B[1][-1])
print('Difference between agent and environment B tensors (reward)')
print(np.round(jnp.abs(info["agent"].B[1][-1]-env.params["B"][1]), decimals=2))

# Create visualization
frames = []
for t in range(min(info["observation"][0].shape[0], 10)):
    observations_t = [
        info["observation"][0][t, :, :],
        info["observation"][1][t, :, :],  
        info["observation"][2][t, :, :]   
    ]
    frame = env.render(mode="rgb_array", observations=observations_t)
    frame = np.asarray(frame, dtype=np.uint8)
    plt.close()
    frames.append(frame)

frames = np.array(frames, dtype=np.uint8)
mediapy.show_video(frames, fps=1)

print(f"\n=== SIMULATION COMPLETE ===")
print(f"Agent learned environment structure over {T} timesteps")
print("Joint A and B learning with random initialization successful!")

#%%
