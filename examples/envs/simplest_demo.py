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
import matplotlib.pyplot as plt


# if __name__ == "__main__":
key = jr.PRNGKey(2)  # Initialize master random key at the start

# ### 1. Initialize environment and get its parameters
#
# First, we'll create an instance of the simplest environment and get its observation (A) and transition (B) tensors.

# In[2]:
batch_size = 1

# Initialize environment
env = SimplestEnv(batch_size=batch_size)

# Get A and B tensors from environment
A = [jnp.array(a, dtype=jnp.float32) for a in env.params["A"]]
A_dependencies = env.dependencies["A"]

B = [jnp.array(b, dtype=jnp.float32) for b in env.params["B"]]
B_dependencies = env.dependencies["B"]

# ### 2. Set up the agent's generative model
#
# Now we'll create the agent's model of the world. In this case, since the environment is fully observed and deterministic,
# we use the same A and B tensors as the environment.

# In[3]:

# Initialize agent's generative model
# In this case, we use the same A and B tensors as the environment since it's fully observed and deterministic
A_gm = [a.copy() for a in A]
B_gm = [b.copy() for b in B]

# Set up preference (C) matrix
# The agent prefers to be in the right state (state 1)
num_obs = [a.shape[0] for a in A]
# C = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 1].set(1.0)]  # Prefer right state
C = [jnp.zeros((batch_size, 2), dtype=jnp.float32)]  # All states equally preferred

# Set up initial beliefs (D)
# Start with certainty about being in the left state (matching the environment's initial state)
num_states = [b.shape[0] for b in B]
# D = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 0].set(1.0)]  # Certain about starting in left state
D = [jnp.ones((batch_size, 2), dtype=jnp.float32) * 0.5]  # Equal probability for left and right states


# ### 3. Initialize the agent and run simulation
#
# Finally, we'll create the agent with our model parameters and run it in the environment.

# In[4]:

# Initialize the agent
agent = Agent(
    A=A_gm,
    B=B_gm,
    C=C,
    D=D,
    policy_len=1,            # Plan one step ahead
    A_dependencies=A_dependencies,
    B_dependencies=B_dependencies,
    inference_algo="fpi",
    apply_batch=False,
    learn_A=False,
    learn_B=False
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
# Now we'll demonstrate how the agent can learn the observation (A) and transition (B) and (later) initial state (D) tensors.

# Let's start by defining what parameters we want to learn
learn_A = True  # Enable learning of observation model
learn_B = True  # Enable learning of transition model

# Set up random priors over A and B
key, key_A = jr.split(key)
key, key_B = jr.split(key)
pA, A_gm = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=learn_A, key=key_A)
pB, B_gm = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=learn_B, key=key_B)


# In[6]:
# Initialize agent with parameter learning enabled
agent = Agent(A=A_gm,
             B=B_gm,
             C=C,
             D=D,
             pA=pA,  # Prior over A
             pB=pB,  # Prior over B
             A_dependencies=A_dependencies,
             B_dependencies=B_dependencies,
             learn_A=learn_A,  # Enable learning of observation model
             learn_B=learn_B,  # Enable learning of transition model
             apply_batch=False,
             action_selection="stochastic")

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
T = 1  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# In[7]:
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info, learn_A=learn_A, learn_B=learn_B)

# Visualize A learning
if learn_A:
    plot_A_learning(agent, info, env)

# Results:
# Joint A, B learning works under random initialization, not under strictly uniform initialization (as expected). Later could try noisy uniform initialization

# %%
# ## Testing D Learning
# Now let's test learning of the initial state distribution (D) while keeping A and B fixed

# Let's start by defining what parameters we want to learn
learn_D = True   # Enable learning of initial state distribution

# Set up random priors over D
key, key_D = jr.split(key)
pD, D_gm = dirichlet_prior(D, init="like", scale=1.0, learning_enabled=learn_D, key=key_D)

# %%
# Create agent
agent = Agent(
    A=env.params["A"],  # Use true A
    B=env.params["B"],  # Use true B
    C=C,
    D=D_gm,
    pD=pD,
    learn_A=False,
    learn_B=False,
    learn_D=learn_D,
    A_dependencies=A_dependencies,
    B_dependencies=B_dependencies,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
T = 1  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# Rollout with D learning
print("\nRollout with D learning:")
print_rollout(info)

# Print and visualize D learning
if learn_D:
    print('\n Parameter D learning:\n')  # True initial state distribution
    # print('\n Initial D matrix:\n', jnp.array(info["agent"].D[0])[0])  # True initial state distribution
    # print('\n Final learned D matrix:\n', jnp.array(info["agent"].D[0])[-1])  # Learned initial state distribution
    for t in range(T+1):
        print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])

# Results:
# Let's see how well the agent learns the true initial state distribution

# %% #Let's investigate joint A, B, D learning.

# Let's start by defining what parameters we want to learn
learn_A = True  # Enable learning of observation model
learn_B = True  # Enable learning of transition model
learn_D = True  # Enable learning of initial state distribution

#DEBUG LINES
# learn_D = False  # Enable learning of initial state distribution
# D = [jnp.array([[0.5, 0.5]] * batch_size, dtype=jnp.float32)]
# learn_A = False 
# learn_B = False

# Set up random priors over A, B, and D
key, key_A = jr.split(key)
key, key_B = jr.split(key)
key, key_D = jr.split(key)
pA, A_gm = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=learn_A, key=key_A)
pB, B_gm = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=learn_B, key=key_B)
pD, D_gm = dirichlet_prior(D, init="random", scale=1.0, learning_enabled=learn_D, key=key_D)

# In[6]:
# Initialize agent with parameter learning enabled
agent = Agent(A=A_gm,
             B=B_gm,
             C=C,
             D=D,
             pA=pA,  # Prior over A
             pB=pB,  # Prior over B
             pD=pD,
             A_dependencies=A_dependencies,
             B_dependencies=B_dependencies,
             learn_A=learn_A,  # Enable learning of observation model
             learn_B=learn_B,  # Enable learning of transition model
             learn_D=learn_D,
             apply_batch=False,
             action_selection="stochastic")

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
print_parameter_learning(info, learn_A=learn_A, learn_B=learn_B, learn_D=learn_D)

# Visualize A learning
if learn_A:
    plot_A_learning(agent, info, env)

#Result: joint A, B, D learning works as best it can under random initialization. The only thing is that the agent does not learn D well because qs_1 is really imprecise (and is not updated later because there is no smoothing) and that is the only thing the agent uses to learn D.

# %% #Let's investigate active inference and learning under a mispecified generative model.
# Here we will investigate joint A, B, D learning and prediction error accumulation for a one layer, n latent state POMDP in the simplest environment.

# Specify number of latent states for experiment
num_states = 5 # Can fiddle with this

# Configure POMDP dimensions
pomdp_config = {
    'num_obs': 2,           # Number of observations
    'num_states': num_states,        # Number of hidden states
    'num_actions': 2,       # Number of actions
    'num_modalities': 1,    # Number of observation modalities
    'num_factors': 1,       # Number of state factors
    'num_batches': batch_size, # Number of batches
    'T': 100                 # Number of timesteps
}

# Create uniform dummy tensors of the right shape to initialize the generative model
# A: Observation model - P(o|s) - shape (modalities)(batch, obs, states)
A_gm = [
    jnp.ones((pomdp_config['num_batches'], pomdp_config['num_obs'], pomdp_config['num_states']), dtype=jnp.float32) / pomdp_config['num_obs']
] * pomdp_config['num_modalities']

# B: Transition model - P(s'|s,a) - shape (factors)(batch, states', states, actions)
B_gm = [
    jnp.ones((pomdp_config['num_batches'], pomdp_config['num_states'], pomdp_config['num_states'], pomdp_config['num_actions']), dtype=jnp.float32) / pomdp_config['num_states']
] * pomdp_config['num_factors']

# D: Initial state belief - P(s0) - shape (factors)(batch, states)
D_gm = [
    jnp.ones((pomdp_config['num_batches'], pomdp_config['num_states']), dtype=jnp.float32) / pomdp_config['num_states']
] * pomdp_config['num_factors']

# Define what parameters we want to learn
learn_A = True  # Enable learning of observation model
learn_B = True  # Enable learning of transition model
learn_D = True  # Enable learning of initial state distribution

# Set up random priors over A, B, and D using the dummy tensors
key, key_A = jr.split(key)
key, key_B = jr.split(key)
key, key_D = jr.split(key)
pA, A_gm = dirichlet_prior(A_gm, init="random", scale=1.0, learning_enabled=learn_A, key=key_A)
pB, B_gm = dirichlet_prior(B_gm, init="random", scale=1.0, learning_enabled=learn_B, key=key_B)
pD, D_gm = dirichlet_prior(D_gm, init="random", scale=1.0, learning_enabled=learn_D, key=key_D)

# %% Initialize agent and run simulation
# Initialize agent
agent = Agent(A=A_gm,
             B=B_gm,
             C=C, #prior preferences over observations
             D=D_gm,
             pA=pA,  # Prior over A
             pB=pB,  # Prior over B
             pD=pD,
             A_dependencies=A_dependencies,
             B_dependencies=B_dependencies,
             learn_A=learn_A,  # Enable learning of observation model
             learn_B=learn_B,  # Enable learning of transition model
             learn_D=learn_D,
             apply_batch=False,
             action_selection="stochastic")

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
T = pomdp_config['T']  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=rollout_key)

# %% Analyse rollout and learning
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info, learn_A=learn_A, learn_B=learn_B, learn_D=learn_D)

#%% Compute and plot prediction errors

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