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
import matplotlib.pyplot as plt
import numpy as np
import os

from jax import random as jr
from pymdp.envs.simplest import SimplestEnv
from pymdp.envs import rollout
from pymdp.agent import Agent

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
C = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 1].set(1.0)]  # Prefer right state

# Set up initial beliefs (D)
# Start with certainty about being in the left state (matching the environment's initial state)
num_states = [b.shape[0] for b in B]
D = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 0].set(1.0)]  # Certain about starting in left state

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
    apply_batch=False,
    learn_A=False,
    learn_B=False
)

# Run simulation
key = jr.PRNGKey(0)  # Random key for the aif loop
T = 5  # Number of timesteps to rollout
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)

# ### 4. Plot results
#
# Let's visualize the agent's initial beliefs, final beliefs, and preferences.

# In[5]:

# Plot results
plt.figure(figsize=(12, 4))

# Plot initial beliefs as a bar plot
plt.subplot(131)
plt.bar([0, 1], agent.D[0][0])
plt.title('Initial Beliefs')
plt.xticks([0, 1], ['Left', 'Right'])
plt.ylim(0, 1)

# Plot final beliefs as a bar plot
plt.subplot(132)
plt.bar([0, 1], info['qs'][0][-1][0])  # Using qs instead of belief_hist
plt.title('Final Beliefs')
plt.xticks([0, 1], ['Left', 'Right'])
plt.ylim(0, 1)

# Plot preferences as a bar plot
plt.subplot(133)
plt.bar([0, 1], agent.C[0][0])
plt.title('Preferences')
plt.xticks([0, 1], ['Left', 'Right'])
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
# %%
