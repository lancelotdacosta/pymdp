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
import mediapy
from PIL import Image

from jax import random as jr
from pymdp.envs.simplest import SimplestEnv, print_rollout
from pymdp.envs import rollout
from pymdp.agent import Agent


# if __name__ == "__main__":

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
#TODO: when C is uniform, the agent stays in the left state. Why is this?

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
key = jr.PRNGKey(0)  # Random key for the aif loop
T = 3  # Number of timesteps to rollout
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)


# ### 4. Plot results
#
# Let's visualize the agent's initial beliefs, final beliefs, and preferences.

# In[5]:
# Print the trajectory
print_rollout(info)

# In[6]:
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
# Rendering the results
frames = []
for t in range(info["observation"][0].shape[0]):  # iterate over timesteps
    # get observations for this timestep
    observations_t = [info["observation"][0][t, :, :]]  # Only one observation modality (location)

    frame = env.render(mode="rgb_array", observations=observations_t)  # render the environment using the observations for this timestep
    frame = jnp.asarray(frame, dtype=jnp.uint8)
    plt.close()  # close the figure to prevent memory leak
    frames.append(frame)

frames = jnp.array(frames, dtype=jnp.uint8)
mediapy.show_video(frames, fps=1)

# # uncomment the following lines to save the video as a gif
# os.makedirs("figures", exist_ok=True)
# pil_frames = [Image.fromarray(frame) for frame in frames]
# filename = os.path.join("figures", f"simplest_{batch_size}.gif")
# pil_frames[0].save(
#     filename,
#     save_all=True,
#     append_images=pil_frames[1:],
#     duration=1000,  # 1000ms per frame
#     loop=0
# )
# %%