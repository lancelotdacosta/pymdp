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
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, render_rollout
from pymdp.envs import rollout
from pymdp.agent import Agent
from pymdp.priors import dirichlet_prior


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
#TODO: when C is set to uniform, the agent stays in the left state (when action_selection param is deterministic). Why is this?

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

# In[5]:
# Print rollout and visualize results
plot_beliefs(info, agent)
render_rollout(env, info)  # Optionally: render_rollout(env, info, save_gif=True, filename="figures/simplest.gif")
print_rollout(info)

# In[6]:

# ### 5. Parameter Learning Demo
#
# Now we'll demonstrate how the agent can learn the observation (A) and transition (B) matrices.
# We'll start by setting up priors over A and B that match the true parameters.

# Let's start by defining what parameters we want to learn
learn_A = True  # Enable learning of observation model
learn_B = True  # Enable learning of transition model

# Set up random priors over A and B
pA, A_gm = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=learn_A, key=key)
pB, B_gm = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=learn_B, key=key)


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
key = jr.PRNGKey(0)
T = 50  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)

# In[7]:
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print and visualize A learning
if learn_A:
    plot_A_learning(agent, info, env)
    print('\n Final matrix A:\n',jnp.array(info["agent"].A[0])[-1,0,:]) #-1 for last timestep, 0 for first factor

# Print and visualize B learning
if learn_B:
    actions = ['Left', 'Right']
    for a in range(2): 
        print('\n Final matrix B under action', actions[a], ':\n',jnp.array(info["agent"].B[0])[-1,0,:,:,a]) 
        # plot_B_learning(agent, info, env)

# Results:
# Joint A, B learning works under random initialization, not under strictly uniform initialization (as expected). Later could try noisy uniform initialization

# In[ ]:
# ## Testing D Learning
# Now let's test learning of the initial state distribution (D) while keeping A and B fixed

# Create environment

# Let's start by defining what parameters we want to learn
learn_D = True   # Enable learning of initial state distribution

# Set up random priors over D

pD, D_gm = dirichlet_prior(D, init="like", scale=1.0, learning_enabled=learn_D, key=key)
D_gm = D 
#TODO: there is a mistake in the way dirichlet prior normalises pD to obtain D_gm

# %%

# Create agent
agent = Agent(
    A=A_gm,
    B=B_gm,
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
key = jr.PRNGKey(0)
T = 1  # More timesteps to allow for learning
final_state, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)

# Print and visualize D learning
if learn_D:
    print('\n Initial D matrix:\n', env.params["D"][0])  # True initial state distribution
    print('\n Final learned D matrix:\n', jnp.array(info["agent"].D[0])[-1])  # Learned initial state distribution

# Results:
# Let's see how well the agent learns the true initial state distribution

# %%
