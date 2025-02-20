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
from pymdp.maths import compute_free_energy, compute_accuracy, compute_complexity
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

# Get variables from rollout info
observations = info["observation"]  #list of arrays (one per modality) shape: (T+1, batch_size, obs_dim)
beliefs = info["qs"]  # list of arrays (one per factor) shape: (T+1, batch_size, 1, num_states)
empirical_priors = info["empirical_prior"]  # list of arrays (one per factor) shape: (T+1, batch_size, num_states)
actions=info["action"]

# Get A matrix history if available
A_hist = info["agent"].A # list of arrays (one per modality) shape: (T+1, batch_size, num_obs, num_states)

# Initialize array to store free energy for each timestep
num_timesteps = observations[0].shape[0]
pe_t = jnp.zeros(num_timesteps) #initializes prediction error array
negacc_t = jnp.zeros(num_timesteps) #initializes negative accuracy array
comp_t = jnp.zeros(num_timesteps) #initializes complexity array
comp_l2_t = jnp.zeros(num_timesteps)

# Compute prediction error at each timestep
for t in range(num_timesteps):
    # Get current variables
    action_t=actions[t]
    prior_t = [p[t] for p in empirical_priors]  # Current prior (list of arrays)
    obs_t = [jnp.array(o[t].squeeze(), dtype=jnp.int32) for o in observations]  # Current observation (list of arrays)
    qs_t = [q[t] for q in beliefs]  # Current beliefs (list of arrays)
    A_t = [A_hist_mod[t] for A_hist_mod in A_hist] # Current A matrix (list of arrays)
    
    # Compute prediction error and components
    pe_t = pe_t.at[t].set(compute_free_energy(qs_t, prior_t, obs_t, A_t, distr_obs=False))
    negacc_t = negacc_t.at[t].set(-compute_accuracy(qs_t, obs_t, A_t, distr_obs=False))
    comp_t = comp_t.at[t].set(compute_complexity(qs_t, prior_t))
    comp_l2_t = comp_l2_t.at[t].set(jnp.linalg.norm(qs_t[0][0,0,:]- prior_t[0][0,:]))

# Compute accumulated prediction error
pe_accumulated = jnp.cumsum(pe_t)

# Plot prediction error over time
plt.figure(figsize=(10, 5))
plt.plot(pe_t, label='Prediction error', alpha=1.0)
plt.plot(comp_t, label='Complexity', alpha=0.7)
plt.plot(negacc_t, label='Negative accuracy', alpha=0.7)
plt.plot(comp_l2_t, label='L2 norm Complexity', alpha=0.4)
plt.plot(pe_accumulated, label='Accumulated prediction errors')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('nats')
plt.yscale('log')
plt.grid(True)
plt.show()

# %%
# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print and visualize A learning
if learn_A:
    print('\n ====Parameter A learning====')
    # plot_A_learning(agent, info, env)
    print('\n Initial matrix A:\n', info["agent"].A[0][0,0,:])
    print('\n Final matrix A:\n', info["agent"].A[0][-1,0,:]) # -1 for last timestep, 0 for first factor

# Print and visualize B learning
if learn_B:
    print('\n ====Parameter B learning====')
    actions = ['Left', 'Right']
    for a in range(2): 
        print('\n Initial matrix B under action', actions[a], ':\n', info["agent"].B[0][0,0,:,:,a])
    for a in range(2): 
        print('\n Final matrix B under action', actions[a], ':\n', info["agent"].B[0][-1,0,:,:,a]) 
        # plot_B_learning(agent, info, env)

if learn_D:
    print('\n ====Parameter D learning====')
    print('\n Initial D matrix:\n', info["agent"].D[0][0])  # True initial state distribution
    print('\n Final learned D matrix:\n', info["agent"].D[0][-1])  # Learned initial state distribution
    #DEBUG PRINT:
    # for t in range(T+1):
    #     print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])

#Result: joint A, B, D learning works as best it can under random initialization. The only thing is that the agent does not learn D well because qs_1 is really imprecise (and is not updated later because there is no smoothing) and that is the only thing the agent uses to learn D.

# %% #Let's investigate active inference and learning under a mispecified generative model.
# Here we will investigate joint A, B, D learning and prediction error accumulation for a one layer, three latent state POMDP in the simplest environment.

# Let's start by defining what parameters we want to learn
learn_A = True  # Enable learning of observation model
learn_B = True  # Enable learning of transition model
learn_D = True  # Enable learning of initial state distribution

#DEBUG LINES
# learn_D = False  # Enable learning of initial state distribution
# D = [jnp.array([[0.5, 0.5]] * batch_size, dtype=jnp.float32)]
# learn_A = False 
# learn_B = False

# Configure POMDP dimensions
pomdp_config = {
    'num_obs': 2,           # Number of observations
    'num_states': 3,        # Number of hidden states
    'num_actions': 2,       # Number of actions
    'num_modalities': 1,    # Number of observation modalities
    'num_factors': 1,       # Number of state factors
    'num_batches': batch_size, # Number of batches
    'T': 10                 # Number of timesteps
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
# %%
