# In[1]:

# importing necessary libraries
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, render_rollout, print_parameter_learning
from pymdp.envs import rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import plot_prediction_errors, plot_model_comparison
import matplotlib.pyplot as plt
from pymdp.priors import dirichlet_prior


# ### 1. Basic Demo
#
# This demo shows how to use the simplest environment with an active inference agent.
# The environment consists of two states (left and right) and two actions (stay and move).
# The agent can observe which state it is in perfectly.
#
# First, we'll create an instance of the simplest environment and get its observation (A) and transition (B) tensors.

# Set up batch size
batch_size = 1

# Initialize environment
env = SimplestEnv(batch_size=batch_size)

key_idx= 1000

T=100

# %% #Let's investigate joint A, B, D learning.

key = jr.PRNGKey(key_idx)

# Enable learning of all parameters
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Create model from environment
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=T            #can play with this
)

# Initialize agent with parameter learning enabled
agent = Agent.from_model(
    model=model,
    #C=C,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Compute prediction errors
pe_analysis = compute_prediction_errors(info)
plot_prediction_errors(pe_analysis)

# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info)

# Print parameter learning
print_parameter_learning(info, 
    learn_A=learning_config.learn_A,
    learn_B=learning_config.learn_B,
    learn_D=learning_config.learn_D
)

# Visualize A learning
if learning_config.learn_A:
    plot_A_learning(agent, info, env)

# Results:
# Joint A, B, D learning works as best it can under random initialization. The only thing is that the agent does not learn D well because qs_0 is really imprecise (and is not updated retrospectively because there is no smoothing) and that is the only thing the agent uses to learn D.

# %% [DEBUG MODE] This should give the same results as the previous cell.

key = jr.PRNGKey(key_idx)

# Enable learning of all parameters
learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=True)

# Set up random priors over A, B, and D
pA, A_gm, key = dirichlet_prior(env.params["A"], init="random", scale=1.0, learning_enabled=learning_config.learn_A, key=key)
pB, B_gm, key = dirichlet_prior(env.params["B"], init="random", scale=1.0, learning_enabled=learning_config.learn_B, key=key)
pD, D_gm, key = dirichlet_prior(env.params["D"], init="random", scale=1.0, learning_enabled=learning_config.learn_D, key=key)

# Initialize agent with parameter learning enabled
agent2 = Agent(
    A=A_gm,
    B=B_gm,
    #C=C,
    D=D_gm,
    pA=pA,
    pB=pB,
    pD=pD,
    A_dependencies=env.dependencies["A"],
    B_dependencies=env.dependencies["B"],
    learn_A=learning_config.learn_A,
    learn_B=learning_config.learn_B,
    learn_D=learning_config.learn_D,
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation with parameter learning
key, rollout_key = jr.split(key)  # Split key for rollout
final_state, info2, _ = rollout(agent2, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Compute prediction errors
pe_analysis2 = compute_prediction_errors(info2)
plot_prediction_errors(pe_analysis2)

# Print rollout
print("\nRollout with parameter learning:")
print_rollout(info2)

# Print parameter learning
print_parameter_learning(info2, 
    learn_A=learning_config.learn_A,
    learn_B=learning_config.learn_B,
    learn_D=learning_config.learn_D
)

# Visualize A learning
if learning_config.learn_A:
    plot_A_learning(agent, info2, env)

# %% Compare the two implementations
plot_model_comparison(pe_analysis, pe_analysis2, 
                     labels=('Refactored', 'Previous'))

plt.plot(range(T+1), info['observation'][0][:,0,0], alpha=0.5, marker='o', linestyle='--')


#%%

# print(pe_analysis["complexity"][:2]) 

# print_rollout(info)

print(agent2.pD)
print(agent.pD)
# %%
