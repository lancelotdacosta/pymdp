# This is a demo of structure learning on the TMaze environment
# The code should be as modular as possible so that it is straightforward to change the environment
# The demo goes in steps: structure learning is only toward the end

# %% Importing necessary libraries
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, render_rollout, print_parameter_learning
from pymdp.envs import TMaze
from pymdp.envs.rollout import rollout, counterfactual_rollout
from pymdp.agent import Agent
from pymdp.models.pomdp import POMDPModel, POMDPStructure
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import plot_prediction_errors, plot_model_comparison
import matplotlib.pyplot as plt

# if __name__ == "__main__":
key_idx = 0 # Initialize master random key index at the start

#%% Initialise environment

# Set up batch size
batch_size = 1

# Initialize environment
env = TMaze(batch_size=batch_size)

# %% ### 1. Basic Demo

# Set up random key
key = jr.PRNGKey(key_idx)

# Initialize agent's learning config
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

# Initialise POMDP model from environment and learning config
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=10
)

# creating C tensors filled with zeros for [location], [reward], [cue] based on A shapes
C = [jnp.zeros((batch_size, a.shape[1]), dtype=jnp.float32) for a in model.A] 
# setting preferences for outcomes only
C[1] = C[1].at[:,1].set(2.0)    # prefer reward
C[1] = C[1].at[:,2].set(-3.0)   # avoid punishment

# Initialize the agent based on model and other parameters
agent = Agent.from_model(
    model=model,
    C=C,
    policy_len=2,            # Plan two steps ahead
    inference_algo="fpi",
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

#%%
# Print rollout and visualize results
plot_beliefs(info, agent)
render_rollout(env, info)  # Optionally: render_rollout(env, info, save_gif=True, filename="figures/simplest.gif")
print_rollout(info)