# This is a demo of structure learning on the TMaze environment
# The code should be as modular as possible so that it is straightforward to change the environment
# The demo goes in steps: structure learning is only toward the end

# %% Importing necessary libraries
import jax.numpy as jnp
from jax import random as jr
from pymdp.learning import LearningConfig
from pymdp.envs.env_factory import make, EnvType
from pymdp.envs.simplest import SimplestEnv, plot_beliefs, plot_A_learning, print_parameter_learning
from pymdp.envs.simplest import print_rollout as legacy_print_rollout
from pymdp.analysis import print_rollout, render_rollout
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
env = make(
    EnvType.T_MAZE, 
    batch_size=batch_size
)

#%% ### 1. Basic Demo

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

# Run simulation
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

#%%
# Print rollout and visualize results
render_rollout(env, info, fps=1)
print_rollout(info, env)

#%%