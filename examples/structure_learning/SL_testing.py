from pymdp.agents.factory import make as make_agent, AgentType
from pymdp.analysis import print_parameter_learning
from pymdp.envs.env_factory import make as make_env, EnvType
from jax import random as jr

from pymdp.envs.rollout import multi_trial_rollout
from pymdp.learning import LearningConfig

if __name__ == "__main__":

    # Get a random key.
    key = jr.PRNGKey(0)

    # Create the environment and agent.
    env = make_env(EnvType.T_MAZE)
    agent, key = make_agent(AgentType.T_MAZE_POMDP_LEARNING_A_B, key=key, env=env)

    # Run simulation
    _, key, info = multi_trial_rollout(agent, env, num_timesteps=10, num_trials=1, rng_key=key)

    # Print the TODO
    learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)
    print_parameter_learning(info, learning_config, env)
