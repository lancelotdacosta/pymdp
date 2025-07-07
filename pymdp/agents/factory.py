from enum import IntEnum
from functools import partial
from typing import Any, Tuple

from pymdp.agent import Agent
from pymdp.envs.tmaze import TMaze
import jax.numpy as jnp
import jax.random as jr

from pymdp.priors import dirichlet_prior


class RandomAgent(Agent):

    def __init__(self, n_policies: int) -> None:  # TODO handle multiple actions with list
        """
        Create an agent taking random actions.
        :param n_policies: the number of policies in the environment
        """
        # TODO We should have an agent interface, not all agent must have A and B matrices
        super().__init__(
            A=[jnp.zeros((1, 1), dtype=jnp.float32)],  # Dummy matrices no belief updates for random agent
            B=[jnp.zeros((1, 1), dtype=jnp.float32)],  # Dummy matrices no belief updates for random agent
        )
        self.n_policies = n_policies

    def infer_states(self, obs, empirical_prior):
        pass

    def infer_policies(self, qs):
        pass

    def sample_action(self, q_pi):
        pass  # TODO

    def infer_parameters(self, qs, obs, actions):
        pass


class AgentType(IntEnum):
    """
    The list of supported agents.
    """
    T_MAZE_ORACLE_POMDP = 0  # A simple POMDP agent for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_A = 1  # A POMDP agent equipped with Dirichlet over A for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_B = 2  # A POMDP agent equipped with Dirichlet over B for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_A_B = 3  # A POMDP agent equipped with Dirichlet over A and B for the T_MAZE environment.
    RANDOM = 4  # An agent taking random actions in the environment, you need to specify n_actions as parameters.


def make(
    agent_type : AgentType,
    key: jr.PRNGKey,
    **kwargs : Any
) -> Tuple[Agent, jr.PRNGKey]:
    """
    Create the agent requested by the user.
    :param agent_type: the type of agent to create
    :param key: the jax pseudo random key
    :param kwargs: the argument to forward to the agent constructor
    :return: the created agent
    """
    agents_fc = {
        AgentType.T_MAZE_ORACLE_POMDP: create_t_maze_pomdp,
        AgentType.T_MAZE_POMDP_LEARNING_A: partial(create_t_maze_pomdp, learn_a=True),
        AgentType.T_MAZE_POMDP_LEARNING_B: partial(create_t_maze_pomdp, learn_b=True),
        AgentType.T_MAZE_POMDP_LEARNING_A_B: partial(create_t_maze_pomdp, learn_a=True, learn_b=True),
        AgentType.RANDOM: create_random_agent,
    }
    return agents_fc[agent_type](key=key, **kwargs)


def create_t_maze_pomdp(
    env : TMaze,
    key : jr.PRNGKey,
    learn_a : bool = False,
    learn_b : bool = False,
    batch_size : int = 1,
    **kwargs : Any
) -> Tuple[Agent, jr.PRNGKey]:
    """
    Create an POMDP agent for the T_MAZE environment (the agent learn using Dirichlet distribution).
    :param env: the T_MAZE environment for which the environment is created
    :param key: the jax pseudo random key
    :param learn_a: True, if the A matrix should be learned, False otherwise
    :param learn_b: True, if the B matrix should be learned, False otherwise
    :param batch_size: the batch size
    :param kwargs: keyword arguments passed to the agent's constructor
    :return: the created agent and the new Jax random key
    """

    # Create sensory likelihoods.
    pA, A, key = dirichlet_prior(env.params["A"], init="random", learning_enabled=learn_a, key=key)
    a_dependencies = env.dependencies["A"]

    # Create transition mappings.
    pB, B, key = dirichlet_prior(env.params["B"], init="random", learning_enabled=learn_b, key=key)
    b_dependencies = env.dependencies["B"]

    # Create initial states.
    d_arrays = [
        jnp.zeros((batch_size, B[0].shape[1]), dtype=jnp.float32),
        jnp.ones((batch_size, 2), dtype=jnp.float32) * 0.5
    ]
    d_arrays[0] = d_arrays[0].at[0].set(1)

    # Create prior preferences.
    c_arrays = [jnp.ones((batch_size, a.shape[0])) / a.shape[0] for a in A]
    c_arrays[1] = c_arrays[1].at[0, 1].set(2.0)
    c_arrays[1] = c_arrays[1].at[0, 2].set(-2.0)

    # Create the agent.
    return Agent(
        A=A,
        B=B,
        D=d_arrays,
        C=c_arrays,
        pA=pA,
        pB=pB,
        A_dependencies=a_dependencies,
        B_dependencies=b_dependencies,
        learn_A=learn_a,
        learn_B=learn_b,
        apply_batch = False,
        **kwargs
    ), key


def create_random_agent(
    n_actions: int,
    key: jr.PRNGKey,
    **kwargs: Any
) -> Tuple[Agent, jr.PRNGKey]:
    """
    Create an agent taking random actions.
    :param n_actions: the number of actions in the environment
    :param kwargs: keyword arguments (unused)
    :return: the created agent and the new Jax random key
    """
    return RandomAgent(n_actions), key
