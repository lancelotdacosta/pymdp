from enum import IntEnum
from functools import partial
from typing import Any

from pymdp.agent import Agent
from pymdp.envs.tmaze import TMaze
import jax.numpy as jnp


class AgentType(IntEnum):
    """
    The list of supported agents.
    """
    T_MAZE_ORACLE_POMDP = 0  # A simple POMDP agent for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_A = 1  # A POMDP agent equipped with Dirichlet over A for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_B = 2  # A POMDP agent equipped with Dirichlet over B for the T_MAZE environment.
    T_MAZE_POMDP_LEARNING_A_B = 3  # A POMDP agent equipped with Dirichlet over A and B for the T_MAZE environment.


def make(agent_type : AgentType, **kwargs : Any) -> Agent:
    """
    Create the agent requested by the user.
    :param agent_type: the type of agent to create
    :param kwargs: the argument to forward to the agent constructor
    :return: the created agent
    """
    agents_fc = {
        AgentType.T_MAZE_ORACLE_POMDP: create_t_maze_pomdp,
        AgentType.T_MAZE_POMDP_LEARNING_A: partial(create_t_maze_pomdp, learn_a=True),
        AgentType.T_MAZE_POMDP_LEARNING_B: partial(create_t_maze_pomdp, learn_b=True),
        AgentType.T_MAZE_POMDP_LEARNING_A_B: partial(create_t_maze_pomdp, learn_a=True, learn_b=True)
    }
    return agents_fc[agent_type](**kwargs)


def create_t_maze_pomdp(
    env : TMaze,
    learn_a : bool = False,
    learn_b : bool = False,
    batch_size : int = 1,
    **kwargs : Any
) -> Agent:
    """
    Create an POMDP agent for the T_MAZE environment (the agent learn using Dirichlet distribution).
    :param env: the T_MAZE environment for which the environment is created
    :param learn_a: True, if the A matrix should be learned, False otherwise
    :param learn_b: True, if the B matrix should be learned, False otherwise
    :param batch_size: the batch size
    :param kwargs: keyword arguments (unused)
    :return: the created agent
    """

    # Create sensory likelihoods.
    a_arrays = [jnp.array(a, dtype=jnp.float32) for a in env.params["A"]]
    a_dependencies = env.dependencies["A"]
    a_dirichlet_arrays = [1e16 * a_array for a_array in a_arrays] if learn_a is True else None  # TODO isn't that already learned?

    # Create transition mappings.
    b_arrays = [jnp.array(b, dtype=jnp.float32) for b in env.params["B"]]
    b_dependencies = env.dependencies["B"]
    b_dirichlet_arrays = [1e16 * b_array for b_array in b_arrays] if learn_b is True else None  # TODO isn't that already learned?

    # Create initial states.
    d_arrays = [
        jnp.zeros((batch_size, b_arrays[0].shape[1]), dtype=jnp.float32),
        jnp.ones((batch_size, 2), dtype=jnp.float32) * 0.5
    ]
    d_arrays[0] = d_arrays[0].at[0].set(1)

    # Create prior preferences.
    c_arrays = [jnp.ones((batch_size, a_array.shape[0])) / a_array.shape[0] for a_array in a_arrays]
    c_arrays[1] = c_arrays[1].at[0, 1].set(2.0)
    c_arrays[1] = c_arrays[1].at[0, 2].set(-2.0)

    # Create the agent.
    return Agent(
        A=a_arrays,
        B=b_arrays,
        D=d_arrays,
        C=c_arrays,
        pA=a_dirichlet_arrays,
        pB=b_dirichlet_arrays,
        A_dependencies=a_dependencies,
        B_dependencies=b_dependencies,
        learn_A=learn_a,
        learn_B=learn_b,
        apply_batch = False
    )
