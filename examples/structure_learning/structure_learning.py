from enum import IntEnum
from typing import Dict, List

from jax import numpy as jnp
from jax import random as jr
from PIL import Image, ImageDraw, ImageFont
from pymdp import maths
from pymdp.agent import Agent
from pymdp.envs import Env
from pymdp.envs.tmaze import TMaze
import imageio

from pymdp.agents.factory import AgentType
from pymdp.envs.factory import EnvType
from pymdp import agents
from pymdp import envs


class TMazeAgentPosition(IntEnum):
    """
    The list of supported position T_MAZE states.
    """
    CENTER = 0
    LEFT_ARM = 1
    RIGHT_ARM = 2
    HINT = 3


class TMazeState(IntEnum):
    """
    The list of supported T_MAZE states.
    """
    EMPTY = 0
    MOUSE = 1
    CHEESE = 1 << 1
    SHOCK = 1 << 2
    EATING_MOUSE = MOUSE | CHEESE
    SHOCKED_MOUSE = MOUSE | SHOCK
    CONFUSED_MOUSE = MOUSE | CHEESE | SHOCK
    CHEESE_OR_SHOCK = CHEESE | SHOCK
    WALL = 1 << 3


def render_t_maze(
    obs : List[jnp.ndarray],
    env : TMaze,
    images : Dict[TMazeState, Image],
    cell_size : int = 500
) -> jnp.ndarray:
    """
    Render the environment state as a human-readable image.
    :param obs: the observation made by the agent
    :param env: the environment
    :param images: the sprites used to render the environment
    :param cell_size: the size of each cell in the maze
    :return: an image representing the T_MAZE
    """

    # Create an empty image.
    n_rows = 2
    n_cols = 3
    image = Image.new("RGB", (cell_size * n_cols, cell_size * n_rows), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Create a grid representing the maze.
    grid = [[TMazeState.EMPTY] * n_cols for _ in range(n_rows)]
    grid[1][0] = TMazeState.WALL
    grid[1][2] = TMazeState.WALL

    # Update the grid to reflect the agent position.
    positions_2d = {
        TMazeAgentPosition.CENTER: (0, 1),
        TMazeAgentPosition.LEFT_ARM: (0, 0),
        TMazeAgentPosition.RIGHT_ARM: (0, 2),
        TMazeAgentPosition.HINT: (1, 1)
    }
    position = TMazeAgentPosition(jnp.where(env._state[0] == 1)[0].item())
    pos_x, pos_y = positions_2d[position]
    grid[pos_x][pos_y] |= TMazeState.MOUSE

    # Update the grid to reflect the agent beliefs about the reward position.
    prob_obs = maths.spm_dot(env._likelihood_dist[2], env._state)
    left_pos_x, left_pos_y = positions_2d[TMazeAgentPosition.LEFT_ARM]
    right_pos_x, right_pos_y = positions_2d[TMazeAgentPosition.RIGHT_ARM]
    if prob_obs[0] == 0.5:
        grid[left_pos_x][left_pos_y] |= TMazeState.CHEESE
        grid[left_pos_x][left_pos_y] |= TMazeState.SHOCK
        grid[right_pos_x][right_pos_y] |= TMazeState.CHEESE
        grid[right_pos_x][right_pos_y] |= TMazeState.SHOCK
    elif prob_obs[0] == 1.0:
        grid[left_pos_x][left_pos_y] |= TMazeState.CHEESE
        grid[right_pos_x][right_pos_y] |= TMazeState.SHOCK
    else:
        grid[left_pos_x][left_pos_y] |= TMazeState.SHOCK
        grid[right_pos_x][right_pos_y] |= TMazeState.CHEESE

    # Display the grid as an image.
    for x in range(n_rows):
        for y in range(n_cols):
            cell = grid[x][y]
            if cell != TMazeState.EMPTY:
                image.paste(images[cell], (cell_size * y, cell_size * x))

    # Display the reward in the image representing the environment state.
    reward = (obs[1].item() + 1) % 3 - 1
    font = ImageFont.load_default(30)
    draw.text((0, 0), f"Reward: {reward}", (0, 0, 0), font)
    return jnp.array(image)


def run_episode(
    agent : Agent,
    env : Env,
    time_horizon : int = 10,
    gif_path : str = "./data/agent_policy.gif"
) -> None:
    """
    Run action perception-cycles.
    :param agent: the agent making decision
    :param env: the environment in which the agent is acting
    :param time_horizon: the number of time steps for which the agent is run
    :param gif_path: the path where the GIF demonstrating the agent's policy must be created
    """

    # A list of images obtained from rendering the environment.
    frames = []

    # Reset the environment.
    obs, _ = env.reset()
    frames.append(env.render(mode="rgb_array"))

    # Initialize the empirical prior using the agent initial prior over state.
    qs = agent.D  # TODO is that what agent.infer_states expects?

    # Run several action-perception cycles.
    for t in range(time_horizon):

        # Infer the posterior over latent variables.
        qs = agent.infer_states(obs, empirical_prior=qs)

        # Infer the posterior over policies.
        q_pi, _ = agent.infer_policies(qs)

        # Chose the next action to perform in the environment.
        actions = agent.sample_action(q_pi)

        # Update the A and B arrays.
        # TODO agent.infer_parameters(qs, obs, actions)

        # Perform the next action in the environment.
        obs, _ = env.step(actions)
        frames.append(env.render(mode="rgb_array"))

    # Create a GIF demonstrating the agent policy.
    imageio.mimwrite(gif_path, frames, duration=1000)


if __name__ == "__main__":

    # Create the environment.
    env = envs.make(EnvType.T_MAZE)

    # Create the agent.
    agent = agents.make(AgentType.T_MAZE_ORACLE_POMDP, env=env)

    # Run action-perception cycles.
    # TODO improve TMaze.render?
    # images = {
    #     TMazeState.CHEESE: Image.open("./envs/assets/small/cheese.png"),
    #     TMazeState.MOUSE: Image.open("./envs/assets/small/mouse.png"),
    #     TMazeState.SHOCK: Image.open("./envs/assets/small/shock.png"),
    #     TMazeState.CHEESE_OR_SHOCK: Image.open("./envs/assets/small/cheese_or_shock.png"),
    #     TMazeState.EATING_MOUSE: Image.open("./envs/assets/small/mouse_eating.png"),
    #     TMazeState.SHOCKED_MOUSE: Image.open("./envs/assets/small/mouse_shocked.png"),
    #     TMazeState.CONFUSED_MOUSE: Image.open("./envs/assets/small/confused_mouse.png"),
    #     TMazeState.WALL: Image.open("./envs/assets/small/wall.png")
    # }
    run_episode(agent, env, time_horizon=1000)
