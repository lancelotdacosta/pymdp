from enum import IntEnum
from typing import Dict, Any
from pymdp.envs import Env
from pymdp.envs.tmaze import TMaze
from pymdp.envs.simplest import SimplestEnv
from pymdp.envs.gridworld import GridWorldEnv

'''Environment factory'''


class EnvType(IntEnum):
    """
    The list of supported environments.
    """
    SIMPLEST = 0
    T_MAZE = 1
    GRIDWORLD = 2


def make(env_type : EnvType, **kwargs : Dict[str, Any]) -> Env:
    """
    Create the environment requested by the user.
    :param env_type: the type of environment to create
    :param kwargs: the argument to forward to the environment constructor
    :return: the created environment
    """
    envs_fc = {
        # [SimplestEnv Environment]
        # ==> States:
        # LOCATION_FACTOR_ID = 0 -> Describes the agent location (left=0, right=1).
        # ==> Actions.
        # There are two possible actions (left=0, right=1) which deterministically lead to their respective states.
        # ==> Observations:
        # LOCATION_MODALITY_ID = 0 -> Agent directly observes its location (left=0, right=1)
        EnvType.SIMPLEST: SimplestEnv,
        
        # [TMaze Environment]
        # ==> States:
        # LOCATION_FACTOR_ID = 0 -> Describe the agent location (center, left?, right?, bottom/cue).
        # REWARD_FACTOR_ID = 1 -> Describe the reward location (left arm?, right arm?).
        # ==> Actions.
        # There are two actions per time step, i.e., one for each latent factor:
        # LOCATION_FACTOR_ID = 0 => Move agent, four possibilities (move center, move left?, move right?, move bottom/cue).
        # REWARD_FACTOR_ID = 1 => Only one action (dummy/do nothing) as the agent does not control reward.
        # ==> Observations:
        # LOCATION_MODALITY_ID = 0  => Agent observes its locations (center, left?, right?, bottom/cue).
        # REWARD_MODALITY_ID = 1 => Agent observes rewards (NO_REWARD = 0, REWARD_IDX = 1, LOSS_IDX = 2).
        # CUE_MODALITY_ID = 2 => Agent observes cues (reward is in left arm, reward is in right arm), uniform if not in cue location.
        EnvType.T_MAZE: TMaze,
        
        # [GridWorld Environment]
        # ==> States:
        # LOCATION_FACTOR_ID = 0 -> Describes the agent location in a grid (rows × cols states).
        # ==> Actions.
        # There are five possible actions (up=0, left=1, right=2, down=3, stay=4) which deterministically move the agent.
        # ==> Observations:
        # LOCATION_MODALITY_ID = 0 -> Agent directly observes its location (fully observed grid world)
        EnvType.GRIDWORLD: GridWorldEnv
    }
    return envs_fc[env_type](**kwargs)
