from enum import IntEnum
from typing import List

from pymdp.envs import Env
from pymdp.envs.tmaze import TMaze


class EnvType(IntEnum):
    """
    The list of supported environments.
    """
    T_MAZE = 0


def make(env_type : EnvType, **kwargs : List[int]) -> Env:
    """
    Create the environment requested by the user.
    :param env_type: the type of environment to create
    :param kwargs: the argument to forward to the environment constructor
    :return: the created environment
    """
    envs_fc = {
        # [TMaze Environment]
        # ==> States:
        # LOCATION_FACTOR_ID = 0 -> Describe the agent location (center, left?, right?, bottom/cue).
        # TRIAL_FACTOR_ID = 1 => Describe the reward location (left arm?, right arm?).
        # ==> Actions.
        # There are two actions per time step, i.e., one for each latent factor:
        # LOCATION_FACTOR_ID = 0 => Move agent, four possibilities (move center, move left?, move right?, move bottom/cue).
        # TRIAL_FACTOR_ID = 1 => Only one action (dummy/do nothing) as the agent does not control reward.
        # ==> Observations:
        # LOCATION_MODALITY_ID = 0  => Agent observes its locations (center, left?, right?, bottom/cue).
        # REWARD_MODALITY_ID = 1 => Agent observes rewards (NO_REWARD = 0, REWARD_IDX = 1, LOSS_IDX = 2).
        # CUE_MODALITY_ID = 2 => Agent observes cues (reward is in left arm, reward is in right arm), uniform if not in cue location.
        EnvType.T_MAZE: TMaze
    }
    return envs_fc[env_type](**kwargs)
