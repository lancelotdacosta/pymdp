""" Simplest Environment

__author__: Lancelot Da Costa

'''This is the simplest environment to sanity test pymdp: an environment with two states
In simulations, there will be a starting location and a preferred location'''
"""

from pymdp.envs import Env
from pymdp import utils, maths
import numpy as np

LOCATION_FACTOR_ID = 0
TRIAL_FACTOR_ID = 1

LOCATION_MODALITY_ID = 0
REWARD_MODALITY_ID = 1
CUE_MODALITY_ID = 2

REWARD_IDX = 1
LOSS_IDX = 2


class SimplestEnv(Env):
    """ Implementation of the simplest environment """

    def __init__(self):
        self.num_states = [2]
        self.num_locations = self.num_states[LOCATION_FACTOR_ID]
        self.num_controls = [self.num_locations]
        self.num_obs = [self.num_locations]
        self.num_factors = len(self.num_states) #number of hidden state factors
        self.num_modalities = len(self.num_obs) #number of observation modalities. Any single modality can be caused by multiple hidden state factors

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._state = None

    def reset(self, state=None):
        if state is None: #state is a list of one-hot vectors
            loc_state = utils.onehot(0, self.num_locations) #start at location 0
            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()

    def step(self, actions):
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()

    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist

    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):
        #this seems to be environment agnostic (as long as we are in a POMDP)
        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]

        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs

    def _construct_transition_dist(self):
        B_locs = np.eye(self.num_locations)
        B_locs = B_locs.reshape(self.num_locations, self.num_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.num_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = utils.obj_array(self.num_factors)

        B[LOCATION_FACTOR_ID] = B_locs
        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([[obs_dim] + self.num_states for obs_dim in self.num_obs])

        for loc in range(self.num_states[LOCATION_FACTOR_ID]):

            # The agent always observes its location
            A[LOCATION_MODALITY_ID][loc, loc] = 1.0

        return A

    def _construct_state(self, state_tuple):

        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)

        return state

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition
