#!/usr/bin/env python
# coding: utf-8

# #  Demo: Simplest Environment + Learning of transition probabilities
# In[1]:

import copy
import os
import pathlib
import sys
from array import array

import numpy as np

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp.envs.my_envs.simplest_env import SimplestEnv
from pymdp.agent import Agent
from pymdp.utils import plot_likelihood
from pymdp import utils

# In[2]:


# reward_probabilities = [0.85, 0.15]  # the 'true' reward probabilities
env = SimplestEnv()
A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()

# In[3]:

# set the prior over likelihoods to be uniform
pA = copy.deepcopy(A_gp) #make a copy of the likelihoods for initialization of dirichlet array of right shape

l = 3 # this is the concentration parameter of the Dirichlet distribution. It is a scalar that determines the strength of the prior. A high value means the prior is strong, a low value means the prior is weak.
for i in range(len(pA)): #make each entry uniform
    pA[i] = l * np.ones_like(pA[i])

A_gm = utils.norm_dist_obj_arr(
    pA)  # initialize the agent's beliefs about the likelihoods to be the centre of the prior Dirichlet distribution. Under the hood, this takes the Dirichlet parameters of the prior, and scales them to sum to 1.0

# In[4]:


plot_likelihood(A_gm[0][:, :], 'Likelihood map \n from location state to location observation')

# In[6]:


B_gm = copy.deepcopy(B_gp)

plot_likelihood(B_gm[0][:, :, 0], 'Transition from location states under action LEFT')
plot_likelihood(B_gm[0][:, :, 1], 'Transition from location states under action RIGHT')

# In[7]:

controllable_indices = [0]  # this is a list of the indices of the hidden state factors that are controllable
learnable_modalities = [0]  # this is a list of the modalities (ie indices of observation factors) that you want to be learn-able

# In[8]:


agent = Agent(A=A_gm, pA=pA, B=B_gm,
              control_fac_idx=controllable_indices,
              modalities_to_learn=learnable_modalities,
              lr_pA=1,  # mathematically this should always be equal to 1.0
              use_param_info_gain=True)

# see how this compares to the non-learning case: agent = Agent(A=A_gm, B=B_gm, control_fac_idx=controllable_indices)


# In[9]:

agent.D[0] = utils.onehot(0, agent.num_states[
    0])  # set the initial prior over location state to be dirac at left location

# In[10]:

# set the prior over observations
agent.C[0][0] = 0.0
agent.C[0][1] = 2.0

# In[11]:


T = 100  # number of timesteps

obs = env.reset()  # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
location_observations = ['Left','Right']
msg = """ === Starting experiment === \n Observation: [{}]"""
print(msg.format(location_observations[obs[0]]))

pA_history = []

all_actions = np.zeros((T, 2))  # 2 because there are two state factors.
for t in range(T):
    qx = agent.infer_states(obs)

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()  # the action is a vector that gives the action index for each controllable state factor. in this case it is a vector of two numbers.

    pA_t = agent.update_A(
        obs)  # Update approximate posterior beliefs about Dirichlet parameters. Note that we do so after each observation, not at the end of the trial.
    pA_history.append(pA_t)

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action)

    all_actions[t, :] = action  # store action

    msg = """[Step {}] Observation: [{}]"""
    print(msg.format(t, location_observations[obs[0]]))


# In[12]:

plot_likelihood(A_gm[0][:, :], 'Initial beliefs about location->observation mapping')
plot_likelihood(agent.A[0][:, :], 'Final beliefs about location->observation mapping')
# recall that when learning, agent.A is the expected value of the approximate posterior over A.
plot_likelihood(A_gp[0][:, :], 'True location->observation mapping')

print(pA_history)
