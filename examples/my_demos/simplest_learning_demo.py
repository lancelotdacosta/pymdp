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

# In[]:
# reward_probabilities = [0.85, 0.15]  # the 'true' reward probabilities
env = SimplestEnv()
A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()

# In[]:
learning_A = True # set to True if you want to learn the observation probabilities

if learning_A:
    pA = copy.deepcopy(A_gp) #make a copy of the likelihoods for initialization of dirichlet array of right shape
    cA = 1 # this is the concentration parameter of the Dirichlet distribution. It is a scalar that determines the strength of the prior. A high value means the prior is strong, a low value means the prior is weak.
    for i in range(len(pA)): #make each entry uniform
        pA[i] = cA * np.ones_like(pA[i])
    A_gm = utils.norm_dist_obj_arr(pA)

else:
    A_gm = copy.deepcopy(A_gp)
    pA = None

# plot_likelihood(A_gm[0][:, :], 'Likelihood map \n from location state to location observation')

# In[]:
learning_B = True # set to True if you want to learn the transition probabilities

if learning_B:
    pB = copy.deepcopy(B_gp) #make a copy of the likelihoods for initialization of dirichlet array of right shape
    cB = 1 # this is the concentration parameter of the Dirichlet distribution. It is a scalar that determines the strength of the prior. A high value means the prior is strong, a low value means the prior is weak.
    for i in range(len(pB)): #make each entry uniform
        pB[i] = cB * np.ones_like(pB[i])
    B_gm = utils.norm_dist_obj_arr(pB)

else:
    B_gm = copy.deepcopy(B_gp)
    pB = None

# plot_likelihood(B_gm[0][:, :, 0], 'Transition from location states under action LEFT')
# plot_likelihood(B_gm[0][:, :, 1], 'Transition from location states under action RIGHT')

# In[7]:

controllable_indices = [0]  # this is a list of the indices of the hidden state factors that are controllable
# if learning_A: learnable_modalities = [0]  # this is a list of the modalities (ie indices of observation factors) that you want to be learn-able
# else: learnable_modalities = []
# if learning_B: learnable_factors = [0]  # this is a list of the hidden state factors that you want to be learn-able
# else: learnable_factors = []

# In[8]:


agent = Agent(A=A_gm, pA=pA, B=B_gm, pB=pB,
              control_fac_idx=controllable_indices,
              # modalities_to_learn=learnable_modalities,   #commented as default is to learn all modalities
              # factors_to_learn=learnable_factors,         #commented as default is to learn all factors
              lr_pA=1,lr_pB=1,  # mathematically this should always be equal to 1.0
              use_param_info_gain=True)

# see how this compares to the non-learning case: agent = Agent(A=A_gm, B=B_gm, control_fac_idx=controllable_indices)

# In[9]:

agent.D[0] = utils.onehot(0, agent.num_states[
    0])  # set the initial prior over location state to be dirac at left location

# In[10]:

# set the prior over observations
agent.C[0][0] = 0.0
agent.C[0][1] = 0.0

# In[11]:

T = 10  # number of timesteps

obs = env.reset()  # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
location_observations = ['Left','Right']
msg = """ === Starting experiment === \n Observation: [{}]"""
print(msg.format(location_observations[obs[0]]))

pA_history, pB_history, qx_history, q_pi_history = [], [], [], []

all_actions = np.zeros((T, 1))  # 1 because there is one state factor.

# state inference
qx = agent.infer_states(obs)
qx_history.append(qx)

# parameter learning of matrix A
if learning_A:
    pA_t = agent.update_A(obs)  # Update approximate posterior beliefs about Dirichlet parameters. Note that we do so after each observation, not at the end of the tria
    pA_history.append(pA_t)

for t in range(T):
    # policy inference
    q_pi, efe = agent.infer_policies()
    q_pi_history.append(q_pi)
    if t==0: print('Initial efe:', efe)

    # action selection
    action = agent.sample_action()  # the action is a vector that gives the action index for each controllable state factor. in this case it is a vector of two numbers.
    all_actions[t, :] = action # store the action
    # action print statement
    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    # new observation
    obs = env.step(action)
    # observation print statement
    msg = """[Step {}] Observation: [{}]"""
    print(msg.format(t, location_observations[obs[0]]))

    # state inference
    qx = agent.infer_states(obs)
    qx_history.append(qx)

    if learning_A:    # parameter learning of matrix A
        pA_t = agent.update_A(obs)
        pA_history.append(pA_t)

    if learning_B:    # parameter learning of matrix B
        pB_t = agent.update_B(qx_history[t-1])
        pB_history.append(pB_t)

# In[12]:

if learning_A:
    #plot_likelihood(A_gm[0][:, :], 'Initial beliefs about location->observation mapping')
    plot_likelihood(agent.A[0][:, :], 'Final beliefs about location->observation mapping') # recall that when learning, agent.A is the expected value of the approximate posterior over A.
    #plot_likelihood(A_gp[0][:, :], 'True location->observation mapping')

# print(pA_history)
if learning_B:
    for a in range(agent.num_controls[0]): #loop over actions
        #plot_likelihood(B_gm[0][:, :,a], 'Initial beliefs about transition mapping under action {}'.format(location_observations[a]))
        plot_likelihood(agent.B[0][:, :,a], 'Final beliefs about transition mapping under action {}'.format(location_observations[a])) # recall that when learning, agent.A is the expected value of the approximate posterior over A.
        #plot_likelihood(B_gp[0][:, :,a], 'True transition mapping under action {}'.format(location_observations[a]))