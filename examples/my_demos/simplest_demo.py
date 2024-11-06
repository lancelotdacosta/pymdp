#!/usr/bin/env python
# coding: utf-8

# # Active Inference Demo: Simplest Environment
# This demo notebook provides a full walk-through of active inference using the `Agent()` class of `pymdp`.
# This is adapted from the T-Maze demo

# ### Imports
#
# First, import `pymdp` and the modules we'll need.

# In[2]:

import os
import sys
import pathlib
import numpy as np
import copy

from jax.experimental.pallas.ops.tpu.flash_attention import below_or_on_diag

from pymdp.envs.my_envs.simplest_env import SimplestEnv

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood, onehot
from pymdp import utils
from pymdp.envs.my_envs import simplest_env


# ### 1. Initialize environment

# Initialize an instance of the simplest environment

#States = {left, right}
#Observations = {left, right}
#Actions = {go left, go right}

# In[4]:

env = SimplestEnv()


# In[4]:
# ### Observation Likelihood

A_gp = env.get_likelihood_dist()


# In[5]:


plot_likelihood(A_gp[0][:,:],'Likelihood')



# ### Transition Dynamics
#
# We represent the dynamics of the environment (e.g. changes in the location of the agent and changes to the reward condition) as conditional probability distributions that encode the likelihood of transitions between the states of a given hidden state factor. These distributions are collected into the so-called `B` array, also known as _transition likelihoods_ or _transition distribution_ . As with the `A` array, we denote the true probabilities describing the environmental dynamics as `B_gp`. Each sub-matrix `B_gp[f]` of the larger array encodes the transition probabilities between state-values of a given hidden state factor with index `f`. These matrices encode dynamics as Markovian transition probabilities, such that the entry $i,j$ of a given matrix encodes the probability of transition to state $i$ at time $t+1$, given state $j$ at $t$.

# In[8]:

# ### (Controllable-) Transition Dynamics

B_gp = env.get_transition_dist()


# For example, we can inspect the 'dynamics' of the `Reward Condition` factor by indexing into the appropriate sub-matrix of `B_gp`

# In[9]:

plot_likelihood(B_gp[0][:,:,0],'Transition for go left action')
plot_likelihood(B_gp[0][:,:,1],'Transition for go right action')


# ## The generative model
# Now we can move onto setting up the generative model of the agent - namely, the agent's beliefs about how hidden states give rise to observations, and how hidden states transition among eachother.
#
# In almost all MDPs, the critical building blocks of this generative model are the agent's representation of the observation likelihood, which we'll refer to as `A_gm`, and its representation of the transition likelihood, or `B_gm`.
#
# Here, we assume the agent has a veridical representation of the rules of the T-maze (namely, how hidden states cause observations) as well as its ability to control its own movements with certain consequences (i.e. 'noiseless' transitions).

# In[14]:


A_gm = copy.deepcopy(A_gp) # make a copy of the true observation likelihood to initialize the observation model
B_gm = copy.deepcopy(B_gp) # make a copy of the true transition likelihood to initialize the transition model


# ###  Note !
# It is not necessary, or even in many cases _important_ , that the generative model is a veridical representation of the generative process. This distinction between generative model (essentially, beliefs entertained by the agent and its interaction with the world) and the generative process (the actual dynamical system 'out there' generating sensations) is of crucial importance to the active inference formalism and (in our experience) often overlooked in code.
#
# It is for notational and computational convenience that we encode the generative process using `A` and `B` matrices. By doing so, it simply puts the rules of the environment in a data structure that can easily be converted into the Markovian-style conditional distributions useful for encoding the agent's generative model.
#
# Strictly speaking, however, all the generative process needs to do is generate observations and be 'perturbable' by actions. The way in which it does so can be arbitrarily complex, non-linear, and unaccessible by the agent.

# ## Introducing the `Agent()` class
#
# In `pymdp`, we have abstracted much of the computations required for active inference into the `Agent()` class, a flexible object that can be used to store necessary aspects of the generative model, the agent's instantaneous observations and actions, and perform action / perception using functions like `Agent.infer_states` and `Agent.infer_policies`.
#
# An instance of `Agent` is straightforwardly initialized with a call to `Agent()` with a list of optional arguments.
#

# In our call to `Agent()`, we need to constrain the default behavior with some of our T-Maze-specific needs. For example, we want to make sure that the agent's beliefs about transitions are constrained by the fact that it can only control the `Location` factor - _not_ the `Reward Condition` (which we assumed stationary across an epoch of time). Therefore we specify this using a list of indices that will be passed as the `control_fac_idx` argument of the `Agent()` constructor.
#
# Each element in the list specifies a hidden state factor (in terms of its index) that is controllable by the agent. Hidden state factors whose indices are _not_ in this list are assumed to be uncontrollable.

# In[15]:


controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable


# Now we can construct our agent...

# In[16]:


agent = Agent(A=A_gm, B=B_gm, control_fac_idx=controllable_indices)


# Now we can inspect properties (and change) of the agent as we see fit. Let's look at the initial beliefs the agent has about its starting location and reward condition, encoded in the prior over hidden states $P(s)$, known in SPM-lingo as the `D` array.

# In[17]:

D_gm = agent.D


plot_beliefs(agent.D[0],"Beliefs about initial location")


# Let's make it so that agent starts with precise and accurate prior beliefs about its starting location.

# In[19]:


agent.D[0] = onehot(0, agent.num_states[0])


# And now confirm that our agent knows (i.e. has accurate beliefs about) its initial state by visualizing its priors again.

# In[20]:


plot_beliefs(agent.D[0],"Beliefs about initial location")


# Another thing we want to do in this case is make sure the agent has a 'sense' of reward / loss and thus a motivation to be in the 'correct' arm (the arm that maximizes the probability of getting the reward outcome).
#
# We can do this by changing the prior beliefs about observations, the `C` array (also known as the _prior preferences_ ). This is represented as a collection of distributions over observations for each modality. It is initialized by default to be all 0s. This means agent has no preference for particular outcomes. Since the second modality (index `1` of the `C` array) is the `Reward` modality, with the index of the `Reward` outcome being `1`, and that of the `Loss` outcome being `2`, we populate the corresponding entries with values whose relative magnitudes encode the preference for one outcome over another (technically, this is encoded directly in terms of relative log-probabilities).
#
# Our ability to make the agent's prior beliefs that it tends to observe the outcome with index `1` in the `Reward` modality, more often than the outcome with index `2`, is what makes this modality a Reward modality in the first place -- otherwise, it would just be an arbitrary observation with no extrinsic value _per se_.

# In[21]:

C_gm = agent.C

agent.C[0][0] = -1
agent.C[0][1] = 3.0 # right location is preferred


# In[22]:


plot_beliefs(agent.C[0],"Prior beliefs about observations")


# ## Active Inference
# Now we can start off the T-maze with an initial observation and run active inference via a loop over a desired time interval.

# In[23]:


T = 2 # number of timesteps

obs = env.reset() # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
location_observations = ['Left','Right']
msg = """ === Starting experiment === \n Observation: [{}]"""
print(msg.format(location_observations[obs[0]]))

for t in range(T):
    qx = agent.infer_states(obs) # interestingly this does not seem to be needed for policy inference
    #interestingly also, the inference algo does not require the history of observations, only the current observation

    msg = """[Step {}] State inference: {}"""
    print(msg.format(t, qx[0]))

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action) #this is the new observation following the action taken \n
    #perhaps it would be better for it to come before the action, but this is a minor detail

    msg = """[Step {}] Observation: [{}]"""
    print(msg.format(t, location_observations[obs[0]]))



# Now we can inspect the agent's final beliefs about the agent's location characterizing the 'trial,' having undergone 5 timesteps of active inference.


plot_beliefs(qx[0],"Final posterior beliefs about location")

