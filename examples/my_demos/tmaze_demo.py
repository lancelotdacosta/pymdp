#!/usr/bin/env python
# coding: utf-8

# # Active Inference Demo: T-Maze Environment
# This demo notebook provides a full walk-through of active inference using the `Agent()` class of `pymdp`. The canonical example used here is the 'T-maze' task, often used in the active inference literature in discussions of epistemic behavior (see, for example, ["Active Inference and Epistemic Value"](https://pubmed.ncbi.nlm.nih.gov/25689102/))

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

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood, onehot
from pymdp import utils
from pymdp.envs import TMazeEnv


# ## Environment
# 
# Here we consider an agent navigating a three-armed 'T-maze,' with the agent starting in a central location of the maze. The bottom arm of the maze contains an informative cue, which signals in which of the two top arms ('Left' or 'Right', the ends of the 'T') a reward is likely to be found. 
# 
# At each timestep, the environment is described by the joint occurrence of two qualitatively-different 'kinds' of states (hereafter referred to as _hidden state factors_). These hidden state factors are independent of one another.
# 
# We represent the first hidden state factor (`Location`) as a $ 1 \ x \ 4 $ vector that encodes the current position of the agent, and can take the following values: {`CENTER`, `RIGHT ARM`, `LEFT ARM`, or `CUE LOCATION`}. For example, if the agent is in the `CUE LOCATION`, the current state of this factor would be $s_1 = [0 \ 0 \ 0 \ 1]$.
# 
# We represent the second hidden state factor (`Reward Condition`) as a $ 1 \ x \ 2 $ vector that encodes the reward condition of the trial: {`Reward on Right`, or `Reward on Left`}.  A trial where the condition is reward is `Reward on Left` is thus encoded as the state $s_2 = [0 \ 1]$.
# 
# The environment is designed such that when the agent is located in the `RIGHT ARM` and the reward condition is `Reward on Right`, the agent has a specified probability $a$ (where $a > 0.5$) of receiving a reward, and a low probability $b = 1 - a$ of receiving a 'loss' (we can think of this as an aversive or unpreferred stimulus). If the agent is in the `LEFT ARM` for the same reward condition, the reward probabilities are swapped, and the agent experiences loss with probability $a$, and reward with lower probability $b = 1 - a$. These reward contingencies are intuitively swapped for the `Reward on Left` condition. 
# 
# For instance, we can encode the state of the environment at the first time step in a `Reward on Right` trial with the following pair of hidden state vectors: $s_1 = [1 \ 0 \ 0 \ 0]$, $s_2 = [1 \ 0]$, where we assume the agent starts sitting in the central location. If the agent moved to the right arm, then the corresponding hidden state vectors would now be $s_1 = [0 \ 1 \ 0 \ 0]$, $s_2 = [1 \ 0]$. This highlights the _independence_ of the two hidden state factors -- the location of the agent ($s_1$) can change without affecting the identity of the reward condition ($s_2$).
# 

# ### 1. Initialize environment
# Now we can initialize the T-maze environment using the built-in `TMazeEnv` class from the `pymdp.envs` module.

# Choose reward probabilities $a$ and $b$, where $a$ and $b$ are the probabilities of reward / loss in the 'correct' arm, and the probabilities of loss / reward in the 'incorrect' arm. Which arm counts as 'correct' vs. 'incorrect' depends on the reward condition (state of the 2nd hidden state factor).

# In[3]:


reward_probabilities = [0.8, 0.2] # probabilities used in the original SPM T-maze demo


# Initialize an instance of the T-maze environment

# In[4]:


env = TMazeEnv(reward_probs = reward_probabilities)


# ### Structure of the state --> outcome mapping
# We can 'peer into' the rules encoded by the environment (also known as the _generative process_ ) by looking at the probability distributions that map from hidden states to observations. Following the SPM version of active inference, we refer to this collection of probabilistic relationships as the `A` array. In the case of the true rules of the environment, we refer to this array as `A_gp` (where the suffix `_gp` denotes the generative process). 
# 
# It is worth outlining what constitute the agent's observations in this task. In this T-maze demo, we have three sensory channels or observation modalities: `Location`, `Reward`, and `Cue`. 
# 
# >The `Location` observation values are identical to the `Location` hidden state values. In this case, the agent always unambiguously observes its own state - if the agent is in `RIGHT ARM`, it receives a `RIGHT ARM` observation in the corresponding modality. This might be analogized to a 'proprioceptive' sense of one's own place.
# 
# >The `Reward` observation modality assumes the values `No Reward`, `Reward` or `Loss`. The `No Reward` (index 0) observation is  observed whenever the agent isn't occupying one of the two T-maze arms (the right or left arms). The `Reward` (index 1) and `Loss` (index 2) observations are observed in the right and left arms of the T-maze, with associated probabilities that depend on the reward condition (i.e. on the value of the second hidden state factor).
# 
# > The `Cue` observation modality assumes the values `Cue Right`, `Cue Left`. This observation unambiguously signals the reward condition of the trial, and therefore in which arm the `Reward` observation is more probable. When the agent occupies the other arms, the `Cue` observation will be `Cue Right` or `Cue Left` with equal probability. However (as we'll see below when we intialise the agent), the agent's beliefs about the likelihood mapping render these observations uninformative and irrelevant to state inference.
# 
# In `pymdp`, we store the set of probability distributions encoding the conditional probabilities of observations, under different configurations of hidden states, as a set of matrices referred to as the likelihood mapping or `A` array (this is a convention borrowed from SPM). The likelihood mapping _for a single modality_ is stored as a single matrix `A[i]` with the larger likelihood array, where `i` is the index of the corresponding modality. Each modality-specific A matrix has `n_observations[i]` rows, and as many lagging dimensions (e.g. columns, 'slices' and higher-order dimensions) as there are hidden state factors. `n_observations[i]` tells you the number of observation values for observation modality `i`, and is usually stored as a property of the `Env` class (e.g. `env.n_observations`).
# 
# 

# In[4]:


A_gp = env.get_likelihood_dist()

A_gp.dtype
A_gp.shape
A_gp[0].shape
A_gp[1].shape
A_gp[2].shape


# In[5]:


plot_likelihood(A_gp[1][:,:,0],'Reward Right')


# In[6]:


plot_likelihood(A_gp[1][:,:,1],'Reward Left')


# In[7]:


plot_likelihood(A_gp[2][:,3,:],'Cue Mapping')


# ### Transition Dynamics
# 
# We represent the dynamics of the environment (e.g. changes in the location of the agent and changes to the reward condition) as conditional probability distributions that encode the likelihood of transitions between the states of a given hidden state factor. These distributions are collected into the so-called `B` array, also known as _transition likelihoods_ or _transition distribution_ . As with the `A` array, we denote the true probabilities describing the environmental dynamics as `B_gp`. Each sub-matrix `B_gp[f]` of the larger array encodes the transition probabilities between state-values of a given hidden state factor with index `f`. These matrices encode dynamics as Markovian transition probabilities, such that the entry $i,j$ of a given matrix encodes the probability of transition to state $i$ at time $t+1$, given state $j$ at $t$. 

# In[8]:


B_gp = env.get_transition_dist()


# For example, we can inspect the 'dynamics' of the `Reward Condition` factor by indexing into the appropriate sub-matrix of `B_gp`

# In[9]:

B_gp[1].shape
plot_likelihood(B_gp[1][:,:,0],'Reward Condition Transitions')


# The above transition array is the 'trivial' identity matrix, meaning that the reward condition doesn't change over time (it's mapped from whatever it's current value is to the same value at the next timestep).

# ### (Controllable-) Transition Dynamics
# 
# Importantly, some hidden state factors are _controllable_ by the agent, meaning that the probability of being in state $i$ at $t+1$ isn't merely a function of the state at $t$, but also of actions (or from the agent's perspective, _control states_ ). So now each transition likelihood encodes conditional probability distributions over states at $t+1$, where the conditioning variables are both the states at $t-1$ _and_ the actions at $t-1$. This extra conditioning on actions is encoded via an optional third dimension to each factor-specific `B` matrix.
# 
# For example, in our case the first hidden state factor (`Location`) is under the control of the agent, which means the corresponding transition likelihoods `B[0]` are index-able by both previous state and action.

# In[10]:

B_gp[0].shape

plot_likelihood(B_gp[0][:,:,0],'Transition likelihood for "Move to Center"')


# In[11]:


plot_likelihood(B_gp[0][:,:,1],'Transition likelihood for "Move to Right Arm"')


# In[12]:


plot_likelihood(B_gp[0][:,:,2],'Transition likelihood for "Move to Left Arm"')


# In[13]:


plot_likelihood(B_gp[0][:,:,3],'Transition likelihood for "Move to Cue Location"')


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



# In[18]:


plot_beliefs(agent.D[1],"Beliefs about reward condition")


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

agent.C[1][1] = 3.0
agent.C[1][2] = -3.0


# In[22]:


plot_beliefs(agent.C[1],"Prior beliefs about observations")


# ## Active Inference
# Now we can start off the T-maze with an initial observation and run active inference via a loop over a desired time interval.

# In[23]:


T = 5 # number of timesteps

obs = env.reset() # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
reward_conditions = ["Right", "Left"]
location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
reward_observations = ['No reward','Reward!','Loss!']
cue_observations = ['Cue Right','Cue Left']
msg = """ === Starting experiment === \n Reward condition: {}, Observation: [{}, {}, {}]"""
print(msg.format(reward_conditions[env.reward_condition], location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))

for t in range(T):
    qx = agent.infer_states(obs)

    msg = """[Step {}] Location inference: {}, Reward condition inference: {}"""
    print(msg.format(t, qx[0], qx[1]))

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()

    msg = """[Step {}] Action: [Move to {}]"""
    print(msg.format(t, location_observations[int(action[0])]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [{},  {}, {}]"""
    print(msg.format(t, location_observations[obs[0]], reward_observations[obs[1]], cue_observations[obs[2]]))


# 

# The agent begins by moving to the `CUE LOCATION` to resolve its uncertainty about the reward condition - this is because it knows it will get an informative cue in this location, which will signal the true reward condition unambiguously. At the beginning of the next timestep, the agent then uses this observaiton to update its posterior beliefs about states `qx[1]` to reflect the true reward condition. Having resolved its uncertainty about the reward condition, the agent then moves to `RIGHT ARM` to maximize utility and continues to do so, given its (correct) beliefs about the reward condition and the mapping between hidden states and reward observations. 
# 
# Notice, perhaps confusingly, that the agent continues to receive observations in the 3rd modality (i.e. samples from `A_gp[2]`). These are observations of the form `Cue Right` or `Cue Left`. However, these 'cue' observations are random and totally umambiguous unless the agent is in the `CUE LOCATION` - this is reflected by totally entropic distributions in the corresponding columns of `A_gp[2]` (and the agents beliefs about this ambiguity, reflected in `A_gm[2]`. See below.

# In[24]:


plot_likelihood(A_gp[2][:,:,0],'Cue Observations when condition is Reward on Right, for Different Locations')


# In[25]:


plot_likelihood(A_gp[2][:,:,1],'Cue Observations when condition is Reward on Left, for Different Locations')


# The final column on the right side of these matrices represents the distribution over cue observations, conditioned on the agent being in `CUE LOCATION` and the appropriate Reward Condition. This demonstrates that cue observations are uninformative / lacking epistemic value for the agent, _unless_ they are in `CUE LOCATION.`

# Now we can inspect the agent's final beliefs about the reward condition characterizing the 'trial,' having undergone 10 timesteps of active inference.

# In[26]:


plot_beliefs(qx[1],"Final posterior beliefs about reward condition")
