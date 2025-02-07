#  Demo: Simplest Environment + Learning of contingencies
# In[1]:

import copy
import os
import pathlib
import sys

import numpy as np

# path = pathlib.Path(os.getcwd())
# module_path = str(path.parent) + '/'
# sys.path.append(module_path)

# Add the pymdp root directory to Python path
current_file = pathlib.Path().absolute()
project_root = str(current_file.parent.parent)  # Go up two levels from examples/my_demos to reach root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pymdp.envs.my_envs.simplest_env import SimplestEnv
from pymdp.agent import Agent
from pymdp.utils import plot_likelihood, update_matrix
from pymdp import utils

# In[]:
env = SimplestEnv()
A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()

# In[]:


# Initialise the likelihood and transition distributions of the generative model
learning_A=True
A_gm, pA = utils.dirichlet_uniform(template_categorical=A_gp, learning_enabled=learning_A)
learning_B=False
B_gm, pB = utils.dirichlet_uniform(template_categorical=B_gp, learning_enabled=learning_B)
learning_D=True
D_gm, pD = utils.dirichlet_uniform(template_categorical=utils.obj_array_uniform(env.num_states), learning_enabled=learning_D)
'Learning B and D but not A: works as expected, because with the right A, the agent learns the true D at the zeroth timestep without the need for smoothing'
'Learning A and D but not B: does not work because as the agent learns A (correctly), it does not update its past beliefs through smoothing so that additions to pD remain uniform'
'Learning A and B but not D does not work in the all uniform case because the agent does not have any information to update its beliefs, however if one jiggles D slightly, the agent will learn A and B (interestingly, it will learn in mirror fashion if the jiggling favours the other initial state)'
'Learning A, B, and D should work if agent implements smoothing and if one breaks the symmetry of some (which?) distributions, presumably initial state should suffice'

# In[]:
# plot_likelihood(A_gm[0][:, :], 'Likelihood map \n from location state to location observation')
# plot_likelihood(B_gm[0][:, :, 0], 'Transition from location states under action LEFT')
# plot_likelihood(B_gm[0][:, :, 1], 'Transition from location states under action RIGHT')

# In[9]:

# agent.D[0] = utils.onehot(0, agent.num_states[0])  # set the initial prior over location state to be dirac at left location
#agent.D = utils.obj_array_uniform(agent.num_states) # set the initial prior over location state to be uniform
D_gm[0] = np.array([0.51, 0.49])  # set the initial prior over location state to be close to uniform


# In[10]:

# Create prior preferences
C = utils.obj_array_zeros(env.num_obs)

# set the unnormalised log prior over observations
C[0][0] = 0.0
C[0][1] = 0.0

# In[11]:

T = 2  # number of timesteps

inference_algo = 'MMP'  # Enable MMP for smoothing
inference_horizon = T+1  # Smoothing over entire trial length
controllable_indices = [0]  # this is a list of the indices of the hidden state factors that are controllable


agent = Agent(A=A_gm, pA=pA, B=B_gm, pB=pB,C=copy.deepcopy(C), D=D_gm, pD=pD,  # pass the likelihood, transition, and initial state probability distributions
              control_fac_idx=controllable_indices,
              # modalities_to_learn=learnable_modalities,   #commented as default is to learn all modalities
              # factors_to_learn=learnable_factors,         #commented as default is to learn all factors
              lr_pA=1,lr_pB=1,lr_pD=1,
              save_belief_hist=True,  # this saves the inferences about states and policies over time: in agent.q_pi_hist and agent.qs_hist
              use_param_info_gain=True,
              inference_algo=inference_algo,
              inference_horizon=inference_horizon)

obs = env.reset()  # reset the environment and get an initial observation

# these are useful for displaying read-outs during the loop over time
location_observations = ['Left','Right']
msg = """ === Starting experiment === \n Observation: [{}]"""
print(msg.format(location_observations[obs[0]]))

pA_history, pB_history, pD_history = [pA], [pB], [pD]

action_hist = np.zeros((T, 1))  # 1 because there is one state factor.

# state inference
qx = agent.infer_states(obs)
print('Initial state inference: Left', qx[0][0], 'Right', qx[0][1])

# Call the helper function for each matrix with the appropriate learning flag
# update_matrix(agent, "update_A", pA_history, learning_A, obs) # parameter learning of matrix A
# update_matrix(agent, "update_D", pD_history, learning_D)

for t in range(T):
    # policy inference
    q_pi, efe = agent.infer_policies()
    #if t==0: print('Initial efe:', efe)
    
    # action selection
    action = agent.sample_action()  # the action is a vector that gives the action index for each controllable state factor. in this case it is a vector of two numbers.
    action_hist[t, :] = action # store the action
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
    print(f'[Step {t}] State inference: Left {qx[0][0]}, Right {qx[0][1]}')
    
    # Call the helper function for each matrix with the appropriate learning flag
    update_matrix(agent, "update_A", pA_history, learning_A, obs)  # parameter learning of matrix A
    update_matrix(agent, "update_B", pB_history, learning_B, agent.qs_hist[t - 1]) # parameter learning of matrix B
    update_matrix(agent, "update_D", pD_history, learning_D) # parameter learning of matrix D

# In[12]:

if learning_A:
    # plot_likelihood(A_gm[0][:, :], 'Initial beliefs about location->observation mapping')
    plot_likelihood(agent.A[0][:, :], 'Final beliefs about location->observation mapping') # recall that when learning, agent.A is the expected value of the approximate posterior over A.
    # plot_likelihood(A_gp[0][:, :], 'True location->observation mapping')

# print(pA_history)
if learning_B:
    for u in range(agent.num_controls[0]): #loop over actions
        #plot_likelihood(B_gm[0][:, :,a], 'Initial beliefs about transition mapping under action {}'.format(location_observations[a]))
        plot_likelihood(agent.B[0][:, :, u], 'Final beliefs about transition mapping under action {}'.format(location_observations[u])) # recall that when learning, agent.A is the expected value of the approximate posterior over A.
        #plot_likelihood(B_gp[0][:, :,a], 'True transition mapping under action {}'.format(location_observations[a]))

if learning_D:
    print(D_gm[0], 'Initial beliefs about initial state distribution')
    print(agent.D[0], 'Final beliefs about initial state distribution') # recall that when learning, agent.D is the expected value of the approximate posterior over A.

# %%
