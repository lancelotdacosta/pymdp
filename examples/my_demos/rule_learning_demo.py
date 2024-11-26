#!/usr/bin/env python
# coding: utf-8

# # Active Inference Demo: Rule Learning Environment
# This demo implements an active inference agent that must learn to follow different rules through 
# interaction with the environment. The task involves understanding the relationship between colors 
# and locations to infer the current rule and make correct choices.

# ### Environment Structure

# **Hidden State Factors:**
# 1. **Rule** (3 states: Left, Top, Right)
#    - Determines which location provides the informative color cue
# 2. **Colour** (3 states: Red, Green, Blue)
#    - The correct color to be chosen
# 3. **Where** (4 states: Left, Top, Right, Centre)
#    - Current focus of attention
# 4. **Choice** (4 states: Red, Green, Blue, Undecided)
#    - Agent's color selection

# **Observation Modalities:**
# 1. **What** - Color seen at current location
#    - Top location: indicates rule (red->left, green->top, blue->right)
#    - Centre: always white
#    - Left/Right: random fixed colors
# 2. **Where** - Current gaze location
# 3. **Feedback** - Response to choices (Neutral, Correct, Incorrect)

# ### Script Structure
# 1. Environment setup and agent configuration
# 2. Testing loop showing:
#    - Observations and actions at each timestep
#    - Agent's beliefs about all state factors
#    - Feedback received
# 3. Results visualization:
#    - Belief evolution for all state factors
#    - Trial-by-trial performance

# ### Imports
import os
import sys
import pathlib
import numpy as np
import copy
from contextlib import redirect_stdout
import io

# Add the pymdp root directory to Python path
current_file = pathlib.Path().absolute()
project_root = str(current_file.parent.parent)  # Go up two levels from examples/my_demos to reach root
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
print(f"Added to Python path: {project_root}")

from pymdp.agent import Agent
from pymdp.utils import plot_beliefs, plot_likelihood, update_matrix
from pymdp import utils
from pymdp.envs.my_envs.rule_learning import RuleLearningEnv

# Initialize environment
env = RuleLearningEnv()


# In[3]: get the likelihood distribution and transitiion probabilities and initial state sample
A_gp = env.get_likelihood_dist()
B_gp = env.get_transition_dist()


# In[4]: TEST Plot the transition probabilities
# plot_likelihood(B_gp[0][:,:,0],'Rule Factor Transition Probabilities')
# plot_likelihood(B_gp[1][:,:,0],'True Color Factor Transition Probabilities')
# plot_likelihood(B_gp[2][:,:,0],'Where Factor Transition Probabilities - action Left')
# plot_likelihood(B_gp[2][:,:,1],'Where Factor Transition Probabilities - action Top')
# plot_likelihood(B_gp[2][:,:,2],'Where Factor Transition Probabilities - action Right')
# plot_likelihood(B_gp[2][:,:,3],'Where Factor Transition Probabilities - action Centre')
# plot_likelihood(B_gp[3][:,:,0],'Choice Factor Transition Probabilities - action Red')
# plot_likelihood(B_gp[3][:,:,1],'Choice Factor Transition Probabilities - action Green')
# plot_likelihood(B_gp[3][:,:,2],'Choice Factor Transition Probabilities - action Blue')
# plot_likelihood(B_gp[3][:,:,3],'Choice Factor Transition Probabilities - action Undecided')

# In[4]: TEST Plot the likelihood distribution
# i = np.random.choice(env.num_rules)
# print('rule:', i, '(0:left->red, 1:top->green, 2:right->blue)')
# j = np.random.choice(env.num_colours)
# k = np.random.choice(env.num_locations)
# plot_likelihood(A_gp[0][:,i,j,:,k],'What Modality Likelihoods (depends on location and rule for top location)')
# plot_likelihood(A_gp[1][:,i,j,:,k],'Where Modality Likelihoods (pure deterministic mapping from where state)')
# print('Feedback Modality Likelihoods (deterministic based on choice matching hidden color):')
# print(A_gp[2][:,i,j,k,j])#,'Feedback Modality Likelihoods (should be independent of rule and where)')
#A[0]is correct

# for l in range(env.num_choices):    
#     plot_likelihood(A_gp[0][:,2,:,2,l],'.')


# In[5]: Test initial state sample
obs=env.reset()
D_gp = env.get_state()
# print(D_gp)


# In[5]: Setup agent generative model
# ### Agent

# Get dimensionalities of observation modalities and hidden state factors
num_obs = env.num_obs
num_states = env.num_states
num_controls = env.num_controls

print("\nEnvironment dimensions:")
print(f"num_obs: {num_obs}, type: {type(num_obs)}")
print(f"num_states: {num_states}, type: {type(num_states)}")
print(f"num_controls: {num_controls}, type: {type(num_controls)}")

# %%
# Initialize prior beliefs about likelihoods
# A_gp is the generative process likelihood mapping from the environment
# We create a Dirichlet prior (pA) that matches A_gp with high confidence (scale=128)
# This means the agent has accurate prior beliefs about the task structure
pA = utils.dirichlet_like(template_categorical=A_gp, scale=[128,128,128]) 

# However, we only want the agent to have informative priors about the top location (where=1)
# For all other locations, we set uniform likelihoods (all 1's)
# This means the agent only has strong beliefs about what colors mean at the top location
# where the rule is indicated
for where in range(env.num_locations): 
    if where != 1: #if location is not top
        pA[0][:,:,:,where,:] = np.ones_like(pA[0][:,:,:,where,:]) 

# Convert Dirichlet parameters to normalized probabilities for the generative model
A_gm = utils.norm_dist_obj_arr(pA)

# %%
# Plot what modality (0) for each rule at the top location (1) with choice red (0)
# This shows how different rules lead to different color expectations at the top location
for r in range(env.num_rules):
    plot_likelihood(A_gm[0][:,r,:,1,0], f'rule {r}, loc {1}, choice {0}')

# %% Initialize prior beliefs about transitions
learning_B = False
B_gm, pB = utils.dirichlet_uniform(template_categorical=B_gp, learning_enabled=learning_B)

# %%

# Create (time dependent) prior preferences
T = 3  # Number of timesteps in the simulations

C = utils.obj_array_zeros(env.num_obs)

# Incorrect feedback always to avoid, zero preference for correct or neutral feedback (but we will change the preference for neutral feedback at the third timestep, to make it take a decision)
C[2][2] = -4 

# %%

# # Allow feedback modality preferences to vary over time
# C_gm[2] = np.zeros((3, T))  # Shape: (num_feedback_obs, num_timesteps)

# # evolving over time
# for t in range(T):
#     # Incorrect feedback always to avoid
#     C_gm[2][2, t] = -4
#     # Correct feedback preferred
#     C_gm[2][1, t] = 4
#     # Neutral feedback is neutral
#     # C_gm[2][0, t] = -4
#     if t>=1: # Neutral feedback to avoid from timestep 3
#         C_gm[2][0, t] = -32

# Initialize the agent's prior beliefs about initial states
D_gm = utils.obj_array_uniform(num_states) # uniform prior on rule or true color
D_gm[2] = utils.onehot(3, env.num_locations) #start at the centre location
D_gm[3] = utils.onehot(3, env.num_choices) #start undecided about choice


# In[5]: Create agent and test
agent = Agent(A=A_gm, pA=pA, B=B_gm, pB=pB, C=copy.deepcopy(C), D=D_gm,
              num_controls=num_controls, policy_len=1,
              inference_horizon=1, inference_algo='VANILLA')


n_trials = 1  # Number of trials to run
trial_length = T  # Length of each trial (same as our planning horizon)

# Simple labels for printing
locations = ['LEFT', 'TOP', 'RIGHT', 'CENTRE']
colors = ['RED', 'GREEN', 'BLUE', 'WHITE']
feedback = ['NEUTRAL', 'CORRECT', 'INCORRECT']

obs_history = []
action_history = []
q_s_history = []

for trial in range(n_trials):
    print(f"\n=== Trial {trial} ===")
    
    obs = env.reset()  # Reset environment at the start of each trial
    agent.reset()      # Reset agent's beliefs
    
    trial_obs = []
    trial_actions = []
    trial_q_s = []

    # infer states  
    q_s = agent.infer_states(obs)

    # Log observation and beliefs about states 
    trial_obs.append(obs)
    trial_q_s.append(q_s)
    
    print(f"\nInitial observation: sees {colors[int(obs[0])]} at {locations[int(obs[1])]}, feedback: {feedback[int(obs[2])]}")
    print(f"Beliefs: rule={q_s[0].round(2)}, color={q_s[1].round(2)}")
    print(f"        location={q_s[2].round(2)}, choice={q_s[3].round(2)}")
    
    for t in range(trial_length):
        print(f"\n--- Timestep {t} ---")
        
        if t == 2:  # Change preference for neutral feedback to avoid
            agent.C[2][0] = -8

        # Get agent's action
        q_pi, efe = agent.infer_policies()
        action = agent.sample_action()   
        print(f"Action: Look {locations[int(action[2])]}, Choice: {colors[int(action[3])] if int(action[3]) < 3 else 'UNDECIDED'}")
        
        # Environment step
        obs = env.step(action)
        print(f"Observation: sees {colors[int(obs[0])]} at {locations[int(obs[1])]}, feedback: {feedback[int(obs[2])]}")

        # Update agent's beliefs
        q_s = agent.infer_states(obs)
        print(f"Beliefs: rule={q_s[0].round(2)}, color={q_s[1].round(2)}")
        print(f"        location={q_s[2].round(2)}, choice={q_s[3].round(2)}")

        # Log observation, beliefs and action
        trial_actions.append(action)
        trial_obs.append(obs)
        trial_q_s.append(q_s)
        
    obs_history.append(trial_obs)
    action_history.append(trial_actions)
    q_s_history.append(trial_q_s)

# In[5]: # Print summary statistics
correct_choices = 0
total_choices = 0

for trial_obs in obs_history:
    for obs in trial_obs:
        if obs[2] == 1:  # Correct feedback
            correct_choices += 1
        if obs[2] != 0:  # Any feedback (excluding neutral)
            total_choices += 1

print(f"\nPerformance Summary:")
print(f"Correct choices: {correct_choices}/{total_choices} ({(correct_choices/total_choices)*100:.2f}%)")

# In[5]: # Visualization of results
import matplotlib.pyplot as plt

# Extract beliefs about rules, colors and choices over time
rule_beliefs = []
color_beliefs = []
choice_beliefs = []
location_beliefs = []

for beliefs in q_s_history[0]:
    rule_beliefs.append(beliefs[0])  # Factor 0 is rule
    color_beliefs.append(beliefs[1])  # Factor 1 is color
    location_beliefs.append(beliefs[2])  # Factor 2 is location
    choice_beliefs.append(beliefs[3])  # Factor 3 is choice

# Convert to numpy arrays for easier plotting
rule_beliefs = np.array(rule_beliefs)
color_beliefs = np.array(color_beliefs)
location_beliefs = np.array(location_beliefs)
choice_beliefs = np.array(choice_beliefs)

# Create figure with subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

# Plot rule beliefs
ax1.plot(rule_beliefs[:, 0], label='Left Rule', color='red')
ax1.plot(rule_beliefs[:, 1], label='Top Rule', color='green')
ax1.plot(rule_beliefs[:, 2], label='Right Rule', color='blue')
ax1.set_title('Rule Beliefs Over Time')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Belief Probability')
ax1.legend()
ax1.grid(True)

# Plot color beliefs
ax2.plot(color_beliefs[:, 0], label='Red', color='red')
ax2.plot(color_beliefs[:, 1], label='Green', color='green')
ax2.plot(color_beliefs[:, 2], label='Blue', color='blue')
ax2.set_title('Color Beliefs Over Time')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Belief Probability')
ax2.legend()
ax2.grid(True)

# Plot location beliefs
ax3.plot(location_beliefs[:, 0], label='Left', color='red')
ax3.plot(location_beliefs[:, 1], label='Top', color='green')
ax3.plot(location_beliefs[:, 2], label='Right', color='blue')
ax3.plot(location_beliefs[:, 3], label='Centre', color='gray', linestyle='--')
ax3.set_title('Location Beliefs Over Time')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Belief Probability')
ax3.legend()
ax3.grid(True)

# Plot choice beliefs
ax4.plot(choice_beliefs[:, 0], label='Choose Red', color='red')
ax4.plot(choice_beliefs[:, 1], label='Choose Green', color='green')
ax4.plot(choice_beliefs[:, 2], label='Choose Blue', color='blue')
ax4.plot(choice_beliefs[:, 3], label='Undecided', color='gray', linestyle='--')
ax4.set_title('Choice Beliefs Over Time')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Belief Probability')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# Plot performance over trials
trial_performances = []
for trial_idx, trial_obs in enumerate(obs_history):
    correct = sum(1 for obs in trial_obs if obs[2] == 1)  # Count correct feedback
    total = sum(1 for obs in trial_obs if obs[2] != 0)    # Count non-neutral feedback
    if total > 0:
        performance = correct / total
    else:
        performance = 0
    trial_performances.append(performance)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_trials + 1), trial_performances, marker='o')
plt.title('Performance Over Trials')
plt.xlabel('Trial Number')
plt.ylabel('Proportion Correct')
plt.grid(True)
plt.show()
