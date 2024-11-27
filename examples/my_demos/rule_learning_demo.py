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
from pymdp.envs.my_envs.rule_learning import RuleLearningEnv, RULE_FACTOR_ID

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
learning_A = False
if learning_A:
    learning_mode_A = 'paper' # 'uniform' or 'paper'
    print(f"\nLearning A matrix {learning_A} with learning mode: {learning_mode_A}")
    if learning_mode_A == 'paper': #like in the curiosity paper
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

        modalities_to_learn=[0]

    elif learning_mode_A == 'uniform':
        A_gm, pA = utils.dirichlet_uniform(template_categorical=A_gp, scale=1)

        modalities_to_learn=[0,1,2]
    else: 
        raise ValueError("learning_mode_A must be 'uniform' or 'paper'")
else: 
    print(f"\nLearning A matrix {learning_A}")
    A_gm,pA = utils.dirichlet_uniform(template_categorical=A_gp,learning_enabled=learning_A)

# %%
# Plot what modality (0) for each rule at the top location (1) with choice red (0)
# This shows how different rules lead to different color expectations at the top location
# for r in range(env.num_rules):
#     plot_likelihood(A_gm[0][:,r,:,1,0], f'rule {r}, loc {1}, choice {0}')

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


# In[5]: Create agent

agent = Agent(A=A_gm, pA=pA, B=B_gm, pB=pB, C=copy.deepcopy(C), D=D_gm,
              num_controls=num_controls, policy_len=1,
              inference_horizon=1, inference_algo='MMP',
              lr_pA=0.01,  # Set learning rate for A matrix
              modalities_to_learn=modalities_to_learn)  # Only learn the 'what' modality

# In[5]: Test agent

n_trials = 1  # Number of trials to run
trial_length = T  # Length of each trial (same as our planning horizon)

# Simple labels for printing
locations = ['LEFT', 'TOP', 'RIGHT', 'CENTRE']
colors = ['RED', 'GREEN', 'BLUE', 'WHITE']
feedback = ['NEUTRAL', 'CORRECT', 'INCORRECT']
rule_names = ['Left Rule', 'Top Rule', 'Right Rule']

obs_history = []
action_history = []
q_s_history = []
likelihood_history = []  # Store A matrices over time
pA_history = [pA] # Initialize learning history
trial_rules = []  # Store the rule for each trial

DEBUG = True

for trial in range(n_trials):
    print(f"\n=== Trial {trial} ===")
    
    # Reset environment, agent's beliefs and initialize trial records
    obs = env.reset()
    if DEBUG:
        while env._state[RULE_FACTOR_ID].argmax() != 0: #if rule is not left
            obs = env.reset()
    trial_rules.append(env._state[RULE_FACTOR_ID].argmax())  # Store the rule at the start of each trial
    agent.reset()      # Reset agent's beliefs at start of each trial
    agent.C[2][0] = 0  # Set preference to be neutral about neutral feedback at start of trial
    
    trial_obs = [obs]
    trial_actions = []
    trial_q_s = []
    trial_likelihoods = []  # Store A matrices for this trial
    
    # Store initial likelihood for this trial
    trial_likelihoods.append(copy.deepcopy(agent.A))
    
    # infer states  
    q_s = agent.infer_states(obs)
    trial_q_s.append(q_s)  # Log beliefs about states 
    
    # Update A matrix after initial observation
    update_matrix(agent, "update_A", pA_history, learning_A, obs)
    
    # Store updated likelihood
    trial_likelihoods.append(copy.deepcopy(agent.A))
    
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
        trial_actions.append(action)  # Log action right after printing it
        
        # Environment step
        obs = env.step(action)
        print(f"Observation: sees {colors[int(obs[0])]} at {locations[int(obs[1])]}, feedback: {feedback[int(obs[2])]}")
        trial_obs.append(obs)  # Log observation right after printing it

        # Update agent's beliefs
        q_s = agent.infer_states(obs)
        print(f"Beliefs: rule={q_s[0].round(2)}, color={q_s[1].round(2)}")
        print(f"        location={q_s[2].round(2)}, choice={q_s[3].round(2)}")
        trial_q_s.append(q_s)  # Log beliefs right after printing them
        
        # Update A matrix after observation
        update_matrix(agent, "update_A", pA_history, learning_A, obs)
        
        # Store updated likelihood
        trial_likelihoods.append(copy.deepcopy(agent.A))
        
    obs_history.append(trial_obs)
    action_history.append(trial_actions)
    q_s_history.append(trial_q_s)
    likelihood_history.append(trial_likelihoods)

# %% Analyze A matrix evolution at top location 
plotting_A = False
if learning_A and plotting_A:
    print("\nA matrix values at top location (every 100 trials):")

    for trial in range(0, n_trials, 100):
        print(f"\nTrial {trial}:")
        pA = pA_history[trial]  # what modality (0)
        A = utils.norm_dist_obj_arr(pA)
        
        # Plot what modality (0) for each rule at the top location (1) with choice red (0)
        # This shows how different rules lead to different color expectations at the top location
        loc = 0
        for r in range(env.num_rules):
            plot_likelihood(A[0][:,r,:,loc,0], f'rule {r}, loc {loc}, choice {0}')

# %% # Calculate performance statistics
total_choices = 0
correct_choices = 0
performance_history = []  # Track overall performance per trial
performance_history_per_rule = [[], [], []]  # Track performance for each rule

for trial_idx, trial_obs in enumerate(obs_history):
    trial_correct = 0
    trial_total = 0
    
    # Get the true rule for this trial from our stored rules
    true_rule = trial_rules[trial_idx]
    
    for obs in trial_obs:
        if obs[2] == 1:  # Correct feedback
            correct_choices += 1
            trial_correct += 1
        if obs[2] != 0:  # Any feedback (excluding neutral)
            total_choices += 1
            trial_total += 1
    
    # Calculate trial performance
    if trial_total > 0:
        trial_performance = trial_correct / trial_total
        performance_history.append(trial_performance)
        performance_history_per_rule[true_rule].append(trial_performance)
    else:
        performance_history.append(0)
        performance_history_per_rule[true_rule].append(0)

print(f"\nPerformance Summary:")
print(f"Correct choices: {correct_choices}/{total_choices} ({(correct_choices/total_choices)*100:.2f}%)")

# Plot overall performance
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(range(len(performance_history)), performance_history, 'b-', label='Performance')
plt.plot(range(len(performance_history)), 
         np.convolve(performance_history, np.ones(20)/20, mode='same'),
         'r-', label='Moving Average (20 trials)')
plt.xlabel('Trial Number')
plt.ylabel('Performance (Correct/Total Choices)')
plt.title('Overall Learning Performance')
plt.legend()
plt.grid(True)

# Plot performance for each rule
rule_names = ['Left Rule', 'Top Rule', 'Right Rule']
colors = ['red', 'green', 'blue']

for rule_idx in range(env.num_rules):
    plt.subplot(2, 2, rule_idx + 2)
    perf = performance_history_per_rule[rule_idx]
    if len(perf) > 0:  # Only plot if we have data for this rule
        plt.plot(range(len(perf)), perf, '-', color=colors[rule_idx], label='Performance')
        plt.plot(range(len(perf)), 
                np.convolve(perf, np.ones(20)/20, mode='same'),
                'k-', label='Moving Average (20 trials)')
    plt.xlabel('Trial Number')
    plt.ylabel('Performance (Correct/Total Choices)')
    plt.title(f'Learning Performance - {rule_names[rule_idx]}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# %%
