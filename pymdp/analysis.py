"""
Tools for analyzing and visualizing active inference agent behavior.

__author__: Lancelot Da Costa
"""

import jax.numpy as jnp
import jax.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import mediapy
from PIL import Image
import os
import numpy as np
import io
from matplotlib.gridspec import GridSpec
from pymdp.maths import smooth_data
from pymdp.envs.rollout import is_multi_trial, get_info_trial
from pymdp.utils import flatten_multi_trial_tensor_list, flatten_multi_trial_tensor, add_trial_boundary_lines
from warnings import warn

def analyze_rollout(info, agent, env, render=True, plot=True, print=True):
    if plot: plot_preferences(agent, env)
    if render: render_rollout(env, info)
    if plot: plot_beliefs(info, env)
    if print: print_rollout(info, env)
    # List of further possible analysis functions 
    # print_experiment_setup(info)
    # print_rollout(info)
    # print_parameter_learning(info, learning_config, verbose=False)
    # plot_parameter_learning(info, learning_config, env)
    # pe_analysis = compute_prediction_errors(info)
    # plot_prediction_errors(pe_analysis, yscale='log', smoothing=None, num_trials=num_trials)
    # plot_preferences(agent, env)
    # plot_model_comparison
    # print_initial_state(info)
    # # initial_state
    # render_rollout(env, info, fps=10)
    # plot_beliefs(info, env)

def plot_prediction_errors(pe_analysis: Dict, 
    title: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 5), 
    yscale: str = 'log', 
    smoothing: Optional[int] = None, 
    num_trials: Optional[int] = None, 
    trial_lines: Optional[bool] = True) -> plt.Figure:
    """
    Plot prediction error metrics from the output of compute_prediction_errors.

    Parameters
    ----------
    pe_analysis : Dict
        Dictionary containing prediction error metrics with keys:
        - 'pred_error': prediction error at each timestep
        - 'complexity': complexity at each timestep
        - 'neg_accuracy': negative accuracy at each timestep
        - 'complexity_l2': complexity measured with L2 norm (instead of KL divergence) at each timestep
        - 'pe_accumulated': cumulative sum of prediction errors
    title : str, optional
        Title for the plot. If None, no title is shown.
    figsize : tuple, optional
        Figure size as (width, height). Default is (10, 5).
    smoothing : int, optional
        Window size for moving average smoothing. If None or <= 1, no smoothing is applied.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot
    """
    # Apply smoothing if requested, creating a copy of the data
    pe_data = smooth_data(pe_analysis, window_size=smoothing)
    
    fig = plt.figure(figsize=figsize)

    # Add vertical lines at the beginning of each trial if there are trials
    if trial_lines and num_trials is not None and num_trials > 1:
        n_timesteps = len(pe_data["pred_error"])
        add_trial_boundary_lines(plt.gca(), n_timesteps, num_trials)

    plt.plot(pe_data["complexity_l2"], label='L2 norm Complexity', alpha=0.4)
    plt.plot(pe_data["complexity"], label='Complexity', alpha=0.7)
    plt.plot(pe_data["neg_accuracy"], label='Negative accuracy', alpha=0.7)
    plt.plot(pe_data["pred_error"], label='Prediction error', alpha=1.0)
    plt.plot(pe_data["pe_accumulated"], label='Accumulated prediction errors')
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('nats')
    plt.yscale(yscale)
    # plt.grid(True)
    
    if title is not None:
        plt.title(title)
    
    #return fig

def plot_model_comparison(pe_analyses,
 labels=None, 
 figsize: Tuple[int, int] = (15, 12), 
 alpha: float = 0.7, 
 lw: float = 1.0, 
 yscale: str = 'log', 
 smoothing: Optional[int] = None, 
 num_trials: Optional[int] = None, 
 trial_lines: bool = True) -> plt.Figure:
    """
    Create comparison plots between multiple models showing their prediction error metrics.

    Parameters
    ----------
    pe_analyses : Union[List[Dict], Tuple[Dict], Dict, Tuple[Dict, Dict]]
        Either a single prediction error analysis dictionary, a list/tuple of prediction error analysis
        dictionaries from compute_prediction_errors, or two dictionaries for backward compatibility.
    labels : Union[List[str], Tuple[str], None], optional
        Labels for the models in the plots. If None, will use 'Model 1', 'Model 2', etc.
    figsize : Tuple[int, int], optional
        Figure size as (width, height). Default is (15, 12)
    alpha : float, optional
        Alpha value for transparency in plots. Default is 0.7
    lw : float, optional
        Line width for plots. Default is 1.0
    yscale : str, optional
        Scale for y-axis. Default is 'log'
    smoothing : int, optional
        Window size for moving average smoothing. If None or <= 1, no smoothing is applied.
    num_trials : int, optional
        Number of trials in the data. If provided along with trial_lines=True, 
        vertical lines will be added at trial boundaries.
    trial_lines : bool, optional
        Whether to show vertical lines at trial boundaries. Default is True.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the subplots
    """
    # Handle different input types for backward compatibility
    if isinstance(pe_analyses, dict):
        # Single PE analysis
        pe_analyses = [pe_analyses]

    # Apply smoothing to all analyses if requested
    pe_analyses = [smooth_data(pe, window_size=smoothing) for pe in pe_analyses]

    # Handle labels
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(pe_analyses))]
    elif isinstance(labels, str):
        # Single label provided as string
        labels = [labels]
    
    # Ensure we have a label for each model
    if len(labels) < len(pe_analyses):
        # Add generic labels for any missing
        labels.extend([f'Model {i+1}' for i in range(len(labels), len(pe_analyses))])

    # Create figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Add vertical lines at the beginning of each trial if requested and there are trials
    if trial_lines and num_trials is not None and num_trials > 1:
        n_timesteps = len(pe_analyses[0]["pred_error"])
        add_trial_boundary_lines(ax1, n_timesteps, num_trials)
        add_trial_boundary_lines(ax2, n_timesteps, num_trials)
        add_trial_boundary_lines(ax3, n_timesteps, num_trials)
        add_trial_boundary_lines(ax4, n_timesteps, num_trials)

    # Plot 1: Accumulated Prediction Error
    for i, pe_analysis in enumerate(pe_analyses):
        ax1.plot(pe_analysis["pe_accumulated"], label=labels[i], alpha=alpha, lw=lw)
    ax1.set_title('Accumulated Prediction Error')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Accumulated PE (nats)')
    ax1.legend()
    ax1.set_yscale(yscale)

    # Plot 2: Prediction Error
    for i, pe_analysis in enumerate(pe_analyses):
        ax2.plot(pe_analysis["pred_error"], label=labels[i], alpha=alpha, lw=lw)
    ax2.set_title('Prediction Error')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('PE (nats)')
    ax2.legend()
    ax2.set_yscale(yscale)

    # Plot 3: Complexity
    for i, pe_analysis in enumerate(pe_analyses):
        ax3.plot(pe_analysis["complexity"], label=labels[i], alpha=alpha, lw=lw)
    ax3.set_title('Complexity')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Complexity (nats)')
    ax3.legend()
    ax3.set_yscale(yscale)

    # Plot 4: Negative Accuracy
    for i, pe_analysis in enumerate(pe_analyses):
        ax4.plot(pe_analysis["neg_accuracy"], label=labels[i], alpha=alpha, lw=lw)
    ax4.set_title('Negative Accuracy')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Negative Accuracy (nats)')
    ax4.legend()
    ax4.set_yscale(yscale)

    plt.tight_layout()
    #return fig

def print_experiment_setup(info):
    
    multi_trials, num_trials = is_multi_trial(info)
    info = get_info_trial(info, 0, verbose=False)
    
    # Extract variables from info dictionary
    observations = info["observation"] # List of modality arrays, shape: (T+1, batch_size, 1)
    beliefs = info["qs"] # List of factor arrays, shape: (T+1, batch_size, 1, num_states[f])
    policies = info["qpi"] # Shape: (T+1, batch_size, num_policies)
    actions = info["action"] # Shape: (T+1, batch_size, control_factors)
    empirical_priors = info["empirical_prior"] # List of prior belief arrays for each state factor

    # Get the environment labels
    labels = info['env'].get_labels()
    
    # Get dimensions
    num_timesteps = observations[0].shape[0] # Number of timesteps including initial (t=0)
    num_state_factors = len(labels["state_factors"])
    num_obs_modalities = len(labels["observation_modalities"])
    num_control_factors = len(labels["control_factors"])
    
    # Get labels for each component
    state_factor_names = list(labels["state_factors"].keys())
    observation_modality_names = list(labels["observation_modalities"].keys())
    control_factor_names = list(labels["control_factors"].keys())
    
    # Print experiment setup
    print("\n=== Experiment Setup ===")
    
    # Trial information
    if multi_trials: print(f"Multi-trial experiment: {num_trials} trials")
    else: print("Single-trial experiment")
    print(f"Number of timesteps per trial: {num_timesteps-1}")  # -1 because includes initial observation
    print(f"Batch size: {observations[0].shape[1]}")
    print(f"Number of policies: {policies.shape[-1]}")
    print(f"State factors: {state_factor_names}")
    print(f"Observation modalities: {observation_modality_names}")
    print(f"Control factors: {control_factor_names}")
    #TODO: add more info such as planning horizon of agent

def print_rollout(info, batch_idx=0, timesteps='all', trials='all'):
    """Print a human-readable version of the rollout using environment labels.
    
    Parameters
    ----------
    info : Dict
        Dictionary containing rollout information with keys:
        - 'observation': List of observation arrays from environment
        - 'qs': List of belief arrays for each state factor
        - 'qpi': Policy distributions
        - 'action': Selected actions
        - 'empirical_prior': Prior beliefs before observations
    batch_idx : int, optional
        Batch index to print for, by default 0
    timesteps : str, int, list, optional
        Timesteps to print, by default 'all'
    trials : str, int, list, optional
        Trials to print (for multi-trial rollouts), by default 'all'
    """
    # Check if multi-trial, and call recursively for each trial if so
    is_multi, num_trials = is_multi_trial(info)
    if is_multi:
        # Process trials parameter
        if trials == 'all':
            trials_to_print = list(range(num_trials))
        elif isinstance(trials, int):   
            trials_to_print = [trials] if trials in range(num_trials) else []
        elif isinstance(trials, (list, range)):
            trials_to_print = [t for t in trials if t in range(num_trials)]
        else:
            raise ValueError(f"Invalid trials parameter type: {type(trials)}")
                 
        # Call print rollout for each trial in trials_to_print
        for trial_idx in trials_to_print:
            info_trial = get_info_trial(info, trial_idx, verbose=False)
            print(f"\n=== Trial {trial_idx} ==================")
            print_rollout(info_trial, batch_idx=batch_idx, timesteps=timesteps)
        return

    # Extract variables from info dictionary
    observations = info["observation"] # List of modality arrays, shape: (T+1, batch_size, 1)
    beliefs = info["qs"] # List of factor arrays, shape: (T+1, batch_size, 1, num_states[f])
    policies = info["qpi"] # Shape: (T+1, batch_size, num_policies)
    actions = info["action"] # Shape: (T+1, batch_size, control_factors)
    empirical_priors = info["empirical_prior"] # List of prior belief arrays for each state factor

    # Get the environment labels
    labels = info['env'].get_labels()
    
    # Get dimensions
    num_timesteps = observations[0].shape[0] # Number of timesteps including initial (t=0)
    num_state_factors = len(labels["state_factors"])
    num_obs_modalities = len(labels["observation_modalities"])
    num_control_factors = len(labels["control_factors"])
    
    # Get labels for each component
    state_factor_names = list(labels["state_factors"].keys())
    observation_modality_names = list(labels["observation_modalities"].keys())
    control_factor_names = list(labels["control_factors"].keys())
    
    def format_state_dist(factor_idx, state_probs):
        """Helper to format state distribution nicely using labels"""
        factor_name = state_factor_names[factor_idx]
        state_labels = labels["state_factors"][factor_name]
        
        # Create formatted string of probabilities with labels
        probs_str = ", ".join([f"{state_labels[i]}: {float(prob):.3f}" 
                             for i, prob in enumerate(state_probs)])
        return f"[{probs_str}]"
    
    # Process timesteps parameter
    if timesteps == 'all':
        timesteps_to_print = list(range(num_timesteps))
    elif isinstance(timesteps, int):
        timesteps_to_print = [timesteps] if timesteps in range(num_timesteps) else []
    elif isinstance(timesteps, (list, range)):
        timesteps_to_print = [t for t in timesteps if t in range(num_timesteps)]
    else:
        raise ValueError(f"Invalid timesteps parameter type: {type(timesteps)}")
    
    if 0 in timesteps_to_print:
        # Print initial timestep info
        print("\n--- Initial Timestep (t=0) ---")
        
        # Print initial beliefs for each state factor
        for f in range(num_state_factors):
            print(f"Prior beliefs ({state_factor_names[f]}): ", 
                format_state_dist(f, empirical_priors[f][0, batch_idx]))
        
        # Print initial observations for each modality
        for m in range(num_obs_modalities):
            modality_name = observation_modality_names[m]
            obs_idx = int(observations[m][0, batch_idx, 0])
            obs_label = labels["observation_modalities"][modality_name][obs_idx]
            print(f"Observation ({modality_name}): [{obs_label}]")
        
        # Print posterior beliefs for each factor
        for f in range(num_state_factors):
            print(f"Posterior beliefs ({state_factor_names[f]}): ", 
                format_state_dist(f, beliefs[f][0, batch_idx, 0]))
        
        # Remove 0 from timesteps_to_print
        timesteps_to_print.remove(0)

    # Print trajectory
    for t in timesteps_to_print:
        print(f"\n--- Timestep {t} ---")
        
        # Print policy distribution in a concise format
        # Get top 5 policies by probability
        top_policies = sorted([(i, float(p)) for i, p in enumerate(policies[t, batch_idx])], 
                           key=lambda x: x[1], reverse=True)[:5]
        
        # Filter to only include those with probability > 0.01 (1%)
        top_policies = [(i, p) for i, p in top_policies if p > 0.01]
        
        # Calculate sum of remaining policies
        remaining_sum = sum([float(p) for i, p in enumerate(policies[t, batch_idx]) 
                          if i not in [idx for idx, _ in top_policies]])
        
        # Format policy strings
        policy_strs = [f"P{idx}:{prob:.3f}" for idx, prob in top_policies]
        if remaining_sum > 0.001:  # Only show 'Others' if there's a meaningful remainder
            policy_strs.append(f"Others:{remaining_sum:.3f}")
        
        print(f"Policies: {', '.join(policy_strs)}")
 
        # Print actions for each control factor and their consequences
        for c in range(num_control_factors):
            control_name = control_factor_names[c]
            action_idx = int(actions[t, batch_idx, c].item())
            action_label = labels["control_factors"][control_name][action_idx]
            print(f"Action ({control_name}): [{action_label}]")
        
        # Print predicted next state (empirical prior) for each factor
        for f in range(num_state_factors):
            print(f"Predicted state ({state_factor_names[f]}): ", 
                  format_state_dist(f, empirical_priors[f][t, batch_idx]))
        
        # Print actual observations for each modality
        for m in range(num_obs_modalities):
            modality_name = observation_modality_names[m]
            obs_idx = int(observations[m][t, batch_idx, 0].item())
            obs_label = labels["observation_modalities"][modality_name][obs_idx]
            print(f"Observation ({modality_name}): [{obs_label}]")
        
        # Print posterior beliefs for each factor
        for f in range(num_state_factors):
            print(f"Posterior beliefs ({state_factor_names[f]}): ", 
                  format_state_dist(f, beliefs[f][t, batch_idx, 0]))


def render_rollout(env, info, save_gif=False, filename=None, fps=1):
    """Render a video of the agent's trajectory through any environment that implements a render method.
    
    This function iterates through the rollout information and renders each timestep using the 
    environment's built-in render method. It works with any environment that implements the
    standard render(mode="rgb_array", observations=observations_t) interface.
    
    Parameters
    ----------
    env : Env
        Environment instance (TMaze, SimplestEnv, etc.)
    info : dict
        Dictionary containing rollout information, as returned by the rollout function
    save_gif : bool, optional
        Whether to save the animation as a gif, by default False
    filename : str, optional
        Path to save the gif if save_gif is True, by default None
    fps : int, optional
        Frames per second for the rendered video, by default 1
        
    Returns
    -------
    None
        Displays the animation in the notebook or saves it as a gif
    """

    # Check if multi-trial
    is_multi, _ = is_multi_trial(info)
    if is_multi: 
        warn("render_rollout is currently not implemented for multi-trial rollouts.")
        return 0
    
    # Get the number of timesteps in the rollout
    num_timesteps = info["observation"][0].shape[0]
    
    # Get the number of observation modalities
    num_modalities = len(info["observation"])
    
    frames = [None] * num_timesteps
    for t in range(num_timesteps):  # iterate over timesteps
        # Prepare observations for current timestep
        observations_t = [info["observation"][mod_idx][t] for mod_idx in range(num_modalities)]
        
        # Call the environment's render method
        frame = env.render(mode="rgb_array", observations=observations_t)
        frames[t] = jnp.asarray(frame, dtype=jnp.uint8)
        plt.close()  # close the figure to prevent memory leak
    
    # Convert frames to array and display video
    frames = jnp.array(frames, dtype=jnp.uint8)
    mediapy.show_video(frames, fps=fps)
    
    # Save as gif if requested
    if save_gif:
        if filename is None:
            raise ValueError("If save_gif is True, a filename must be provided")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # milliseconds per frame
            loop=0
        )

def plot_beliefs(info, env=None, save_gif=False, filename=None, figsize=None, fps=1, batch_idx=0):
    """Create a GIF animation showing the evolution of beliefs for each state factor over time.
    
    Parameters
    ----------
    info : dict
        Rollout info dictionary with 'qs' for belief history
    env : Env, optional
        Environment instance, used to get labels for state factors
    save_gif : bool, optional
        Whether to save the animation as a gif, by default False
    filename : str, optional
        Path to save the gif if save_gif is True, by default None
    figsize : tuple, optional
        Figure size as (width, height), by default None (auto-calculated)
    fps : int, optional
        Frames per second for the rendered video, by default 1
    batch_idx : int, optional
        Batch index to plot, by default 0
        
    Returns
    -------
    None
        Displays the animation in the notebook or saves it as a gif
    """
    # Check if multi-trial
    is_multi, _ = is_multi_trial(info)
    if is_multi: 
        warn("plot_beliefs is currently not implemented for multi-trial rollouts.")
        return 0

    # Extract beliefs and num_state_factors
    beliefs = info['qs']
    num_state_factors = len(beliefs)
    num_timesteps = beliefs[0].shape[0]
    
    # Get labels from environment if available
    if env is not None:
        state_factor_names = list(env.labels['state_factors'].keys())
        state_labels = [env.labels['state_factors'][factor] for factor in state_factor_names]
    else:
        # No environment provided
        state_factor_names = [f"Factor {i}" for i in range(num_state_factors)]
        state_labels = [[f"State {i}" for i in range(beliefs[f].shape[-1])] for f in range(num_state_factors)]
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (4 * num_state_factors, 4)
    
    # Create frames for each timestep
    frames = []
    
    for t in range(num_timesteps):
        # Create a new figure for this timestep
        fig, axes = plt.subplots(1, num_state_factors, figsize=figsize)
        if num_state_factors == 1:
            axes = [axes]  # Handle case of single state factor
        
        # Set a title for the figure showing the timestep
        fig.suptitle(f'Beliefs at timestep {t}', fontsize=16)
        
        # Plot beliefs for each state factor
        for f in range(num_state_factors):
            # Get number of states for this factor
            num_states = beliefs[f].shape[-1]
            
            # Plot beliefs for this factor at this timestep
            axes[f].bar(range(num_states), beliefs[f][t, batch_idx, 0])
            axes[f].set_title(f'{state_factor_names[f]}')
            axes[f].set_xticks(range(num_states))
            axes[f].set_xticklabels(state_labels[f], rotation=45, ha='right')
            axes[f].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Convert the figure to an image and add to frames
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')  # Convert to RGB format directly
        frame = np.array(img)
        frames.append(frame)
        
        plt.close(fig)  # Close the figure to free memory
    
    # Convert frames to array for mediapy
    frames = np.array(frames)
    
    # Display animation with mediapy
    if not save_gif:
        mediapy.show_video(frames, fps=fps)
    
    # Save as gif if requested
    if save_gif:
        if filename is None:
            raise ValueError("If save_gif is True, a filename must be provided")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # milliseconds per frame
            loop=0
        )
        print(f"GIF saved to {filename}")

def plot_agent_preferences(agent, env=None, figsize=None, show=True, batch_idx=0):
    """Plot the agent's preferences for each modality (Note: does not currently support time-dependent preferences)
    
    Parameters
    ----------
    agent : Agent
        Agent instance with preferences (C array)
    env : Env, optional
        Environment instance, used to get labels for observation modalities
    figsize : tuple, optional
        Figure size as (width, height), by default None (auto-calculated)
    show : bool, optional
        Whether to call plt.show(), by default True
        
    Returns
    -------
    plt.Figure
        Matplotlib figure containing the preference plots
    """
    if not hasattr(agent, 'C') or agent.C is None:
        raise ValueError("Agent does not have preferences (C array)")
    
    num_modalities = len(agent.C)
    
    # Get observation modality labels from environment if available
    if env is not None:
        env_labels = env.labels
        if env_labels is not None and 'observation_modalities' in env_labels:
            modality_names = list(env_labels['observation_modalities'].keys())
            modality_labels = [env_labels['observation_modalities'][modality] for modality in modality_names]
    else:
        # No environment provided
        modality_names = [f"Modality {i}" for i in range(num_modalities)]
        modality_labels = [None] * num_modalities
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (4 * num_modalities, 4)
    
    # Create figure
    fig, axes = plt.subplots(1, num_modalities, figsize=figsize)
    if num_modalities == 1:
        axes = [axes]  # Handle case of single modality
    
    # Plot preferences for each modality
    for m in range(num_modalities):
        # Get preferences for this modality
        C_m = agent.C[m]
        if C_m.ndim == 1:
            # C is just a vector for this modality
            preferences = nn.softmax(C_m)
        else:
            # C might be a matrix (e.g., batches)
            # TODO: note there might be an indexing clash here if preferences are time-dependent. Do not currently support this
            preferences = nn.softmax(C_m[batch_idx])
        
        # Get x-tick labels for this modality
        if modality_labels[m] is not None:
            x_labels = modality_labels[m]
        else:
            x_labels = [f"Obs {i}" for i in range(len(preferences))]
        
        # Plot preferences for this modality
        axes[m].bar(range(len(preferences)), preferences)
        axes[m].set_title(f'Preferences: {modality_names[m]}')
        axes[m].set_xticks(range(len(preferences)))
        axes[m].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[m].set_ylim(0, 1)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return plt

def print_parameter_learning(info, learning_config, env=None, verbose=False, batch_idx=0):
    """Print and analyze parameter learning results in an environment-agnostic way.
    
    Parameters
    ----------
    info : Dict
        Dictionary containing agent learning information with keys like 'agent'
    learning_config : object
        Configuration specifying which parameters are being learned.
        Should have boolean attributes: learn_A, learn_B, learn_D
    env : Env, optional
        Environment instance containing true parameters for comparison, by default None
    verbose : bool, optional
        Whether to print learned parameters at each timestep, by default False
    batch_idx : int, optional
        Batch index to analyze, by default 0
    """
    # Check if multi-trial and produce a summary (beginning of initial trial and end of final trial)
    is_multi, _ = is_multi_trial(info)
    if is_multi and verbose:
        raise ValueError('Verbose option not implemented for multi trial parameter learning')

    # Get number of timesteps
    if not is_multi: num_timesteps = info["agent"].A[0].shape[0]
    else: num_timesteps = info["agent"].A[0].shape[1]
    
    # Extract labels from environment
    modality_names = list(info["env"].labels['observation_modalities'].keys())
    factor_names = list(info["env"].labels['state_factors'].keys())
    control_factor_names = list(info["env"].labels['control_factors'].keys())
    
    # Helper function to round array values to 2 decimal places for display
    def round_array(arr, decimals=2):
        # Convert to numpy array if it's not already
        import numpy as np
        arr_np = np.array(arr)
        return np.round(arr_np, decimals)
    
    # Print A parameter learning if applicable
    if learning_config.learn_A:
        print('\n==== Parameter A learning ====')
        for m, modality in enumerate(modality_names):
            print(f"\nModality: {modality}")
            if not is_multi:
                # For single-trial data, show first and last timesteps
                print(f"Initial A matrix:\n{round_array(info['agent'].A[m][0, batch_idx])}") # First timestep
                print(f"Final A matrix:\n{round_array(info['agent'].A[m][-1, batch_idx])}") # Last timestep
            else:
                # For multi-trial data, we have an additional trial dimension
                # First index is trial, second is timestep within trial
                print(f"Initial A matrix (start of first trial):\n{round_array(info['agent'].A[m][0, 0, batch_idx])}")
                print(f"Final A matrix (end of last trial):\n{round_array(info['agent'].A[m][-1, -1, batch_idx])}")
            if env is not None: # Useful for debugging/comparison:
                print(f"True A matrix:\n{round_array(env.params['A'][m])}")
            if verbose:
                print(f"\nLearning progression for A matrix (Modality {modality}):")
                for t in range(num_timesteps):
                    print(f"t={t}:\n{round_array(info['agent'].A[m][t, batch_idx])}")
    
    # Print B parameter learning if applicable
    if learning_config.learn_B:
        print('\n==== Parameter B learning ====')
        
        # Get B_action_dependencies - which control factors affect each state factor
        B_action_dependencies = info["agent"].B_action_dependencies
        
        for f, factor in enumerate(factor_names):
            print(f"\nState Factor: {factor}")
            
            # Get number of actions for this state factor
            # Note: In pymdp, B tensors with multiple control factors are flattened during agent initialization
            # (see agent._flatten_B_action_dims). So the last dimension already represents the flattened action space,
            # combining all control factors affecting this state factor. The get_action_indices function maps
            # from this flattened index back to the individual control factor action indices.
            num_actions = info["agent"].B[f][0, batch_idx].shape[-1]
            
            # Get control factors that affect this state factor
            control_indices = B_action_dependencies[f]
            
            # Get control factor names and their action labels
            control_factor_actions = []
            for idx in control_indices:
                if idx < len(control_factor_names):
                    control_name = control_factor_names[idx]
                    control_factor_actions.append((control_name, info["env"].labels['control_factors'][control_name]))
            
            if control_factor_actions:
                control_names = [name for name, _ in control_factor_actions]
                print(f"Control factors affecting this state factor: {', '.join(control_names)}")
            
            # For each action in the B tensor
            for a in range(num_actions):
                # Get action label(s) based on control factor dependencies
                if not control_factor_actions:
                    # No control factors - use generic label
                    action_label = f"Action {a}"
                elif len(control_factor_actions) == 1:
                    # Single control factor - use its action label if available
                    control_name, actions = control_factor_actions[0]
                    action_label = actions[a] if a < len(actions) else f"Action {a}"
                else:
                    # Multiple control factors - map flat index to action combinations
                    action_indices = _get_action_indices(a, control_factor_actions)
                    action_labels = []
                    
                    for i, (control_name, actions) in enumerate(control_factor_actions):
                        if i < len(action_indices):
                            action_idx = action_indices[i]
                            label = actions[action_idx] if action_idx < len(actions) else f"Action {action_idx}"
                            action_labels.append(f"{control_name}:{label}")
                    
                    action_label = ", ".join(action_labels) if action_labels else f"Action {a}"
                    action_label = f"[{action_label}]"
                
                # Print B matrices with rounded values - single trial case
                if not is_multi:
                    print(f"Initial B matrix under action {action_label}:\n{round_array(info['agent'].B[f][0, batch_idx, ..., a])}") #beginning of trial
                    print(f"Final B matrix under action {action_label}:\n{round_array(info['agent'].B[f][-1, batch_idx, ..., a])}") #end of trial
                    # diff_B = env.params['B'][f][..., a] - info['agent'].B[f][-1, batch_idx, ..., a]
                    # print(f"Absolute difference (true - final) B matrix under action {action_label}:\n{round_array(jnp.abs(diff_B), decimals=1)}")
                else:
                    # Multi-trial case - first index is trial, second is timestep within trial
                    print(f"Initial B matrix under action {action_label} (start of first trial):\n{round_array(info['agent'].B[f][0, 0, batch_idx, ..., a])}")
                    print(f"Final B matrix under action {action_label} (end of last trial):\n{round_array(info['agent'].B[f][-1, -1, batch_idx, ..., a])}")
                    # print(f"True B matrix under action {action_label}:\n{round_array(info['env'].params['B'][f][0,0,..., a])}")
                if env is not None: # Useful for debugging/comparison:
                    print(f"True B matrix under action {action_label}:\n{round_array(env.params['B'][f][..., a])}")
                if verbose:
                    print(f"\nLearning progression for B matrix (Factor {factor}, Action {action_label}):")
                    for t in range(num_timesteps):
                        print(f"t={t}:\n{round_array(info['agent'].B[f][t, batch_idx, ..., a])}")
    
    # Print D parameter learning if applicable
    if learning_config.learn_D:
        print('\n==== Parameter D learning ====')
        for f, factor in enumerate(factor_names):
            print(f"\nState Factor: {factor}")
            if not is_multi:
                print(f"Initial D matrix:\n{round_array(info['agent'].D[f][0, batch_idx])}") #beginning of trial
                print(f"Final D matrix:\n{round_array(info['agent'].D[f][-1, batch_idx])}") #end of trial
            else:
                print(f"Initial D matrix (start of first trial):\n{round_array(info['agent'].D[f][0,0, batch_idx])}") #beginning of first trial
                print(f"Final D matrix (end of last trial):\n{round_array(info['agent'].D[f][-1, -1, batch_idx])}") #end of last trial
            if env is not None: # Useful for debugging/comparison:
                print(f"True D matrix:\n{round_array(env.params['D'][f])}")
            if verbose:
                print(f"\nLearning progression for D matrix (Factor {factor}):")
                for t in range(num_timesteps):
                    print(f"t={t}, qD: {round_array(info['agent'].pD[f][t, batch_idx])}, D: {round_array(info['agent'].D[f][t, batch_idx])}")

def _get_action_indices(flat_index, control_factor_actions):
    """Convert a flat action index to individual action indices for multiple control factors.
    #TODO: consider moving this to a utils file
    
    Parameters
    ----------
    flat_index : int
        The flattened action index
    control_factor_actions : list
        List of tuples (control_name, actions) for each control factor
    
    Returns
    -------
    list
        List of action indices, one for each control factor
    """
    # Get the number of actions for each control factor
    num_actions_per_factor = [len(actions) for _, actions in control_factor_actions]
    
    # Calculate the product of action counts for each control factor
    action_products = []
    for i in range(len(num_actions_per_factor) - 1, -1, -1):
        product = 1
        for j in range(i + 1, len(num_actions_per_factor)):
            product *= num_actions_per_factor[j]
        action_products.insert(0, product)
    
    # Convert flat index to action indices
    remaining_idx = flat_index
    action_indices = []
    
    for product in action_products:
        if product > 0:  # Avoid division by zero
            action_idx = remaining_idx // product
            remaining_idx = remaining_idx % product
            action_indices.append(action_idx)
    
    return action_indices

def plot_parameter_learning(info, learning_config, env, yscale='linear', trial_lines: Optional[bool] = True, num_trials: Optional[int] = None):
    
    """Plot the agent's learning progress for parameters (A, B, D) over time.
    
    This function generates plots showing the distance between the agent's learned
    parameters and the environment's true parameters over time. Only parameters
    that are being learned (as specified in learning_config) will be plotted.
    
    Parameters
    ----------
    info : Dict
        Dictionary containing rollout information with parameter history in info["agent"]
    learning_config : object
        Configuration specifying which parameters are being learned.
        Should have boolean attributes: learn_A, learn_B, learn_D
    env : Env
        Environment instance containing true parameters
    yscale : str, optional
        Scale for y-axis, e.g. 'linear' or 'log', by default 'linear'
    num_trials : int, optional
        Number of trials in the data. If provided along with trial_lines=True, 
        vertical lines will be added at trial boundaries.
    trial_lines : bool, optional
        Whether to show vertical lines at trial boundaries. Default is True.

    Returns
    -------
    plt : matplotlib.pyplot
        The pyplot object with the generated plots
    """

    # Check if multi-trial
    multi_trials, num_trials = is_multi_trial(info)
    
    # Get agent from info dictionary, get its tensors, and flatten them along the time dimension if multi-trial
    agent = info["agent"]
    A_flat = flatten_multi_trial_tensor_list(agent.A, multi_trials)
    B_flat = flatten_multi_trial_tensor_list(agent.B, multi_trials)
    D_flat = flatten_multi_trial_tensor_list(agent.D, multi_trials)
    #TODO: Optionally, one could remove dependency on env of this function by extracting its tensors directly from info.
    #This would be like this if multi_trial (indexing at zeroth trial and timestep)
    # A_true = [combined_info["env"].params["A"][m][0,0] for m in range(len(combined_info["env"].params["A"]))] # this is the same as env.params["A"]
    # B_true = [combined_info["env"].params["B"][f][0,0] for f in range(len(combined_info["env"].params["B"]))] # this is the same as env.params["B"]
    # D_true = [combined_info["env"].params["D"][f][0,0] for f in range(len(combined_info["env"].params["D"]))] # this is the same as env.params["D"]
    # And would have one less zero indexing in the absence of trials (indexing at zeroth timestep only)

    # Create figure with appropriate number of subplots
    n_plots = learning_config.learn_A + learning_config.learn_B + learning_config.learn_D
    fig = plt.figure(figsize=(5*n_plots, 5))
    gs = GridSpec(1, n_plots, figure=fig)
    plot_idx = 0
    
    # Plot each parameter type if it's being learned
    if learning_config.learn_A:
        _plot_matrix_learning(
            fig.add_subplot(gs[0, plot_idx]),
            A_flat, env.params["A"],
            list(env.labels['observation_modalities'].keys()),
            'A Matrix Learning (Observations)',
            'Linf distance to true A',
            yscale=yscale,
            num_trials=num_trials,
            trial_lines=trial_lines
        )
        plot_idx += 1
    
    if learning_config.learn_B:
        _plot_matrix_learning(
            fig.add_subplot(gs[0, plot_idx]),
            B_flat, env.params["B"],
            list(env.labels['state_factors'].keys()),
            'B Matrix Learning (Transitions)',
            'Linf distance to true B',
            yscale=yscale,
            num_trials=num_trials,
            trial_lines=trial_lines
        )
        plot_idx += 1
    
    if learning_config.learn_D:
        _plot_matrix_learning(
            fig.add_subplot(gs[0, plot_idx]),
            D_flat, env.params["D"],
            list(env.labels['state_factors'].keys()),
            'D Matrix Learning (Initial States)',
            'Linf distance to true D',
            yscale=yscale,
            num_trials=num_trials,
            trial_lines=trial_lines
        )
    
    plt.tight_layout()
    plt.show()
    return plt


def _plot_matrix_learning(ax, 
agent_tensor, 
env_tensor, 
labels, 
title, 
ylabel, 
yscale='linear', 
num_trials= None, 
trial_lines: Optional[bool] = True):
    #TODO: note this works only for batch_size==1
    """Helper function to plot learning curves for a set of matrices.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    agent_tensor : List[Array]
        List of parameter history arrays from the agent
    env_tensor : List[Array]
        List of true parameter arrays from the environment
    labels : List[str]
        List of labels for each matrix (e.g., modality or factor names)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    yscale : str, optional
        Scale for y-axis, either 'linear' or 'log', by default 'linear'
    """
    
    # Get timesteps
    n_timesteps = agent_tensor[0].shape[0]
    timesteps = range(n_timesteps)

    # Add dashed grey vertical line at the start of each trial
    if trial_lines and num_trials is not None and num_trials > 1:
        add_trial_boundary_lines(ax, n_timesteps, num_trials)
    
    # Plot distance for each matrix
    for i, array_hist in enumerate(agent_tensor):
        # i loop over factors/modalities and array_hist is the history of the agent's parameters for that factor/modality
        # Calculate distances over time
        distances = [float(jnp.max(jnp.abs(array - env_tensor[i]))) for array in array_hist]

        # Plot with label from environment if available
        label = labels[i] if i < len(labels) else f"Factor/Modality {i}"
        ax.plot(timesteps, distances, label=label, linewidth=2)
    
    # Configure the plot
    ax.set_xlabel('Timestep')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.legend()

def print_initial_state(info, trial_idx=None):
    """Print the initial state of the environment."""
    #TODO: note this works only for batch_size==1
    #TODO: add printing initial state for several trials
    # Check if multi-trial
    is_multi, _ = is_multi_trial(info)
    if is_multi and trial_idx is None: 
        raise ValueError("trial_idx must be specified for printing initial state in a multi-trial rollout.")
    elif is_multi:
        print(f"Initial state at trial {trial_idx}: {initial_state(info, trial_idx)}")
        return

    print(f"Initial state: {initial_state(info, trial_idx)}")

def initial_state(info, trial_idx=None):
    """Get the initial state of the environment."""
    # Check if multi-trial
    is_multi, num_trials = is_multi_trial(info)
    if is_multi and trial_idx is None: 
        raise ValueError("trial_idx must be specified for getting initial state in a multi-trial rollout.")
    elif is_multi and trial_idx not in range(num_trials):
        raise ValueError(f"trial_idx must be in range(0, {num_trials-1}) for getting initial state in a multi-trial rollout.")
    elif is_multi:
        return [int(info['env'].state[f][trial_idx][0][0]) for f in range(len(info['env'].state))]
    elif not is_multi:
        return [int(info['env'].state[f][0][0]) for f in range(len(info['env'].state))]


def plot_rollout_preferences(prefs: Dict,
dict_key: str, # "cumulative_preferences" or "combined_preferences"
batch_idx: Optional[int] = 0,
title: Optional[str] = None, 
figsize: Tuple[int, int] = (10, 5), 
trial_lines: Optional[bool] = True,
zoom: Optional[bool] = True):
    """
    Plot preference data returned by compute_preferences.
    
    This function visualizes agent preferences data in two different modes:
    1. zoom=True (default): Shows preference values across all timesteps and trials,
       flattening multi-trial data into a single timeline with trial boundary markers.
    2. zoom=False: Shows only the last timestep preference value from each trial,
       useful for tracking how final preferences evolve over many trials.
    
    Parameters
    ----------
    prefs : Dict
        Dictionary returned by compute_preferences containing preference data
    dict_key : str
        Key specifying which preference data to plot:
        - "combined_preferences": Sum of preferences across modalities
        - "cumulative_preferences": Cumulative sum of combined preferences
    batch_idx : int, optional
        Batch index to plot, by default 0
    title : str, optional
        Plot title, by default None
    figsize : tuple(int, int), optional
        Figure size as (width, height), by default (10, 5)
    trial_lines : bool, optional
        Whether to show vertical lines at trial boundaries when zoom=True, by default True
    zoom : bool, optional
        Visualization mode - True to show all timesteps, False to show only final timestep 
        of each trial, by default True
        
    Note
    ----
    This function handles both single-trial and multi-trial data formats. For multi-trial
    data, the preference arrays will have a leading trial dimension added by lax.scan.
    """
    
    # get number of trials
    if prefs[dict_key].ndim == 2: 
        is_multi, num_trials = False, None
    elif prefs[dict_key].ndim == 3: 
        is_multi, num_trials = True, prefs[dict_key].shape[0]
    else: 
        raise ValueError("Unexpected shape for combined preferences.")

    plt.figure(figsize=figsize)

    if zoom:
        # Flatten preference data for plotting (time axis)
        prefs_to_plot = flatten_multi_trial_tensor(prefs[dict_key][..., batch_idx], is_multi)
        total_timesteps = prefs_to_plot.shape[0]

        # Add vertical lines at the beginning of each trial if there are trials
        if trial_lines and num_trials is not None and num_trials > 1:
            add_trial_boundary_lines(plt.gca(), total_timesteps, num_trials)

        # Plot preferences as a bar plot
        plt.bar(range(total_timesteps), prefs_to_plot)
        plt.xlabel('Timestep')

    else: # plot last timestep from each trial

        prefs_to_plot = prefs[dict_key][...,-1, batch_idx]
        
        if is_multi:
            plt.bar(range(num_trials), prefs_to_plot)
        else:
            plt.bar(range(1), prefs_to_plot)
        plt.xlabel('Trial')
    
    plt.ylim(bottom=prefs_to_plot.min(),top=prefs_to_plot.max())
        
    plt.ylabel('nats')

    if title is not None:
        plt.title(title)
