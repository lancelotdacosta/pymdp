"""
Tools for analyzing and visualizing active inference agent behavior.

__author__: Lancelot Da Costa
"""

import jax.numpy as jnp
import jax.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union
import mediapy
from PIL import Image
import os
import numpy as np
import io

def plot_prediction_errors(pe_analysis: Dict, title: Optional[str] = None, figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
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

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the plot
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(pe_analysis["pred_error"], label='Prediction error', alpha=1.0)
    plt.plot(pe_analysis["complexity"], label='Complexity', alpha=0.7)
    plt.plot(pe_analysis["neg_accuracy"], label='Negative accuracy', alpha=0.7)
    plt.plot(pe_analysis["complexity_l2"], label='L2 norm Complexity', alpha=0.4)
    plt.plot(pe_analysis["pe_accumulated"], label='Accumulated prediction errors')
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel('nats')
    plt.yscale('log')
    plt.grid(True)
    
    if title is not None:
        plt.title(title)
    
    #return fig

def plot_model_comparison(pe_analyses, labels=None, figsize: Tuple[int, int] = (15, 12), alpha: float = 0.7, lw: float = 1.0) -> plt.Figure:
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
    alphas : Tuple[float, float, float, float], optional
        Alpha values for transparency in each plot. Default is (0.5, 0.5, 0.5, 0.5)

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the subplots
    """
    # Handle different input types for backward compatibility
    if isinstance(pe_analyses, dict):
        # Single PE analysis
        pe_analyses = [pe_analyses]

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

    # Plot 1: Accumulated Prediction Error
    for i, pe_analysis in enumerate(pe_analyses):
        ax1.plot(pe_analysis["pe_accumulated"], label=labels[i], alpha=alpha, lw=lw)
    ax1.set_title('Accumulated Prediction Error')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Accumulated PE (nats)')
    ax1.legend()
    ax1.set_yscale('log')

    # Plot 2: Prediction Error
    for i, pe_analysis in enumerate(pe_analyses):
        ax2.plot(pe_analysis["pred_error"], label=labels[i], alpha=alpha, lw=lw)
    ax2.set_title('Prediction Error')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('PE (nats)')
    ax2.legend()
    ax2.set_yscale('log')

    # Plot 3: Complexity
    for i, pe_analysis in enumerate(pe_analyses):
        ax3.plot(pe_analysis["complexity"], label=labels[i], alpha=alpha, lw=lw)
    ax3.set_title('Complexity')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Complexity (nats)')
    ax3.legend()
    ax3.set_yscale('log')

    # Plot 4: Negative Accuracy
    for i, pe_analysis in enumerate(pe_analyses):
        ax4.plot(pe_analysis["neg_accuracy"], label=labels[i], alpha=alpha, lw=lw)
    ax4.set_title('Negative Accuracy')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Negative Accuracy (nats)')
    ax4.legend()
    ax4.set_yscale('log')

    plt.tight_layout()
    #return fig

def print_rollout(info, env, batch_idx=0):
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
    env : Env
        Environment instance containing labels dictionary
    batch_idx : int, optional
        Batch index to print for, by default 0
    """
    # Get the environment labels
    labels = env.get_labels()
    
    # Extract variables from info dictionary
    observations = info["observation"] # List of modality arrays, shape: (T+1, batch_size, 1)
    beliefs = info["qs"] # List of factor arrays, shape: (T+1, batch_size, 1, num_states[f])
    policies = info["qpi"] # Shape: (T+1, batch_size, num_policies)
    actions = info["action"] # Shape: (T+1, batch_size, control_factors)
    empirical_priors = info["empirical_prior"] # List of prior belief arrays for each state factor
    
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
    print(f"Number of timesteps: {num_timesteps-1}")  # -1 because includes initial observation
    print(f"Batch size: {observations[0].shape[1]}")
    print(f"Number of policies: {policies.shape[-1]}")
    print(f"State factors: {state_factor_names}")
    print(f"Observation modalities: {observation_modality_names}")
    print(f"Control factors: {control_factor_names}")
    
    def format_state_dist(factor_idx, state_probs):
        """Helper to format state distribution nicely using labels"""
        factor_name = state_factor_names[factor_idx]
        state_labels = labels["state_factors"][factor_name]
        
        # Create formatted string of probabilities with labels
        probs_str = ", ".join([f"{state_labels[i]}: {float(prob):.3f}" 
                             for i, prob in enumerate(state_probs)])
        return f"[{probs_str}]"
    
    # Print initial timestep info
    print("\n=== Initial Timestep (t=0) ===")
    
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

    # Print trajectory
    for t in range(1, num_timesteps):
        print(f"\n=== Timestep {t} ===")
        
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
    
    # Get the number of timesteps in the rollout
    num_timesteps = info["observation"][0].shape[0]
    
    # Get the number of observation modalities
    num_modalities = len(info["observation"])
    
    frames = []
    for t in range(num_timesteps):  # iterate over timesteps
        # Prepare observations for current timestep
        observations_t = [info["observation"][mod_idx][t] for mod_idx in range(num_modalities)]
        
        # Call the environment's render method
        frame = env.render(mode="rgb_array", observations=observations_t)
        frame = jnp.asarray(frame, dtype=jnp.uint8)
        plt.close()  # close the figure to prevent memory leak
        frames.append(frame)
    
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

def plot_preferences(agent, env=None, figsize=None, show=True, batch_idx=0):
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
