"""
Tools for analyzing and visualizing active inference agent behavior.

__author__: Lancelot Da Costa
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union

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
    print(f"\nState factors: {state_factor_names}")
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
    
    print("-" * 50)

    # Print trajectory
    for t in range(1, num_timesteps):
        print(f"\n=== Timestep {t} ===")
        
        # Print policy distribution
        print("Policy selection:")
        for p_idx, p_prob in enumerate(policies[t, batch_idx]):
            prob_str = f"{float(p_prob):.3f}"
            print(f"  Policy {p_idx:<15} : {prob_str:>8}")
 
        # Print actions for each control factor and their consequences
        for c in range(num_control_factors):
            control_name = control_factor_names[c]
            action_idx = int(actions[t, batch_idx, c].item())
            action_label = labels["control_factors"][control_name][action_idx]
            print(f"Action ({control_name}): [{action_label}]")
        
        # Print predicted next state (empirical prior) for each factor
        for f in range(num_state_factors):
            print(f"Predicted next state ({state_factor_names[f]}): ", 
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
