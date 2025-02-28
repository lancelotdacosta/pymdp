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
