"""
Tools for analyzing and visualizing active inference agent behavior.

__author__: Lancelot Da Costa
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

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

def plot_model_comparison(pe_analysis1: Dict, pe_analysis2: Dict, labels: Tuple[str, str] = ('Model 1', 'Model 2'), 
                         figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create comparison plots between two models showing their prediction error metrics.

    Parameters
    ----------
    pe_analysis1 : Dict
        First model's prediction error analysis dictionary from compute_prediction_errors
    pe_analysis2 : Dict
        Second model's prediction error analysis dictionary from compute_prediction_errors
    labels : Tuple[str, str], optional
        Labels for the two models in the plots. Default is ('Model 1', 'Model 2')
    figsize : Tuple[int, int], optional
        Figure size as (width, height). Default is (15, 12)

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the subplots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Accumulated Prediction Error
    ax1.plot(pe_analysis1["pe_accumulated"], label=labels[0], alpha=0.7)
    ax1.plot(pe_analysis2["pe_accumulated"], label=labels[1], alpha=0.7)
    ax1.set_title('Accumulated Prediction Error')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Accumulated PE (nats)')
    ax1.legend()
    ax1.set_yscale('log')

    # Plot 2: Prediction Error
    ax2.plot(pe_analysis1["pred_error"], label=labels[0], alpha=0.7)
    ax2.plot(pe_analysis2["pred_error"], label=labels[1], alpha=0.7)
    ax2.set_title('Prediction Error')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('PE (nats)')
    ax2.legend()
    ax2.set_yscale('log')

    # Plot 3: Complexity
    ax3.plot(pe_analysis1["complexity"], label=labels[0], alpha=0.7)
    ax3.plot(pe_analysis2["complexity"], label=labels[1], alpha=0.7)
    ax3.set_title('Complexity')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Complexity (nats)')
    ax3.legend()
    ax3.set_yscale('log')

    # Plot 4: Negative Accuracy
    ax4.plot(pe_analysis1["neg_accuracy"], label=labels[0], alpha=0.7)
    ax4.plot(pe_analysis2["neg_accuracy"], label=labels[1], alpha=0.7)
    ax4.set_title('Negative Accuracy')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Negative Accuracy (nats)')
    ax4.legend()
    ax4.set_yscale('log')

    plt.tight_layout()
    #return fig
