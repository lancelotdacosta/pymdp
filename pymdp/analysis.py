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
    
    return fig
