#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for defining POMDP structure.

__author__: Lancelot Da Costa
"""

import equinox as eqx
from typing import Dict, List
import jax.numpy as jnp


class POMDPStructure(eqx.Module):
    """
    Structure specification for a Partially Observable Markov Decision Process (POMDP).
    
    Attributes
    ----------
    num_obs : List[int]
        Number of observations for each modality
    num_states : List[int]
        Number of states for each factor
    num_actions : List[int]
        Number of actions for each control factor
    num_modalities : int
        Number of observation modalities
    num_factors : int
        Number of state factors
    num_batches : int
        Number of batches for parallel processing (default=1)
    T : int
        Number of timesteps for rollouts
    """
    num_obs: List[int]
    num_states: List[int]
    num_actions: List[int]
    num_modalities: int
    num_factors: int
    num_batches: int
    T: int

    def __init__(
        self,
        num_obs: List[int] | int,
        num_states: List[int] | int,
        num_actions: List[int] | int,
        num_modalities: int = None,
        num_factors: int = None,
        num_batches: int = 1,
        T: int = 100,
    ):
        """Initialize POMDP structure.

        Parameters
        ----------
        num_obs : List[int] or int
            Number of observations for each modality. If int, creates single-modality list.
        num_states : List[int] or int
            Number of states for each factor. If int, creates single-factor list.
        num_actions : List[int] or int
            Number of actions for each control factor. If int, creates single-factor list.
        num_modalities : int, optional
            Number of observation modalities. If None, inferred from num_obs length.
        num_factors : int, optional
            Number of state factors. If None, inferred from num_states length.
        num_batches : int, default=1
            Number of parallel batches for processing. Must be positive.
        T : int, default=100
            Number of timesteps for rollouts.
        """
        # Convert single integers to lists if needed
        self.num_obs = [num_obs] if isinstance(num_obs, int) else num_obs
        self.num_states = [num_states] if isinstance(num_states, int) else num_states
        self.num_actions = [num_actions] if isinstance(num_actions, int) else num_actions
        
        # Infer num_modalities and num_factors if not provided
        self.num_modalities = len(self.num_obs) if num_modalities is None else num_modalities
        self.num_factors = len(self.num_states) if num_factors is None else num_factors
        
        # Batch size for parallel processing (default=1)
        self.num_batches = num_batches
        
        # Number of timesteps
        self.T = T
        
        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate the POMDP structure"""
        assert len(self.num_obs) == self.num_modalities, "Number of observation dimensions must match num_modalities"
        assert len(self.num_states) == self.num_factors, "Number of state dimensions must match num_factors"
        assert len(self.num_actions) == self.num_factors, "Number of action dimensions must match num_factors"
        assert all(n > 0 for n in self.num_obs), "Number of observations must be positive"
        assert all(n > 0 for n in self.num_states), "Number of states must be positive"
        assert all(n > 0 for n in self.num_actions), "Number of actions must be positive"
        assert self.T > 0, "Number of timesteps must be positive"
        assert self.num_batches > 0, "Number of batches must be positive"

    @classmethod
    def default(cls) -> "POMDPStructure":
        """Default structure for a simple POMDP"""
        return cls(
            num_obs=2,
            num_states=2,
            num_actions=2,
            T=100,
            num_batches=1,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "POMDPStructure":
        """Create structure from dictionary"""
        # Ensure num_batches has a default
        if "num_batches" not in config_dict:
            config_dict["num_batches"] = 1
            
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Convert structure to dictionary"""
        return {
            "num_obs": self.num_obs,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "num_modalities": self.num_modalities,
            "num_factors": self.num_factors,
            "num_batches": self.num_batches,
            "T": self.T,
        }

    def __repr__(self) -> str:
        """String representation showing POMDP structure"""
        structure = [
            f"obs: {self.num_obs}",
            f"states: {self.num_states}",
            f"actions: {self.num_actions}"
        ]
        structure.append(f"batches: {self.num_batches}")
        structure.append(f"T: {self.T}")
        
        return f"POMDPStructure({', '.join(structure)})"