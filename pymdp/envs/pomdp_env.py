#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for POMDP environments.
Provides structure extraction and common POMDP functionality.

__author__: Lancelot Da Costa
"""

from typing import Dict, List, Tuple
import jax.numpy as jnp
from .env import Env
from ..models.pomdp import POMDPStructure


class POMDPEnv(Env):
    """Base class for POMDP environments.
    
    Provides common functionality for POMDP environments and methods to extract
    POMDP structure from environment parameters.
    """
    
    def get_structure(self) -> POMDPStructure:
        """Extract POMDP structure from environment parameters.
        
        Returns
        -------
        POMDPStructure
            Structure specification containing dimensions and dependencies
        """
        # Get dimensions from A matrix shapes
        # A[m] shape: (batch, obs_m, state1, state2, ...)
        num_obs = [a.shape[1] for a in self.params["A"]]
        num_modalities = len(num_obs)
        
        # Get dimensions from B matrix shapes
        # B[f] shape: (batch, next_state_f, curr_state_f, action)
        # For each factor f, num_states[f] is the number of states for that factor
        num_states = [b.shape[1] for b in self.params["B"]]  # second dim is current state
        num_factors = len(num_states)
        
        # Get number of actions for each factor from B matrix shapes
        num_actions = [b.shape[-1] for b in self.params["B"]]  # last dim is actions
        
        # Get batch size from A matrix
        num_batches = self.params["A"][0].shape[0]
        
        # Dependencies are already stored in self.dependencies
        A_dependencies = self.dependencies["A"]
        B_dependencies = self.dependencies["B"]
        
        return POMDPStructure(
            num_obs=num_obs,
            num_states=num_states,
            num_actions=num_actions,
            num_modalities=num_modalities,
            num_factors=num_factors,
            num_batches=num_batches,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
        )

    def generate_A(self) -> Tuple[List[jnp.ndarray], List[List[int]]]:
        """Generate observation likelihood tensor and dependencies.
        
        Must be implemented by subclasses.
        
        Returns
        -------
        Tuple[List[jnp.ndarray], List[List[int]]]
            Tuple containing:
            - List of observation likelihood tensors for each modality
            - List of state factor dependencies for each modality
        """
        raise NotImplementedError("Subclasses must implement generate_A")
    
    def generate_B(self) -> Tuple[List[jnp.ndarray], List[List[int]]]:
        """Generate transition tensor and dependencies.
        
        Must be implemented by subclasses.
        
        Returns
        -------
        Tuple[List[jnp.ndarray], List[List[int]]]
            Tuple containing:
            - List of transition tensors for each factor
            - List of state factor dependencies for each factor
        """
        raise NotImplementedError("Subclasses must implement generate_B")
    
    def generate_D(self) -> List[jnp.ndarray]:
        """Generate initial state distribution.
        
        Must be implemented by subclasses.
        
        Returns
        -------
        List[jnp.ndarray]
            List of initial state distributions for each factor
        """
        raise NotImplementedError("Subclasses must implement generate_D")
