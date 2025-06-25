#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base class for POMDP environments.
Provides structure extraction and common POMDP functionality.

__author__: Lancelot Da Costa
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import jax.numpy as jnp
import jax
from .env import Env
from ..distribution import Distribution
from ..models.pomdp import POMDPStructure
from ..priors import (
    default_A_dependencies,
    default_B_dependencies,
    default_B_action_dependencies
)


class POMDPEnv(Env):
    """Base class for POMDP environments.
    
    Provides common functionality for POMDP environments and methods to extract
    POMDP structure from environment parameters.
    """
    
    def get_tensors(self, copy=False):
        """Get the tensors and dependencies for the POMDP structure.
        
        Parameters
        ----------
        copy : bool, optional
            Whether to return copies of the tensors, by default False
            
        Returns
        -------
        Tuple
            Tuple containing:
            - A: List of observation likelihood matrices
            - B: List of transition likelihood matrices
            - D: List of initial state prior matrices
            - A_dependencies: List of dependencies between observation modalities and state factors
            - B_dependencies: List of dependencies between state factors
            - B_action_dependencies: List of dependencies between control factors and state factors
        """
        # Get dependencies or use defaults if not specified
        A_dependencies = self.dependencies.get("A", default_A_dependencies(len(self.params["A"]), len(self.params["B"])))
        B_dependencies = self.dependencies.get("B", default_B_dependencies(len(self.params["B"])))
        B_action_dependencies = self.dependencies.get("B_action", default_B_action_dependencies(len(self.params["B"])))
        
        if copy:
            return (
                [a.copy() for a in self.params["A"]],
                [b.copy() for b in self.params["B"]],
                [d.copy() for d in self.params["D"]],
                [deps.copy() for deps in A_dependencies],
                [deps.copy() for deps in B_dependencies],
                [deps.copy() for deps in B_action_dependencies]
            )
        return (
            self.params["A"],
            self.params["B"],
            self.params["D"],
            A_dependencies,
            B_dependencies,
            B_action_dependencies
        )

    def get_structure(self) -> POMDPStructure:
        """Extract POMDP structure from environment parameters.
        
        Returns
        -------
        POMDPStructure
            Structure specification containing dimensions and dependencies
        """
        return POMDPStructure.from_env(self)

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

    def get_default_model_params(self):
        """Get default model parameters for this environment.
        
        Returns
        -------
        dict
            Dictionary of default model parameters
        """
        return {
            "T": 10  # Default length of simulation
        }
        
    def get_default_C(self):
        """Get default preference matrices C for this environment.
        
        Returns
        -------
        list
            List of C matrices for each observation modality
        """
        # Basic implementation - all observations equally preferred
        # Each A matrix has shape (batch_size, n_observations, *state_dims)
        # We want zeros of shape (batch_size, n_observations)
        batch_size = self.params["A"][0].shape[0]
        return [jnp.zeros((batch_size, a.shape[1]), dtype=jnp.float32) for a in self.params["A"]]
        
    def get_default_agent_params(self):
        """Get default agent parameters for this environment.
        
        Returns
        -------
        dict
            Dictionary of default agent parameters
        """
        return {
            "policy_len": 1,
            "inference_algo": "fpi",
            "apply_batch": False,
            "action_selection": "stochastic"
        }
