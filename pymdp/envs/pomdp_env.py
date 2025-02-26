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
                A_dependencies,
                B_dependencies,
                B_action_dependencies
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
