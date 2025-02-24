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
    
    def get_tensors(self, copy: bool = True):
        """Get the tensors and dependencies needed to define the POMDP structure.
        
        Parameters
        ----------
        copy : bool, optional
            Whether to return copies of the tensors, by default True, returns originals if False
            
        Returns
        -------
        tuple
            (A, B, D, A_dependencies, B_dependencies) tuple containing:
            - A: List of observation matrices
            - B: List of transition matrices
            - D: List of initial state distributions
            - A_dependencies: List of dependencies between observation modalities and state factors
            - B_dependencies: List of dependencies between state factors
        """
        if copy:
            return (
                [a.copy() for a in self.params["A"]],
                [b.copy() for b in self.params["B"]],
                [d.copy() for d in self.params["D"]],
                self.dependencies["A"],
                self.dependencies["B"]
            )
        return (
            self.params["A"],
            self.params["B"],
            self.params["D"],
            self.dependencies["A"],
            self.dependencies["B"]
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
