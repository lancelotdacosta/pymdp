#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for defining POMDP structure and configuration.

__author__: Lancelot Da Costa
"""

import equinox as eqx
from typing import Dict, List
import jax.numpy as jnp
import jax
from ..learning import LearningConfig
from ..priors import dirichlet_prior


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
    A_dependencies : List[List[int]]
        For each observation modality, list of state factor indices that it depends on
    B_dependencies : List[List[int]]
        For each state factor, list of state factor indices that its transitions depend on
    """
    num_obs: List[int]
    num_states: List[int]
    num_actions: List[int]
    num_modalities: int
    num_factors: int
    num_batches: int
    T: int
    A_dependencies: List[List[int]]
    B_dependencies: List[List[int]]

    def __init__(
        self,
        num_obs: List[int] | int,
        num_states: List[int] | int,
        num_actions: List[int] | int,
        num_modalities: int = None,
        num_factors: int = None,
        num_batches: int = 1,
        T: int = 100,
        A_dependencies: List[List[int]] = None,
        B_dependencies: List[List[int]] = None,
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
        A_dependencies : List[List[int]], optional
            For each observation modality, list of state factor indices that it depends on.
            If None, assumes each modality depends only on the corresponding factor.
        B_dependencies : List[List[int]], optional
            For each state factor, list of state factor indices that its transitions depend on.
            If None, assumes each factor depends only on itself.
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
        
        # Set default dependencies if not provided
        if A_dependencies is None:
            # By default, each modality depends on the corresponding factor if possible
            assert self.num_factors >= self.num_modalities, "Number of factors must be at least number of modalities for default initialisation of A_dependencies"
            self.A_dependencies = [[i] for i in range(self.num_modalities)]
        else:
            self.A_dependencies = A_dependencies
            
        if B_dependencies is None:
            # By default, each factor's transitions depend only on itself
            self.B_dependencies = [[i] for i in range(self.num_factors)]
        else:
            self.B_dependencies = B_dependencies
        
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
        
        # Validate dependencies
        assert len(self.A_dependencies) == self.num_modalities, "A_dependencies length must match num_modalities"
        assert len(self.B_dependencies) == self.num_factors, "B_dependencies length must match num_factors"
        
        for deps in self.A_dependencies:
            assert all(0 <= i < self.num_factors for i in deps), "A_dependencies indices must be valid state factor indices"
            
        for deps in self.B_dependencies:
            assert all(0 <= i < self.num_factors for i in deps), "B_dependencies indices must be valid state factor indices"

    @classmethod
    def default(cls) -> "POMDPStructure":
        """Default structure for a simple POMDP"""
        return cls(
            num_obs=2,
            num_states=2,
            num_actions=2,
            T=100,
            num_batches=1,
            # Default dependencies (each modality/factor depends only on itself)
            A_dependencies=[[0]],
            B_dependencies=[[0]],
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
            "A_dependencies": self.A_dependencies,
            "B_dependencies": self.B_dependencies,
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
        structure.append(f"A_deps: {self.A_dependencies}")
        structure.append(f"B_deps: {self.B_dependencies}")
        
        return f"POMDPStructure({', '.join(structure)})"


class POMDPConfig(eqx.Module):
    """
    Complete configuration for a POMDP, including both structure and learning configuration.
    
    Attributes
    ----------
    structure : POMDPStructure
        Structure specification containing dimensions and dependencies
    learning : LearningConfig
        Configuration for parameter learning
    """
    structure: POMDPStructure
    learning: LearningConfig

    def __init__(
        self,
        structure: POMDPStructure,
        learning: LearningConfig = None,
    ):
        """Initialize POMDP configuration.

        Parameters
        ----------
        structure : POMDPStructure
            Structure specification containing dimensions and dependencies
        learning : LearningConfig, optional
            Configuration for parameter learning. If None, uses no_learning() configuration.
        """
        self.structure = structure
        self.learning = learning if learning is not None else LearningConfig.no_learning()

    def update_learning(self, **kwargs) -> "POMDPConfig":
        """Update learning configuration with new parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Learning parameters to update (learn_A, learn_B, learn_D, lr_pA, lr_pB, lr_pD)
        
        Returns
        -------
        POMDPConfig
            New config with updated learning parameters
        """
        # Create new learning config with updated parameters
        learning_dict = self.learning.to_dict()
        learning_dict.update(kwargs)
        learning = LearningConfig.from_dict(learning_dict)
        
        return POMDPConfig(structure=self.structure, learning=learning)

    @classmethod
    def from_env(cls, env, learning: LearningConfig = None) -> "POMDPConfig":
        """Create configuration from a POMDP environment.
        
        Parameters
        ----------
        env : POMDPEnv
            Environment to extract structure from
        learning : LearningConfig, optional
            Configuration for parameter learning. If None, uses no_learning() configuration.
        
        Returns
        -------
        POMDPConfig
            Complete POMDP configuration
        """
        structure = env.get_structure()
        return cls(structure=structure, learning=learning)

    @classmethod
    def default(cls) -> "POMDPConfig":
        """Default configuration for a simple POMDP"""
        return cls(
            structure=POMDPStructure.default(),
            learning=LearningConfig.default(),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "POMDPConfig":
        """Create configuration from dictionary"""
        # Handle nested configs
        structure_dict = config_dict.pop("structure", None)
        learning_dict = config_dict.pop("learning", None)
        
        if structure_dict is not None:
            structure = POMDPStructure.from_dict(structure_dict)
        else:
            # If no nested structure, assume all structure params are at top level
            structure = POMDPStructure.from_dict(config_dict)
            
        if learning_dict is not None:
            learning = LearningConfig.from_dict(learning_dict)
        else:
            learning = None
            
        return cls(structure=structure, learning=learning)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            "structure": self.structure.to_dict(),
            "learning": self.learning.to_dict(),
        }

    def __repr__(self) -> str:
        """String representation showing complete POMDP configuration"""
        return f"POMDPConfig(structure={self.structure}, learning={self.learning})"


class POMDPModel(eqx.Module):
    """
    Model parameters and priors for a POMDP, initialized according to a configuration.
    
    This class handles the initialization and storage of the generative model parameters
    (A, B, D) and their priors (pA, pB, pD) based on a POMDPConfig.
    
    Attributes
    ----------
    config : POMDPConfig
        Configuration specifying the POMDP structure and learning settings
    A : List[jnp.ndarray]
        Observation model parameters - P(o|s)
    B : List[jnp.ndarray]
        Transition model parameters - P(s'|s,a)
    D : List[jnp.ndarray]
        Initial state distribution parameters - P(s_0)
    pA : List[jnp.ndarray]
        Prior parameters for observation model
    pB : List[jnp.ndarray]
        Prior parameters for transition model
    pD : List[jnp.ndarray]
        Prior parameters for initial state distribution
    """
    config: POMDPConfig
    A: List[jnp.ndarray]
    B: List[jnp.ndarray]
    D: List[jnp.ndarray]
    pA: List[jnp.ndarray]
    pB: List[jnp.ndarray]
    pD: List[jnp.ndarray]

    def __init__(
        self,
        config: POMDPConfig,
        key,
        init: str = "random",
        scale: float = 1.0,
    ):
        """Initialize POMDP model parameters and priors.
        
        Parameters
        ----------
        config : POMDPConfig
            Configuration specifying the POMDP structure and learning settings
        key : jax.random.PRNGKey
            Random key for initialization
        init : str, optional
            Initialization method for priors, by default "random"
        scale : float, optional
            Scale parameter for prior initialization, by default 1.0
        """
        self.config = config
        structure = config.structure

        # Create uniform base tensors for each parameter
        A_base = []
        for i in range(structure.num_modalities):
            # Get shape based on dependencies
            shape = [structure.num_batches, structure.num_obs[i]]
            for state_idx in structure.A_dependencies[i]:
                shape.append(structure.num_states[state_idx])
            A_base.append(
                jnp.ones(shape, dtype=jnp.float32) / structure.num_obs[i]
            )
        
        B_base = []
        for i in range(structure.num_factors):
            # Get shape based on dependencies
            shape = [structure.num_batches, structure.num_states[i]]
            for state_idx in structure.B_dependencies[i]:
                shape.append(structure.num_states[state_idx])
            shape.append(structure.num_actions[i])
            B_base.append(
                jnp.ones(shape, dtype=jnp.float32) / structure.num_states[i]
            )
        
        D_base = [
            jnp.ones(
                (structure.num_batches, structure.num_states[i]), 
                dtype=jnp.float32
            ) / structure.num_states[i] for i in range(structure.num_factors)
        ]

        # Split random key for each parameter
        key, key_A = jax.random.split(key)
        key, key_B = jax.random.split(key)
        key, key_D = jax.random.split(key)

        # Initialize parameters and priors using dirichlet_prior
        self.pA, self.A = dirichlet_prior(
            A_base, init=init, scale=scale, 
            learning_enabled=config.learning.learn_A, key=key_A
        )
        self.pB, self.B = dirichlet_prior(
            B_base, init=init, scale=scale,
            learning_enabled=config.learning.learn_B, key=key_B
        )
        self.pD, self.D = dirichlet_prior(
            D_base, init=init, scale=scale,
            learning_enabled=config.learning.learn_D, key=key_D
        )

    def __repr__(self) -> str:
        """String representation showing model parameters"""
        return f"POMDPModel(config={self.config})"