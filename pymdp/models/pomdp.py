#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for defining POMDP structure and configuration.

__author__: Lancelot Da Costa
"""

import equinox as eqx
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
import jax
from ..learning import LearningConfig
from ..priors import dirichlet_prior, check_consistency, create_uniform_A, create_uniform_B, create_uniform_D, default_A_dependencies, default_B_dependencies, default_B_action_dependencies



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
    B_action_dependencies : List[List[int]]
        For each state factor, list of control factor indices that influence its transitions
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
    B_action_dependencies: List[List[int]]

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
        B_action_dependencies: List[List[int]] = None,
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
            If None, assumes each modality depends on all factors (matching Agent class default).
        B_dependencies : List[List[int]], optional
            For each state factor, list of state factor indices that its transitions depend on.
            If None, assumes each factor depends only on itself.
        B_action_dependencies : List[List[int]], optional
            For each state factor, list of control factor indices that influence its transitions.
            If None, assumes each factor's transitions depend onlyon the corresponding control factor.
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
            # By default, each modality depends on all factors (matching Agent class default)
            A_dependencies = default_A_dependencies(self.num_modalities, self.num_factors)
            
        if B_dependencies is None:
            # By default, each factor's transitions depend only on itself
            B_dependencies = default_B_dependencies(self.num_factors)
        
        if B_action_dependencies is None:
            # By default, each factor's transitions depend on the corresponding control factor
            B_action_dependencies = default_B_action_dependencies(self.num_factors)
        
        self.A_dependencies = A_dependencies
        self.B_dependencies = B_dependencies
        self.B_action_dependencies = B_action_dependencies

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
        assert len(self.B_action_dependencies) == self.num_factors, "B_action_dependencies length must match num_factors"
        
        for deps in self.A_dependencies:
            assert all(0 <= i < self.num_factors for i in deps), "A_dependencies indices must be valid state factor indices"
            
        for deps in self.B_dependencies:
            assert all(0 <= i < self.num_factors for i in deps), "B_dependencies indices must be valid state factor indices"
            
        for deps in self.B_action_dependencies:
            assert all(0 <= i < self.num_factors for i in deps), "B_action_dependencies indices must be valid control factor indices"

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

    @classmethod
    def from_parameters(cls, A, B, A_dependencies=None, B_dependencies=None, B_action_dependencies=None, T=100):
        """Create POMDPStructure from model parameters.
        
        Parameters
        ----------
        A : List[jnp.ndarray]
            List of observation matrices for each modality
        B : List[jnp.ndarray]
            List of transition matrices for each factor
        A_dependencies : List[List[int]]
            List of state factor dependencies for each observation modality
        B_dependencies : List[List[int]]
            List of state factor dependencies for each state factor
        B_action_dependencies : List[List[int]]
            List of control factor dependencies for each state factor
        T : int, optional
            Number of timesteps for rollouts, by default 100
        
        Returns
        -------
        POMDPStructure
            Structure inferred from parameters
        """
        # Get dimensions from A matrix shapes
        # A[m] shape: (batch, obs_m, state1, state2, ...)
        num_obs = [a.shape[1] for a in A]
        num_modalities = len(num_obs)
        
        # Get dimensions from B matrix shapes
        # B[f] shape: (batch, next_state_f, curr_state_f, action)
        num_states = [b.shape[1] for b in B]  # second dim is current state
        num_factors = len(num_states)
        
        # Get number of actions for each factor from B matrix shapes
        num_actions = [b.shape[-1] for b in B]  # last dim is actions
        
        # Get batch size from A matrix
        num_batches = A[0].shape[0]
        
        return cls(
            num_obs=num_obs,
            num_states=num_states,
            num_actions=num_actions,
            num_modalities=num_modalities,
            num_factors=num_factors,
            num_batches=num_batches,
            T=T,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            B_action_dependencies=B_action_dependencies
        )

    @classmethod
    def from_env(cls, env, T=100):
        """Create structure from a POMDP environment.
        
        Parameters
        ----------
        env : POMDPEnv
            Environment to extract structure from
        T : int, optional
            Number of timesteps for rollouts, by default 100
            
        Returns
        -------
        POMDPStructure
            Structure specification containing dimensions and dependencies
        """
        A, B, _, A_dependencies, B_dependencies, B_action_dependencies = env.get_tensors()
        return cls.from_parameters(A, B, A_dependencies, B_dependencies, B_action_dependencies, T=T)

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
            "B_action_dependencies": self.B_action_dependencies,
        }

    def modify(self, **kwargs) -> "POMDPStructure":
        """Create a new POMDPStructure with modified parameters.
        
        This method allows you to create a new structure with specific parameters
        modified while keeping all other parameters the same.
        
        Parameters
        ----------
        **kwargs
            Parameters to modify. Valid parameters are:
            - num_obs: List[int]
            - num_states: List[int]
            - num_actions: List[int]
            - num_modalities: int
            - num_factors: int
            - num_batches: int
            - T: int
            - A_dependencies: List[List[int]]
            - B_dependencies: List[List[int]]
            - B_action_dependencies: List[List[int]]
            
        Returns
        -------
        POMDPStructure
            A new structure with the specified parameters modified
            
        Example
        -------
        >>> new_structure = structure.modify(num_states=[3], T=100)
        """
        # Get current config as dictionary
        config = self.to_dict()
        
        # Update with new parameters
        config.update(kwargs)
        
        # Create new structure
        return self.from_dict(config)

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
        structure.append(f"B_action_deps: {self.B_action_dependencies}")
        
        return f"POMDPStructure({', '.join(structure)})"


class POMDPModel(eqx.Module):
    """
    Model parameters and priors for a POMDP, initialized according to a structure and learning configuration.
    
    This class handles the initialization and storage of the generative model parameters
    (A, B, D) and their priors (pA, pB, pD) based on a POMDPStructure.
    
    Attributes
    ----------
    structure : POMDPStructure
        Structure specification containing dimensions and dependencies
    learning : LearningConfig
        Configuration for parameter learning
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
    structure: POMDPStructure
    learning: LearningConfig
    A: List[jnp.ndarray]
    B: List[jnp.ndarray]
    D: List[jnp.ndarray]
    pA: List[jnp.ndarray]
    pB: List[jnp.ndarray]
    pD: List[jnp.ndarray]

    def __init__(
        self,
        A: List[jnp.ndarray],
        B: List[jnp.ndarray],
        D: List[jnp.ndarray],
        A_dependencies,
        B_dependencies,
        B_action_dependencies,
        pA: List[jnp.ndarray] = None,
        pB: List[jnp.ndarray] = None,
        pD: List[jnp.ndarray] = None,
        T = 100
    ):
        # check consistency between A, B, D, pA, pB, pD
        A = check_consistency(A, pA, "A")
        B = check_consistency(B, pB, "B")
        D = check_consistency(D, pD, "D")
        
        self.A = A
        self.B = B
        self.D = D
        self.pA = pA
        self.pB = pB
        self.pD = pD

        # infer structure from parameters
        self.structure = POMDPStructure.from_parameters(self.A, self.B, A_dependencies, B_dependencies, B_action_dependencies, T=T)

        # infer learning config from parameters
        self.learning = LearningConfig.from_parameters(self.pA, self.pB, self.pD)

    @classmethod
    def from_structure(
        cls,
        structure: POMDPStructure,
        learning: LearningConfig = None,
        init: str = "random",
        scale: float = 1.0,
        key: jax.random.PRNGKey = None,
    ):
        """Create a POMDP model from a POMDPStructure and learning config.

        Parameters
        ----------
        structure : POMDPStructure
            Structure of the POMDP
        learning : LearningConfig, optional
            Configuration for parameter learning, by default None
        init : str, optional
            Initialization method for priors, by default "random"
        scale : float, optional
            Scale for prior initialization, by default 1.0
        key : jax.random.PRNGKey, optional
            Random key for initialization, by default None
            
        Returns
        -------
        POMDPModel
            Initialized model with the given structure
        key : jax.random.PRNGKey
            Random key
        """
        learning = learning if learning is not None else LearningConfig.default()

        # Create default parameters
        A_base, B_base, D_base = cls._create_default_parameters(structure)

        # Initialize parameters with priors
        A, pA, B, pB, D, pD, key = cls._initialize_parameters(A_base, B_base, D_base, learning, init, scale, key)
        
        return cls(
            A=A,
            B=B,
            D=D,
            A_dependencies=structure.A_dependencies,
            B_dependencies=structure.B_dependencies,
            B_action_dependencies=structure.B_action_dependencies,
            pA=pA,
            pB=pB,
            pD=pD,
            T=structure.T
        ), key

    @classmethod
    def from_env(
        cls,
        env,
        learning: LearningConfig = None,
        init: str = "random",
        scale: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None,
        T: int = 100
    ) -> "POMDPModel":
        """Create a POMDP model from an environment.

        Parameters
        ----------
        env : POMDPEnv
            Environment to create model from
        learning : LearningConfig, optional
            Configuration for parameter learning, by default None (uses default() configuration)
        init : str, optional
            Initialization method for priors when learning is enabled, by default "random"
        scale : float, optional
            Scale for prior initialization when learning is enabled, by default 1.0
        key : Optional[jax.random.PRNGKey], optional
            Random key for initialization, by default None
        T : int, optional
            Time horizon, by default 100

        Returns
        -------
        POMDPModel
            Model initialized from environment
        key : jax.random.PRNGKey
            Random key
        """
        learning = learning if learning is not None else LearningConfig.default()
        
        # Get tensors and dependencies from environment
        A_base, B_base, D_base, A_dependencies, B_dependencies, B_action_dependencies = env.get_tensors(copy=True)
        
        # Initialize parameters with priors based on learning configuration
        A, pA, B, pB, D, pD, key = cls._initialize_parameters(
            A_base, B_base, D_base,
            learning=learning,
            init=init,
            scale=scale,
            key=key
        )
            
        return cls(
            A=A,
            B=B,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            B_action_dependencies=B_action_dependencies,
            pA=pA,
            pB=pB,
            pD=pD,
            T=T
        ), key

    @classmethod
    def _create_default_parameters(cls, structure: POMDPStructure):
        """Create default (uniform)parameters for the POMDP model.
        
        Returns
        -------
        Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]
            A, pA, B, pB, D, pD arrays
        """
        
        # Create uniform base tensors
        A_base = create_uniform_A(
            num_batches=structure.num_batches,
            num_obs=structure.num_obs,
            num_states=structure.num_states,
            A_dependencies=structure.A_dependencies
        )
        
        B_base = create_uniform_B(
            num_batches=structure.num_batches,
            num_states=structure.num_states,
            num_actions=structure.num_actions,
            B_dependencies=structure.B_dependencies
        )
        
        D_base = create_uniform_D(
            num_batches=structure.num_batches,
            num_states=structure.num_states
        )
        
        return A_base, B_base, D_base

    @classmethod
    def _initialize_parameters(
        cls,
        A_base: List[jnp.ndarray],
        B_base: List[jnp.ndarray],
        D_base: List[jnp.ndarray],
        learning: LearningConfig,
        init: str = "random",
        scale: float = 1.0,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """Initialize parameters and priors based on learning configuration.
        
        Parameters
        ----------
        A_base : List[jnp.ndarray]
            Base observation matrices
        B_base : List[jnp.ndarray]
            Base transition matrices
        D_base : List[jnp.ndarray]
            Base initial state distributions
        learning : LearningConfig
            Learning configuration specifying which parameters to learn
        init : str, optional
            Initialization method for priors when learning is enabled, by default "random"
        scale : float, optional
            Scale for prior initialization when learning is enabled, by default 1.0
        key : Optional[jax.random.PRNGKey], optional
            Random key for initialization, by default None
            
        Returns
        -------
        Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]
            A, pA, B, pB, D, pD arrays
        key : jax.random.PRNGKey
            Random key
        """

        # Initialize parameters and priors using dirichlet_prior
        # When learning is disabled, pX will be None and X will be the base template
        pA, A, key = dirichlet_prior(A_base, init=init, scale=scale, learning_enabled=learning.learn_A, key=key)
        pB, B, key = dirichlet_prior(B_base, init=init, scale=scale, learning_enabled=learning.learn_B, key=key)
        pD, D, key = dirichlet_prior(D_base, init=init, scale=scale, learning_enabled=learning.learn_D, key=key)
        
        return A, pA, B, pB, D, pD, key

    def set_uniform_D(self) -> "POMDPModel":
        """Set initial state distribution D to uniform.
        
        Returns
        -------
        POMDPModel
            New model with uniform D
            
        Raises
        ------
        ValueError
            If learn_D is True (D should be learned not manually set)
        """
        # Check if D is being learned
        if self.learning.learn_D:
            raise ValueError(
                "Cannot update D when learn_D is True. "
                "Either set learn_D to False or let D be learned from data."
            )
        
        # Create uniform D
        new_D = create_uniform_D(
            self.structure.num_batches,
            self.structure.num_states
        )
        
        # Create new model with updated D
        model_dict = self.to_dict()
        model_dict["D"] = new_D
        
        return self.from_dict(model_dict)

    def to_dict(self) -> dict:
        """Convert model to dictionary of attributes."""
        return {
            "A": self.A,
            "B": self.B,
            "D": self.D,
            "pA": self.pA,
            "pB": self.pB,
            "pD": self.pD,
            "structure": self.structure,
            "learning": self.learning
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "POMDPModel":
        """Create model from dictionary of attributes."""
        return cls(
            A=config_dict["A"],
            B=config_dict["B"],
            D=config_dict["D"],
            pA=config_dict["pA"],
            pB=config_dict["pB"],
            pD=config_dict["pD"],
            A_dependencies=config_dict["structure"].A_dependencies,
            B_dependencies=config_dict["structure"].B_dependencies,
            B_action_dependencies=config_dict["structure"].B_action_dependencies,
            T=config_dict["structure"].T
        )

    def __repr__(self) -> str:
        """String representation showing model parameters"""
        return f"POMDPModel(structure={self.structure}, learning={self.learning})"