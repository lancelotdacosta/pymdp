import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pymdp.utils import fig2img
from equinox import field
from .env import Env

class SimplestEnv(Env):
    """
    Implementation of the simplest environment in JAX.
    This environment has two states (locations) and serves as a minimal test case for pymdp.
    Each state represents a location (left=0, right=1).
    There are two possible actions (left=0, right=1) which deterministically lead to their respective states.
    This is a fully observed environment - the observation likelihood matrix A is the identity matrix,
    meaning that each state maps deterministically to its corresponding observation with probability 1.
    This makes the true state of the environment directly observable to the agent.
    """

    state: jnp.ndarray = field(static=False)

    def __init__(self, batch_size=1):
        """
        Initialize the simplest environment.
        
        Args:
            batch_size: Number of parallel environments
        """
        # Generate and broadcast observation likelihood(A), transition (B), and initial state (D) tensors
        A, A_dependencies = self.generate_A()
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B, B_dependencies = self.generate_B()
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = self.generate_D()
        D = [jnp.broadcast_to(d, (batch_size,) + d.shape) for d in D]

        params = {
            "A": A,
            "B": B,
            "D": D,
        }

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        super().__init__(params, dependencies)

    def generate_A(self):
        """
        Generate observation likelihood tensor.
        Maps true location to observed location with a simple identity mapping.
        """
        A = []
        # Simple identity mapping between states and observations
        A.append(jnp.eye(2))  # 2x2 identity matrix for 2 locations
        
        A_dependencies = [[0]]  # Only depends on location factor
        
        return A, A_dependencies

    def generate_B(self):
        """
        Generate transition tensor.
        Shape: (next_state, current_state, action)
        For each action (left=0, right=1), we deterministically transition to that state.
        
        B[0] has shape (2, 2, 2) where:
        - First dimension (2): next state (left=0, right=1)
        - Second dimension (2): current state (left=0, right=1)
        - Third dimension (2): action (left=0, right=1)
        
        For action=left (0):
            [[1, 1],  # Always go to left state (state 0)
             [0, 0]]
        For action=right (1):
            [[0, 0],  # Always go to right state (state 1)
             [1, 1]]
        """
        B = []
        
        # Initialize transition tensor
        B_locs = jnp.zeros((2, 2, 2))
        
        # For action 0 (left), always go to state 0 (left)
        B_locs = B_locs.at[0, :, 0].set(1.0)
        
        # For action 1 (right), always go to state 1 (right)
        B_locs = B_locs.at[1, :, 1].set(1.0)
        
        B.append(B_locs)
        
        B_dependencies = [[0]]  # Only depends on location factor
        
        return B, B_dependencies

    def generate_D(self):
        """
        Generate initial state distribution.
        Always starts at location 0 (left).
        """
        D = []
        initial_location = jnp.array([1.0, 0.0])  # Start at location 0 (left)
        D.append(initial_location)
        
        return D

    def render(self, mode="human", observations=None):
        """
        Render the environment.
        
        Args:
            mode: 'human' displays plot, 'rgb_array' returns image array
            observations: List of observation arrays from environment
        
        Returns:
            Image array if mode=='rgb_array', else None
        """
        # Get batch size from observations or params
        if observations is not None:
            current_obs = observations
            batch_size = observations[0].shape[0]
        else:
            current_obs = self.current_obs
            batch_size = self.params["A"][0].shape[0]

        # Create figure with subplots (one per batch)
        n = int(jnp.ceil(jnp.sqrt(batch_size)))  # Square grid
        fig, axes = plt.subplots(n, n, figsize=(6, 6))

        # For each environment in the batch
        for i in range(batch_size):
            row = i // n
            col = i % n
            if batch_size == 1:
                ax = axes
            else:
                ax = axes[row, col]
            
            # Set up the plot
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Left', 'Right'])
            ax.set_yticks([])
            ax.set_aspect('equal')
            
            # Draw the two positions
            left_pos = patches.Circle((0, 0), 0.2, facecolor='lightgray', edgecolor='black')
            right_pos = patches.Circle((1, 0), 0.2, facecolor='lightgray', edgecolor='black')
            ax.add_patch(left_pos)
            ax.add_patch(right_pos)
            
            # Show agent position
            loc = current_obs[0][i, 0]
            agent_pos = patches.Circle((loc, 0), 0.1, facecolor='red')
            ax.add_patch(agent_pos)
            
            # Add arrow to show agent direction (this is just for aesthetics)
            ax.arrow(loc, 0, 0.15 if loc == 0 else -0.15, 0,
                    head_width=0.1, head_length=0.1, fc='red', ec='red')

        # Hide any extra subplots if batch_size isn't a perfect square
        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig)
            return img