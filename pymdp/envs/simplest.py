import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pymdp.utils import fig2img
from equinox import field
from .env import Env
import matplotlib.pyplot as plt
from jax import nn


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

def plot_beliefs(info, agent=None, show=True):
    """Plot the agent's initial beliefs, final beliefs, and (if agent provided) preferences.
     
    Args:
        info: Rollout info dict with 'qs' for belief history
        agent: Agent instance (optional)
        show: Whether to call plt.show()
    """
    
    n_plots = 3 if agent is not None else 2
    plt.figure(figsize=(4 * n_plots, 4))

    # Plot initial beliefs as a bar plot
    plt.subplot(1, n_plots, 1)
    plt.bar([0, 1], info['qs'][0][0][0])
    plt.title('Initial Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot final beliefs as a bar plot
    plt.subplot(1, n_plots, 2)
    plt.bar([0, 1], info['qs'][0][-1][0])  # Using qs instead of belief_hist
    plt.title('Final Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot preferences as a bar plot
    if agent is not None:
        plt.subplot(1, n_plots, 3)
        plt.bar([0, 1], agent.C[0][0])
        plt.title('Preferences')
        plt.xticks([0, 1], ['Left', 'Right'])
        plt.ylim(0, 1)

    plt.tight_layout()
    if show:
        plt.show()
    
    return plt


def plot_A_learning(agent, info, env):
    """Plot the agent's learning progress for A matrix.
    
    Args:
        agent: Agent instance with parameter learning enabled
        info: Dict containing rollout info with parameter history
        env: Environment instance containing true parameters
    """
    
    plt.figure(figsize=(12, 5))
    plt.clf()  # Clear the current figure
    
    if agent.learn_A:
        # Create subplot for A matrix
        ax1 = plt.subplot(121)
        
        # Plot distance on left y-axis
        A_hist = info["agent"].A[0]
        timesteps = range(len(A_hist))
        distances = [jnp.linalg.norm(A - env.params["A"][0]) for A in A_hist]
        dist_line = ax1.plot(timesteps, distances, 'k--', label='Distance to true A', linewidth=2)[0]
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Distance to true parameters')
        ax1.set_ylim(bottom=0) # Set y-axis to include 0 
        
        # Create twin axis for probabilities
        ax2 = ax1.twinx()
        
        # Plot individual elements on right y-axis
        A_array = jnp.array(A_hist)
        lines = []
        lines.append(ax2.plot(timesteps, A_array[:, 0, 0], label='A[0,0]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 0, 1], label='A[0,1]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 1, 0], label='A[1,0]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 1, 1], label='A[1,1]', alpha=0.5)[0])
        ax2.set_ylabel('Belief')
        
        # Merge legends
        all_lines = [dist_line] + lines
        labs = [l.get_label() for l in all_lines]
        ax1.legend(all_lines, labs, loc='center left')
        
        plt.title('A Matrix Learning')
    
    plt.tight_layout()
    plt.show()
    
    return plt

def print_rollout(info, batch_idx=0):
    """
    Print a detailed summary of an agent's trajectory through the simplest environment.
    
    Args:
        info: Dictionary containing rollout information from the active inference loop
        batch_idx: Which batch to print information for (default=0)
    """
    location_observations = ['Left', 'Right']
    action_names = ['Left', 'Right']
    
    # Get relevant arrays for the specified batch
    observations = info["observation"][0]  # Shape: (T, batch_size, 1)
    actions = info["action"]               # Shape: (T, batch_size)
    beliefs = info["qs"][0]                # Shape: (T, batch_size, 2)
    policies = info["qpi"]                 # Shape: (T, batch_size, num_policies)
    
    # Print initial setup
    print("\n=== Starting Active Inference Experiment ===")
    print(f"Number of timesteps: {observations.shape[0]-1}")
    print(f"Batch size: {observations.shape[1]}")
    print(f"Number of policies: {policies.shape[-1]}")
    print("\n=== Initial Setup ===")
    print(f"Prior state beliefs (D): Left: {float(beliefs[0, batch_idx, 0]):.3f}, Right: {float(beliefs[0, batch_idx, 1]):.3f}")
    print(f"Initial observation: [{location_observations[int(observations[0, batch_idx, 0])]}]")

    # Print trajectory
    print("\n=== Trajectory ===")
    num_timesteps = observations.shape[0]
    for t in range(1, num_timesteps):
        print(f"\n[Timestep {t}]")
      
        # Print state beliefs after observing
        print(f"State beliefs: Left: {float(beliefs[t, batch_idx, 0]):.3f}, Right: {float(beliefs[t, batch_idx, 1]):.3f}")
        
        # Print policy distribution
        print(f"Policy distribution:")
        for p_idx, p_prob in enumerate(policies[t, batch_idx]):
            print(f"  Policy {p_idx}: {float(p_prob):.3f}")
        
        # Print action and next observation
        next_action = int(actions[t, batch_idx].item())  # Use .item() to get scalar value
        next_obs = int(observations[t, batch_idx, 0].item())  # Use .item() to get scalar value
        
        print(f"Action taken: [Move to {action_names[next_action]}]")
        print(f"Next observation: [{location_observations[next_obs]}]")
    
    print("\n=== End of Experiment ===")

def render_rollout(env, info, save_gif=False, filename=None):
    """Render a video of the agent's trajectory through the environment.
    
    Args:
        env: SimplestEnv instance
        info: Dict containing rollout info
        save_gif: Whether to save the animation as a gif
        filename: Path to save the gif (if save_gif is True)
    """
    import mediapy
    from PIL import Image
    import os
    import jax.numpy as jnp
    
    frames = []
    for t in range(info["observation"][0].shape[0]):  # iterate over timesteps
        # get observations for this timestep
        observations_t = [info["observation"][0][t, :, :]]  # Only one observation modality (location)

        frame = env.render(mode="rgb_array", observations=observations_t)
        frame = jnp.asarray(frame, dtype=jnp.uint8)
        plt.close()  # close the figure to prevent memory leak
        frames.append(frame)

    frames = jnp.array(frames, dtype=jnp.uint8)
    mediapy.show_video(frames, fps=1)

    if save_gif:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000,  # 1000ms per frame
            loop=0
        )