import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pymdp.utils import fig2img
from equinox import field
from typing import Dict, List, Tuple

from .pomdp_env import POMDPEnv


class GridWorldEnv(POMDPEnv):
    """Simple fully-observed grid-world environment.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    batch_size : int, optional
        Number of parallel environments (default = 1).

    The environment has one observation modality and one hidden state factor
    corresponding to the agent's location.  The agent can execute **five**
    deterministic actions:
    0. up
    1. left
    2. right
    3. down
    4. stay

    The hidden‐state space has ``rows × cols`` states.  States are indexed in
    row-major (matrix) order:
    ``state_id = row * cols + col`` with both indices starting at 0.
    Row 0 / Col 0 (top-left corner) is therefore state 0, which is also the
    unique start state.  The observation modality is fully observed, i.e. the
    A matrix is an identity mapping from states to observations.
    """

    # Equinox module fields
    rows: int = field(static=True)
    cols: int = field(static=True) 
    n_states: int = field(static=True)
    n_actions: int = field(static=True)
    # expose state for convenience / rendering
    state: jnp.ndarray = field(static=False)

    def __init__(self, rows: int = 2, cols: int = 2, batch_size: int = 1):
        # Store dimensions for use in initialization
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 5  # up, left, right, down, stay

        # --- build generative process tensors ---
        A, A_deps = self.generate_A(self.n_states)
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]

        B, B_deps = self.generate_B(self.rows, self.cols, self.n_states, self.n_actions)
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]

        D = self.generate_D(self.n_states)
        D = [jnp.broadcast_to(d, (batch_size,) + d.shape) for d in D]

        params = {"A": A, "B": B, "D": D}
        deps = {"A": A_deps, "B": B_deps}

        # Create labels manually before calling super().__init__
        labels = self._create_labels(rows, cols)

        super().__init__(params=params, dependencies=deps, labels=labels)

    # ---------------------------------------------------------------------
    # tensor construction helpers
    # ---------------------------------------------------------------------
    def generate_A(self, n_states):
        """Identity observation model (fully observed)."""
        A = [jnp.eye(n_states)]  # shape (n_obs, n_states)
        return A, [[0]]  # A depends only on location factor

    def generate_B(self, rows, cols, n_states, n_actions):
        """Deterministic transition dynamics for the five actions."""
        B_locs = jnp.zeros((n_states, n_states, n_actions))

        for s in range(n_states):
            r, c = divmod(s, cols)

            up = s - cols if r > 0 else s
            left = s - 1 if c > 0 else s
            right = s + 1 if c < cols - 1 else s
            down = s + cols if r < rows - 1 else s
            stay = s

            B_locs = B_locs.at[up, s, 0].set(1.0)
            B_locs = B_locs.at[left, s, 1].set(1.0)
            B_locs = B_locs.at[right, s, 2].set(1.0)
            B_locs = B_locs.at[down, s, 3].set(1.0)
            B_locs = B_locs.at[stay, s, 4].set(1.0)

        return [B_locs], [[0]]  # B depends only on location factor

    def generate_D(self, n_states):
        """Start distribution: always begin at top-left corner (state 0)."""
        D_loc = jnp.zeros((n_states,))
        D_loc = D_loc.at[0].set(1.0)
        return [D_loc]

    # ------------------------------------------------------------------
    # human-readable labels (optional but nice for debugging)
    # ------------------------------------------------------------------
    def _create_labels(self, rows, cols) -> Dict[str, Dict[str, List[str]]]:
        """Create labels using provided dimensions."""
        loc_labels = [f"r{r+1}c{c+1}" for r in range(rows) for c in range(cols)]
        return {
            "state_factors": {"Location": loc_labels},
            "observation_modalities": {"Location": loc_labels},
            "control_factors": {"Move": ["Up", "Left", "Right", "Down", "Stay"]},
        }

    def _initialize_default_labels(self) -> Dict[str, Dict[str, List[str]]]:
        """Fallback method for creating default labels."""
        return self._create_labels(self.rows, self.cols)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(self, mode: str = "human", observations: List[jnp.ndarray] | None = None):
        """Minimalistic grid rendering showing agent position.

        Parameters
        ----------
        mode : {{"human", "rgb_array"}}
            If "human", display the plot with *matplotlib*.  If "rgb_array",
            return the image as a numpy array.
        observations : list, optional
            List of observation arrays (as returned by environment `step`).  If
            *None*, the environment's current observation is used.
        """
        # determine batch + agent positions
        if observations is not None:
            obs = observations[0]  # first modality
        else:
            obs = self.current_obs[0]
        batch_size = obs.shape[0]

        n_plots = int(jnp.ceil(jnp.sqrt(batch_size)))
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(3 * n_plots, 3 * n_plots))

        def _get_ax(i):
            if batch_size == 1:
                return axes
            row, col = divmod(i, n_plots)
            return axes[row, col]

        for i in range(batch_size):
            ax = _get_ax(i)
            ax.set_aspect("equal")
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-0.5, self.rows - 0.5)
            ax.invert_yaxis()  # origin at top-left like matrix notation
            ax.set_xticks(range(self.cols))
            ax.set_yticks(range(self.rows))
            ax.grid(True, color="lightgray", linewidth=0.5)

            # agent position
            s = int(obs[i, 0]) if obs.ndim == 2 else int(obs[i])
            r, c = divmod(s, self.cols)
            circ = patches.Circle((c, r), 0.3, facecolor="tab:orange", edgecolor="black")
            ax.add_patch(circ)
            ax.set_title(f"Batch {i}")

        plt.tight_layout()
        if mode == "human":
            plt.show()
            return None
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig)
            return img
        else:
            raise ValueError("mode must be 'human' or 'rgb_array'")

    # ------------------------------------------------------------------
    # defaults for agent creation convenience
    # ------------------------------------------------------------------
    def get_default_model_params(self) -> Dict[str, List[jnp.ndarray]]:
        return super().get_default_model_params()

    def get_default_C(self):
        # zero preferences already handled by parent class
        return super().get_default_C()

    def get_default_agent_params(self):
        return super().get_default_agent_params()
