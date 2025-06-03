"""Quick sanity-check script for the `GridWorldEnv`.

Run this file to make sure the new grid-world environment is functional and
plays nicely with the existing structure-learning utilities.  The script:

1. Instantiates a *3 × 4* grid-world with batch-size 1.
2. Renders the initial grid so you can visually confirm the agent starts in the
   top-left corner.
3. Creates an agent directly from the environment (**perfect model**, no
   parameter learning) and runs a short multi-trial rollout.
4. Prints and plots useful diagnostics (initial state, roll-out, prediction
   errors) so any anomalies become obvious.

Feel free to tweak `rows`, `cols`, learning flags, or the number of trials to
explore different scenarios.
"""
#%%
from jax import random as jr
import matplotlib.pyplot as plt

import pymdp
from pymdp.learning import LearningConfig
from pymdp.envs.rollout import rollout
from pymdp.maths import compute_prediction_errors
from pymdp.analysis import (
    plot_prediction_errors,
    print_initial_state,
    print_rollout,
    render_rollout,
)
from pymdp.agent import Agent
from pymdp.envs.gridworld import GridWorldEnv



# ------------------------------------------------------------------
# environment setup & initial render
# ------------------------------------------------------------------
env = GridWorldEnv(rows=3, cols=3, batch_size=1)
env.render()  # visual sanity-check (agent should be at r1c1)

# ------------------------------------------------------------------
# agent setup — perfect model, no learning for this quick test
# ------------------------------------------------------------------
workspace_agent_params = env.get_default_agent_params()
workspace_agent_params.update(
    {
        "action_selection": "stochastic",
        "use_param_info_gain": False,
        "use_states_info_gain": False,
        "learning_mode": "online",
    }
)

learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

key = jr.PRNGKey(0)
agent, model, key = Agent.from_env(
    env=env,
    learning_config=learning_config,
    key=key,
    model_params={"T": 100},
    agent_params=workspace_agent_params,
)

# ------------------------------------------------------------------
# rollout
# ------------------------------------------------------------------
_, info, _ = rollout(
    agent,
    env,
    num_timesteps=model.structure.T,
    rng_key=key,
)

# ------------------------------------------------------------------
# diagnostics
# ------------------------------------------------------------------
print_initial_state(info, trial_idx=0)
print_rollout(info, batch_idx=0, trials=0)

# Render the rollout to visualize agent behavior
render_rollout(env, info,fps=2)

pe_analysis = compute_prediction_errors(info)
plot_prediction_errors(
    pe_analysis,
    yscale="linear",
    smoothing=None,
    num_trials=1,
    trial_lines=False,
    title="GridWorld prediction error (perfect model)",
)

plt.show()

#%%