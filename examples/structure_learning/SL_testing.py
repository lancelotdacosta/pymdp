from jax import random as jr

from pymdp.analysis import print_initial_state, print_rollout, print_parameter_learning, plot_parameter_learning, \
    plot_prediction_errors
from pymdp.envs.rollout import multi_trial_rollout
from pymdp.learning import LearningConfig
from pymdp.envs.env_factory import make, EnvType
from pymdp.agent import Agent
from pymdp.maths import compute_prediction_errors


if __name__ == "__main__":

    # Initialize jax random key.
    key = jr.PRNGKey(8)

    # Initialize environment.
    env = make(EnvType.GRIDWORLD, batch_size=1)

    # Initialize agent.
    learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

    agent, model, key = Agent.from_env(
        env=env,
        learning_config=learning_config,
        key=key,
        agent_params={
            "policy_len": 1,
            "inference_algo": "fpi",
            "apply_batch": False,
            "use_param_info_gain": True,
            "use_states_info_gain": True,
            "action_selection": "stochastic",
            "learning_mode": "offline"
        },
        model_params={
            "T": 100,
            "init": "gaussian",
            "scale": 0.01
        }
    )

    # Display the tensor's shapes.
    print("D => Number of factors: ", len(agent.D))
    for i, D in enumerate(agent.D):
        print(f" - Factor[{i}]: ", D.shape)  # (batch_size, # of states)
    print()
    print("A => Number of modalities: ", len(agent.A))
    for i, (A, A_deps) in enumerate(zip(agent.A, agent.A_dependencies)):
        print(f" - Modality[{i}]: ", A.shape, end="")  # (batch_size, # of obs, # of states, ...)
        print(f", depends on factors: ", A_deps)  # [factor index, ...]
    print()
    print("B => Number of factors: ", len(agent.B))
    for i, (B, B_deps, B_action_deps) in enumerate(zip(agent.B, agent.B_dependencies, agent.B_action_dependencies)):
        print(f" - Factor[{i}]: ", B.shape, end="")  # (batch_size, # of states, # of states, ..., # of actions)
        print(f", depends on factors: ", B_deps, end="")  # [factor index, ...]
        print(", and actions: ", B_action_deps)  # [action index]
    print()

    # Run simulation.
    num_trials = 1000
    _, key, info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

    # Analysis of simulation results
    print_initial_state(info, trial_idx=num_trials - 1)
    print_rollout(info, batch_idx=0, trials=num_trials - 1)
    plot_parameter_learning(info, learning_config, env, trial_lines=False)
    print_parameter_learning(info, learning_config, env, display_dirichlet_counts=True)
    pe_analysis = compute_prediction_errors(info)
    plot_prediction_errors(pe_analysis, yscale='linear', smoothing=100, num_trials=num_trials, trial_lines=False)
