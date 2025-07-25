from jax import random as jr

from pymdp.analysis import print_initial_state, print_rollout, print_parameter_learning, plot_prediction_errors, \
    plot_rollout_preferences, plot_parameter_learning
from pymdp.envs.rollout import multi_trial_rollout
from pymdp.learning import LearningConfig
from pymdp.envs.env_factory import make, EnvType
from pymdp.agent import Agent
from pymdp.maths import compute_prediction_errors, compute_preferences

if __name__ == "__main__":

    #    class A:
    #
    #        def __init__(self):
    #            self.a = 1
    #
    #        def foo(self):
    #            print(self.a)
    #
    #    class B(A):
    #
    #        def __init__(self):
    #            super().__init__()
    #            self.b = 2
    #
    #    b = B()
    #    b.foo()

    # Initialize jax random key.
    key = jr.PRNGKey(1)

    # Initialize environment.
    env = make(EnvType.SIMPLEST, batch_size=32)

    # Initialize agent.
    learning_config = LearningConfig(learn_A=True, learn_B=True, learn_D=False)

    agent, model, key = Agent.from_env(
        env=env,
        learning_config=learning_config,
        key=key,
        model_params={"T": 10}
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
    num_trials = 1
    _, key, info = multi_trial_rollout(agent, env, num_timesteps=model.structure.T, num_trials=num_trials, rng_key=key)

    # Analysis of simulation results
    # TODO print_initial_state(info, trial_idx=num_trials - 1)
    # TODO print_rollout(info, batch_idx=0, trials=num_trials - 1)
    # TODO plot_parameter_learning(info, learning_config, env, trial_lines=False)
    # TODO print_parameter_learning(info, learning_config, env)
    # TODO pe_analysis = compute_prediction_errors(info)
    # TODO plot_prediction_errors(pe_analysis, yscale='linear', smoothing=None, num_trials=num_trials, trial_lines=False)
