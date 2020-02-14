import contextlib
from arg_parser import parse
from pathlib import Path
from continuous_cartpole import ContinuousCartPoleEnv
from cartpole_TD import TDLambda
from utils import D2C, Visualizer, reward_laplacian


if __name__ == '__main__':
    print('--- running main ---')
    args = parse()
    env = ContinuousCartPoleEnv(reward_function=reward_laplacian)
    state_dim = env.observation_space.shape[0]
    action_dim = args.action_dim
    d2c_converter = D2C(action_dim, env.action_space.low,
                        env.action_space.high)

    # --- choose algorithm and hyperparameters ---
    # dqn = DQN(state_dim, action_dim, gamma=0.99, d2c=d2c_converter)
    # state_dim, action_dim, gamma, trace_decay, alpha, d2c
    td_lam = TDLambda(state_dim, action_dim,
                        gamma=args.gamma,
                        trace_decay=args.trace_decay,
                        alpha=args.alpha,
                        d2c=d2c_converter)

    episodes = args.episode
    time_steps = args.steps
    epsilon = args.epsilon
    render = args.render

    # --- run algorithm ---
    # td_lam.load_models()
    stats = td_lam.train(env, episodes, time_steps, epsilon)
    # td_lam.save_models()

    # --- visualize the results ---
    result_folder = Path(__file__).resolve().parent / 'results'
    viz = Visualizer(result_path=result_folder)
    viz.plot_episode_length(stats, plot_name='td_episode_length_{}'.format(args.exp_count))
    viz.plot_reward(stats, plot_name='td_rewards_{}'.format(args.exp_count))

    # --- animation ---
    if render:
        with contextlib.closing(ContinuousCartPoleEnv()) as env:
            for _ in range(2):
                s = env.reset()
                for _ in range(300):
                    env.render()
                    a = td_lam.get_action(s, epsilon=0.02)
                    s, _, d, _ = env.step(a)
                    if d:
                        break
