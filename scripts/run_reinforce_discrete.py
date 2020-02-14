import contextlib
from arg_parser import parse
from pathlib import Path
from continuous_cartpole import ContinuousCartPoleEnv
from reinforce_discrete import REINFORCE
from utils import D2C, Visualizer
from utils import reward_laplacian, reward_carrot_stick, reward_no_fast_rotation
from utils import reward_func_map

if __name__ == '__main__':
    print('--- running main ---')
    args = parse()

    # ============ Parameters ============
    reward_func = reward_func_map[args.reward_function]
    env = ContinuousCartPoleEnv(reward_function=reward_func)
    state_dim = env.observation_space.shape[0]
    action_dim = args.action_dim
    episodes = args.episode
    timesteps = args.steps
    hidden_dim = args.hidden_dim
    policy_lr = args.actor_lr
    baseline_lr = args.critic_lr
    exp_count = args.exp_count
    render_flag = args.render
    load_flag = args.load
    # ====================================

    # --- choose algorithm and hyperparameters ---
    d2c_converter = D2C(action_dim, env.action_space.low,
                        env.action_space.high)

    reinforce = REINFORCE(state_dim,
                          action_dim,
                          gamma=0.99,
                          hidden_dim=hidden_dim,
                          policy_lr=policy_lr,
                          baseline_lr=baseline_lr,
                          d2c=d2c_converter)

    # --- run algorithm ---
    if load_flag:
        reinforce.load_models(model_name=exp_count)
    stats = reinforce.train(env=env, episodes=episodes, time_steps=timesteps)
    reinforce.save_models(model_name=exp_count)

    # --- visualize the results ---
    result_folder = Path(__file__).resolve().parent / 'results'
    viz = Visualizer(result_path=result_folder)
    viz.plot_episode_length(stats, plot_name=f'r_d_episodes_{exp_count}')
    viz.plot_reward(stats, plot_name=f'r_d_rewards_{exp_count}')

    # --- animation ---
    if render_flag:
        with contextlib.closing(ContinuousCartPoleEnv()) as env:
            for _ in range(2):
                s = env.reset()
                for _ in range(500):
                    env.render()
                    a, _ = reinforce.get_action(s)
                    s, _, d, _ = env.step(a)
                    if d:
                        break
