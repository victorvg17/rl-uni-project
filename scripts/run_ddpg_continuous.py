import contextlib
from pathlib import Path
from arg_parser import parse
from continuous_cartpole import ContinuousCartPoleEnv
from ddpg_continuous import DDPG
from utils import Visualizer
from utils import reward_carrot_stick, reward_no_fast_rotation, reward_laplacian
from utils import reward_func_map

if __name__ == '__main__':

    print('--- running main ---')
    args = parse()

    # ========== Parameters ==========
    reward_func = reward_func_map[args.reward_function]
    env = ContinuousCartPoleEnv(reward_function=reward_func)
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    noise_std = args.action_std
    episodes = args.episode
    timesteps = args.steps
    update_timestep = args.update_timesteps
    gamma = 0.99
    entropy_coeff = args.entropy_coeff
    K_epochs = args.K_epochs
    eps_clip = args.epsilon_clip
    hidden_dim = args.hidden_dim
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    exp_count = args.exp_count
    render_flag = args.render
    load_flag = args.load
    verbose_flag = args.verbose
    # ================================

    ddpg = DDPG(state_dim,
                action_dim,
                gamma=gamma,
                noise_std=noise_std,
                hidden_dim=hidden_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                verbose=verbose_flag)

    if load_flag:
        ddpg.load_models(model_name=exp_count)
    stats = ddpg.train(env=env, episodes=episodes, timesteps=timesteps)
    ddpg.save_models(model_name=exp_count)

    # --- visualize the results ---
    result_folder = Path(__file__).resolve().parent / 'results'
    viz = Visualizer(result_path=result_folder)
    viz.plot_episode_length(stats, plot_name=f'ddpg_c_episodes_{exp_count}')
    viz.plot_reward(stats, plot_name=f'ddpg_c_rewards_{exp_count}')

    # --- animation ---
    if render_flag:
        with contextlib.closing(ContinuousCartPoleEnv()) as env:
            for _ in range(3):
                s = env.reset()
                for _ in range(500):
                    env.render()
                    a = ddpg.get_action(s)
                    s, _, d, _ = env.step(a)
                    if d:
                        break
