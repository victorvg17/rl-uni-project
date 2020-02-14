import contextlib
from pathlib import Path
from arg_parser import parse
from continuous_cartpole import ContinuousCartPoleEnv
from ppo_continuous import PPO
from utils import Visualizer
from utils import reward_carrot_stick, reward_no_fast_rotation, reward_laplacian
from utils import reward_func_map

if __name__ == '__main__':
    # --- EXAMPLE RUN
    # python run_ppo_continuous.py -e 500 -s 500 -rw 'sparse' -hd 64 -alr 0.0001 -clr 0.0001 -ke 10 -ad 0.1 -eco 0.02 -ecp 0.2 -us 1024

    print('--- running main ---')
    args = parse()

    # ========== Parameters ==========
    reward_func = reward_func_map[args.reward_function]
    env = ContinuousCartPoleEnv(reward_function=reward_func)
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    action_std = args.action_std
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

    ppo = PPO(state_dim,
              action_dim,
              action_std,
              gamma=gamma,
              hidden_dim=hidden_dim,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              K_epochs=K_epochs,
              eps_clip=eps_clip,
              entropy_coeff=entropy_coeff,
              verbose=verbose_flag)
    if load_flag:
        ppo.load_models(model_name=exp_count)
    stats = ppo.train(env=env,
                      episodes=episodes,
                      timesteps=timesteps,
                      update_timestep=update_timestep)
    ppo.save_models(model_name=exp_count)

    # --- visualize the results ---
    result_folder = Path(__file__).resolve().parent / 'results'
    viz = Visualizer(result_path=result_folder)
    viz.plot_episode_length(stats, plot_name=f'ppo_c_episodes_{exp_count}')
    viz.plot_reward(stats, plot_name=f'ppo_c_rewards_{exp_count}')

    # --- animation ---
    if render_flag:
        with contextlib.closing(ContinuousCartPoleEnv()) as env:
            for _ in range(3):
                s = env.reset()
                for _ in range(500):
                    env.render()
                    a = ppo.get_action(s)
                    s, _, d, _ = env.step(a)
                    if d:
                        break
