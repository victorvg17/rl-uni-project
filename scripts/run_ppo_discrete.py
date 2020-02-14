import contextlib
from pathlib import Path
from arg_parser import parse
from continuous_cartpole import ContinuousCartPoleEnv
from ppo_discrete import PPO
from utils import D2C, Visualizer
from utils import reward_laplacian, reward_carrot_stick, reward_no_fast_rotation
from utils import reward_func_map

if __name__ == '__main__':
    print('--- running main ---')
    args = parse()

    # ========== Parameters ==========
    reward_fun = reward_func_map[args.reward_function]
    env = ContinuousCartPoleEnv(reward_function=reward_fun)
    state_dim = env.observation_space.shape[0]
    action_dim = args.action_dim
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
    # ================================

    d2c_converter = D2C(action_dim, env.action_space.low,
                        env.action_space.high)
    ppo = PPO(state_dim,
              action_dim,
              gamma=gamma,
              K_epochs=K_epochs,
              eps_clip=eps_clip,
              entropy_coeff=entropy_coeff,
              d2c=d2c_converter)
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
    viz.plot_episode_length(stats, plot_name=f'ppo_d_episodes_{exp_count}')
    viz.plot_reward(stats, plot_name=f'ppo_d_rewards_{exp_count}')

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
