import contextlib
from arg_parser import parse
from pathlib import Path
from continuous_cartpole import ContinuousCartPoleEnv
from actor_critic_discrete import ActorCritic
from utils import D2C, Visualizer
from utils import reward_carrot_stick

if __name__ == '__main__':
    print('--- running main ---')
    args = parse()
    env = ContinuousCartPoleEnv(reward_function=reward_carrot_stick)
    state_dim = env.observation_space.shape[0]
    action_dim = args.action_dim
    d2c_converter = D2C(action_dim, env.action_space.low,
                        env.action_space.high)

    # --- choose algorithm and hyperparameters ---
    actorcritic = ActorCritic(state_dim,
                              action_dim,
                              gamma=args.gamma,
                              d2c=d2c_converter)
    episodes = args.episode
    time_steps = args.steps
    render =  args.render
    # --- run algorithm ---
    actorcritic.load_models()
    stats = actorcritic.train(env, episodes, time_steps)
    actorcritic.save_models()

    # --- visualize the results ---
    result_folder = Path(__file__).resolve().parent / 'results'
    viz = Visualizer(result_path=result_folder)
    viz.plot_episode_length(stats, plot_name='ac_episode_length_{}'.format(args.exp_count))
    viz.plot_reward(stats, plot_name='ac_rewards_{}'.format(args.exp_count))

    # --- animation ---
    if render:
        with contextlib.closing(ContinuousCartPoleEnv()) as env:
            for _ in range(5):
                s = env.reset()
                for _ in range(300):
                    env.render()
                    a, _ = actorcritic.get_action(s)
                    s, _, d, _ = env.step(a)
                    if d:
                        break
