import sys
import argparse


def parse():
    parser = argparse.ArgumentParser(description="Load RL experimet Configuration")
    parser.add_argument('-ec', '--exp_count', type=int, nargs='?', default=1 ,help='an integer showing experiment number')
    parser.add_argument('-e', '--episode', type=int, nargs='?', default=2000 ,help='an integer for the Episode')
    parser.add_argument('-s', '--steps', type=int, nargs='?', default=500 ,help='an integer for the steps')
    parser.add_argument('-ga', '--gamma', type=float, nargs='?', default=0.9 ,help='a float for the gamma')
    parser.add_argument('-ep', '--epsilon', type=float, nargs='?', default=0.3 ,help='a float for the epsilon')
    parser.add_argument('-td', '--trace_decay', type=float, nargs='?', default=0.3 ,help='a float for the epsilon')
    parser.add_argument('-al', '--alpha', type=float, nargs='?', default=0.001 ,help='a float for the alpha')
    parser.add_argument('-ac', '--action_dim', type=int, nargs='?', default=1 ,help='an integer for the action dimension')
    parser.add_argument('-sm', '--smoothing', type=int, nargs='?', default=5 ,help='an integer for the smoothing window')
    parser.add_argument('-rw', '--reward_function', type=str, nargs='?', default="sparse" ,help='a string for the reward function')
    parser.add_argument('-hd', '--hidden_dim', type=int, nargs='?', default=64, help='an integer for the hidden unit size')
    parser.add_argument('-alr', '--actor_lr', type=float, nargs='?', default=0.001 ,help='a float for the actor learning rate')
    parser.add_argument('-clr', '--critic_lr', type=float, nargs='?', default=0.001 ,help='a float for the critic learning rate')
    parser.add_argument('-ke', '--K_epochs', type=int, nargs='?', default=5 ,help='an Integer for the K_epochs')
    parser.add_argument('-ad', '--action_std', type=float, nargs='?', default= 0.02 ,help='a float for the action_deviation value')
    parser.add_argument('-eco', '--entropy_coeff', type=float, nargs='?', default= 0.02, help= 'a float for the entropy coefficient')
    parser.add_argument('-ecp', '--epsilon_clip', type=float, nargs='?', default= 0.2, help='a float for the eplison_clip')
    parser.add_argument('-us', '--update_timesteps', type=int, nargs='?', default=1024, help='an integer for the update_timesteps')
    parser.add_argument('-re', '--render', type=bool, nargs='?', default=False, help='a bool for the render parameter')
    parser.add_argument('-ld', '--load', type=bool, nargs='?', default=False, help='a bool for the render parameter')
    parser.add_argument('-vb', '--verbose', type=bool, nargs='?', default=False, help='a bool for verbosity during training')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('--- running main ---')
    args = parse()
    print(args)
