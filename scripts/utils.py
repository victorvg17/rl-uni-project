import itertools
from pathlib import Path
from collections import namedtuple
from heapq import heappush, heapreplace, nsmallest
import numpy as np
import pandas as pd
from scipy.stats import laplace
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'done'])


class D2C:
    def __init__(self, action_dim, low, high):
        """
        Args:
            action_dim: resolution for discretization.
                        This has to match the action_dim
                        passed to the learning algorithm.
            low, high: action_space.low, action_space.high
        """
        self._action_dim = action_dim
        self._low = low
        self._high = high

    def convert(self, action):
        """
        Converts a discrete action to a continuous aciton.

        Args:
            action: discrete action

        Returns:
            A float in range [low, high] that corresponds
            to the discrete action in continuous range.
        """
        return np.linspace(self._low, self._high, num=self._action_dim)[action]

    def __call__(self, action):
        return self.convert(action)


class ModelIO:
    def __init__(self, model_path):
        """
        model_path: path to the model folder
        """
        self._model_path = model_path

    def save(self, model, model_name):
        """
        saves the model with model_name

        Args:
            model: torch_model
            model_name: name of the model
        """
        state_dict = model.state_dict()
        Path.mkdir(self._model_path, parents=True, exist_ok=True)
        dest = self._model_path / model_name
        torch.save(state_dict, dest)

    def load(self, model, model_name):
        """
        loads the model with model_name

        Args:
            model: torch_model
            model_name: name of the model
        """
        try:
            location = self._model_path / model_name
            model.load_state_dict(torch.load(location))
        except FileNotFoundError as ex:
            print(f'--- WARNING --- Could Not Load Model: {ex}')


class Visualizer:
    def __init__(self, result_path, showplots=True):
        self._result_path = result_path
        self._showplots = showplots

    def plot_episode_length(self, stats, plot_name='episode_length'):
        smoothing_window = max(1, len(stats.episode_lengths) // 50)
        fig, ax = plt.subplots()
        smoothed_lengths = pd.Series.rolling(pd.Series(stats.episode_lengths),
                                             smoothing_window).mean()
        smoothed_lengths = [elem for elem in smoothed_lengths]
        ax.plot(smoothed_lengths, alpha=0.9, c='indigo')
        ax.plot(stats.episode_lengths, alpha=0.3, c='indigo', linewidth='2')
        ax.set_title(
            f'Episode Length over Time (Smoothing: {smoothing_window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_facecolor('ghostwhite')
        for key, s in ax.spines.items():
            s.set_visible(False)
        ax.grid()

        Path.mkdir(self._result_path, parents=True, exist_ok=True)
        fig.savefig(self._result_path / f'{plot_name}.png')
        if self._showplots:
            plt.show(fig)
        else:
            plt.close(fig)

    def plot_reward(self, stats, plot_name='rewards'):
        smoothing_window = max(1, len(stats.episode_lengths) // 50)
        fig, ax = plt.subplots()
        smoothed_rewards = pd.Series.rolling(pd.Series(stats.episode_rewards),
                                             smoothing_window).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        ax.plot(smoothed_rewards, alpha=0.9, c='indigo')
        ax.plot(stats.episode_rewards, alpha=0.3, c='indigo', linewidth='2')
        ax.set_title(
            f'Episode Reward over Time (Smoothing: {smoothing_window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Rewards (Smoothed)')
        ax.set_facecolor('ghostwhite')
        for key, s in ax.spines.items():
            s.set_visible(False)
        ax.grid()

        Path.mkdir(self._result_path, parents=True, exist_ok=True)
        fig.savefig(self._result_path / f'{plot_name}.png')
        if self._showplots:
            plt.show(fig)
        else:
            plt.close(fig)


class PriorityQueue:
    def __init__(self, max_size):
        self._max_size = max_size
        self._q = []
        self._counter = itertools.count()  # unique sequence count

    def size(self):
        return len(self._q)

    def push(self, item, priority=0):
        count = next(self._counter)
        if len(self._q) < self._max_size:
            heappush(self._q, [priority, count, item])
        else:
            heapreplace(self._q, [priority, count, item])

    def peeknsmallest(self, n):
        """
        return largest n items without poping them
        """
        num_samples = min(n, len(self._q))
        return nsmallest(num_samples, self._q)

    def sample(self, batch_size):
        idx = np.random.choice(len(self._q), size=batch_size, replace=True)
        samples = [self._q[i] for i in idx]
        return samples


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self,
                 action_space,
                 mu=0.0,
                 theta=0.15,
                 max_sigma=0.3,
                 min_sigma=0.3,
                 decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(
            self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


# in house reward function.
def reward_laplacian(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    # return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else -0.4
    theta_normalise = angle_normalize(cart_pole.state[2])
    # if -0.1 <= theta_normalise <= 0.1:
    reward_lapl = 2 * laplace.pdf(theta_normalise)
    # print(f"inside inhouse reward")
    return reward_lapl


def reward_carrot_stick(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    bonus = 1 if -0.1 <= theta_normalise <= 0.1 else 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward

def reward_test_1(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    bonus = 1 if -0.2 <= theta_normalise <= 0.2 else 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward

def reward_test_2(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    if abs(theta_normalise) < 0.1: 
        bonus = 10  
    # elif abs(theta_normalise) < 0.2:
    #     bonus = 1 * 5e-2
    elif abs (theta_normalise) < 0.5:
        bonus = 1 * 1e-1
    else:
        bonus = 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward

def reward_test_3(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    if abs(theta_normalise) < 0.2: 
        bonus = 1 * 8e-2
    elif abs(theta_normalise) < 0.4:
        bonus = 1 * 4e-2
    elif abs (theta_normalise) < 0.6:
        bonus = 1 * 2e-2
    else:
        bonus = 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward

def reward_test_4(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    if abs(theta_normalise) < 0.1: 
        bonus = 2 
    elif abs(theta_normalise) < 0.2:
        bonus = 1 * 1e-1
    elif abs(theta_normalise) < 0.4:
        bonus = 1 * 2e-2
    # elif abs (theta_normalise) < 0.6:
    #     bonus = 1 * 5e-2
    else:
        bonus = 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward


def reward_test_5(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    if abs(theta_normalise) < 0.2: 
        bonus = 1 * 9e-1
    elif abs (theta_normalise) < 0.5:
        bonus = 1 * 1e-2
    else:
        bonus = 0
    reward = -abs(theta_normalise) * 1e-4 + (
        3.14 - abs(theta_normalise)) * 1e-5 + bonus
    return reward


def reward_test_6(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    if abs(theta_normalise) < 0.2: 
        bonus = 10
        ang_vel_penalty = (((cart_pole.state[3])**2 + cart_pole.state[1])**2 ) * 2e-1
    elif abs(theta_normalise) < 0.5:
        bonus = 2
        ang_vel_penalty = (((cart_pole.state[3])**2 + cart_pole.state[1])**2 ) * 1e-2
    else:
        ang_vel_penalty = 0
        bonus = 0
    reward = (np.pi - abs(theta_normalise))* 1e-2 + bonus - ang_vel_penalty
    return reward


def reward_mith_velocity_penalization(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    bonus = 1 if -0.1 <= theta_normalise <= 0.1 else 0
    # penelize the pole velocity when pole is near upright
    ang_vel_penalty = (abs(theta_normalise) < 0.2) * (
        (cart_pole.state[3]**2 + cart_pole.state[1]**2) * 2e-3)
    reward = bonus - ang_vel_penalty + (3.14 - abs(theta_normalise)) * 1e-4
    return reward


def reward_no_fast_rotation(cart_pole):
    x_threshold = 2.4
    if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
        return -1
    theta_normalise = angle_normalize(cart_pole.state[2])
    bonus = 1 if -0.1 <= theta_normalise <= 0.1 else 0
    # penelize the pole velocity when pole is near upright
    ang_vel_penalty = (abs(theta_normalise) < 0.2) * (
        (cart_pole.state[3]**2) * 2e-3)
    reward = bonus - ang_vel_penalty
    return reward


# exponential decay function for varying exploration
def eps_decay_fun(time_step, eps_start=0.9, eps_end=0.05, eps_decay=200):
    eps = eps_end + (eps_start - eps_end) * np.exp(-time_step / eps_decay)
    return eps


def tt(ndarray, requires_grad=False, cuda=False):
    if cuda:
        return Variable(torch.from_numpy(ndarray).float().cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(ndarray).float(),
                        requires_grad=requires_grad)


def save_config_colab(hyper_params: dict, experiment: int, algo_name: str):
    """
    To be used only in colab
    """

    config_path = Path('./content').resolve().parent / 'config'
    name = config_path / f'exp_{algo_name}_{experiment}.txt'
    Path.mkdir(config_path, parents=True, exist_ok=True)

    f = open(name, "w+")
    f.write(str(hyper_params))
    f.close()


# ========== maps ==========

reward_func_map = {
    'sparse': None,
    'carrot': reward_carrot_stick,
    'laplace': reward_laplacian,
    'slow_rotation': reward_no_fast_rotation,
    'test': reward_test_1,
    'test_v2': reward_test_2,
    'test_v3': reward_test_3,
    'test_v4': reward_test_4,
    'test_v5': reward_test_5,
    'test_v6': reward_test_6,
    'penalty': reward_mith_velocity_penalization,
}
