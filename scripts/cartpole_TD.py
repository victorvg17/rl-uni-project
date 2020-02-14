import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import namedtuple

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, action_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x):
        x = self.block(x)
        return x

class Network(nn.Module):
    def __init__(self, state_dim, non_linearity=F.relu, hidden_dim=10):
        super(Network, self).__init__()


        self.block = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, out_features=1))
        self._init_weights()

    def forward(self, x):
        x = self.block(x)
        return x

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

class TDLambda:
    def __init__(self, state_dim, action_dim, gamma, trace_decay, alpha, d2c):
        self._q = Q(state_dim, action_dim)
        self.v = Network(state_dim)
        self.z = Network(state_dim)

        self.gamma = gamma
        self.trace_decay = trace_decay
        self.alpha = alpha

        self.z._init_weights()

        self._q = Q(state_dim, action_dim)


        # self._gamma = gamma
        # self._loss_function = nn.MSELoss()
        # self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001)
        self._action_dim = action_dim
        self._d2c = d2c

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if (r < epsilon):
            u = np.random.randint(self._action_dim)
        if self._d2c:
            u = self._d2c(u)
        return u

    def train(self, env, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes))
        for e in range(episodes):
            s = env.reset()
            # self.z._init_weights()

            for t in range(time_steps):
                a = self.get_action(s, epsilon)

                ns, r, d, _ = env.step(a)

                stats.episode_rewards[e] += r
                stats.episode_lengths[e] = t

                self.v.zero_grad()
                c_state = tt(np.array([s]))
                self.v(c_state).mean().backward()
                for z_param, v_param in zip(self.z.parameters(), self.v.parameters()):
                    z_param.data.copy_(self.gamma * self.trace_decay * z_param.data + v_param.grad.data)

                n_state = tt(np.array([ns]))
                td_error = r + (1-d) * self.gamma * (self.v(n_state) - self.v(c_state))

                for z_param, v_param in zip(self.z.parameters(), self.v.parameters()):
                    update_val = v_param.data + self.alpha * td_error * z_param.data
                    v_param.data.copy_(update_val.squeeze())

                if d:
                  break

                s = ns
            # logging time.
            if (e+1)%100 == 0:
                print("episode: %s/%s"%(e+1, episodes))
            print(
                f"{int(stats.episode_lengths[e])} Steps in Episode {e}/{episodes}. Reward {int(stats.episode_rewards[e])}"
            )
        return stats
