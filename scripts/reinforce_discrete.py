from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from utils import ModelIO, tt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class StateValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=50):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LeakyReLU(), nn.Linear(in_features=hidden_dim, out_features=1))
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x):
        return self.block(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=50):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_dim),
            nn.Softmax(dim=0))
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x):
        return self.block(x)


class REINFORCE:
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.99,
                 hidden_dim=64,
                 policy_lr=0.001,
                 baseline_lr=0.001,
                 d2c=None):
        self._V = StateValueFunction(state_dim, hidden_dim=hidden_dim)
        self._pi = Policy(state_dim, action_dim, hidden_dim=hidden_dim)
        self.d2c = d2c  # discrete to continuous actions
        # self._V.cuda()
        # self._pi.cuda()
        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._V_optimizer = optim.Adam(self._V.parameters(), lr=baseline_lr)
        self._pi_optimizer = optim.Adam(self._pi.parameters(), lr=policy_lr)
        self._action_dim = action_dim
        # --- ModelIO ---
        self._modelio = ModelIO(model_path=Path(__file__).resolve().parent /
                                'models')

    def get_action(self, s):
        probs = self._pi(tt(s))
        action = np.random.choice(a=self._action_dim,
                                  p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[action])

        # converting the discrete action [0,1,2,...]
        # to an action in the continuous
        # range (actionspace.low <--> actionspace.high)
        if self.d2c:
            action = self.d2c(action)

        return action, log_prob

    def train(self, env, episodes, time_steps):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes))

        for i_episode in range(1, episodes + 1):
            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            s = env.reset()
            for t in range(time_steps):
                a, log_prob_a = self.get_action(s)
                ns, r, d, _ = env.step(a)

                stats.episode_rewards[i_episode - 1] += r
                stats.episode_lengths[i_episode - 1] = t

                episode.append((s, a, log_prob_a, r))

                if d:
                    break
                s = ns
            T = len(episode)
            G = 0.0
            for t in reversed(range(T)):
                s, a, log_prob_a, r = episode[t]
                G = self._gamma * G + r
                baseline = self._V(tt(s))
                advantage = G - baseline
                self._train_baseline(G, baseline)
                self._train_policy(advantage, t, log_prob_a)

            print("\r{} Steps in Episode {}/{}. Reward {}".format(
                len(episode), i_episode, episodes,
                sum([e[3] for i, e in enumerate(episode)])))
        return stats

    def _train_policy(self, G, t, log_prob_a):
        self._pi_optimizer.zero_grad()
        neg_log_prob_a = -log_prob_a
        target = np.power(self._gamma, t) * G * neg_log_prob_a
        target.backward()
        self._pi_optimizer.step()

    def _train_baseline(self, G, baseline):
        self._V_optimizer.zero_grad()
        loss = self._loss_function(tt(np.array([G])), baseline)
        loss.backward(retain_graph=True)
        self._V_optimizer.step()

    def save_models(self, model_name):
        self._modelio.save(model=self._pi,
                           model_name=f'r_d_policy_{model_name}.pt')
        self._modelio.save(model=self._V,
                           model_name=f'r_d_baseline_{model_name}.pt')

    def load_models(self, model_name):
        # if self._model
        self._modelio.load(model=self._pi,
                           model_name=f'r_d_policy_{model_name}.pt')
        self._modelio.load(model=self._V,
                           model_name=f'r_d_baseline_{model_name}.pt')
