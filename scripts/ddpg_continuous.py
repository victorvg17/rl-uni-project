from pathlib import Path
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PriorityQueue, ModelIO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'done'])

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class ReplayBuffer(PriorityQueue):
    def __init__(self, max_size):
        super().__init__(max_size=max_size)

    def add_transition(self,
                       state,
                       action,
                       next_state,
                       reward,
                       done,
                       priority=0):
        trans = Transition(state, action, next_state, reward, done)
        self.push(item=trans, priority=priority)

    def next_random_batch(self, batch_size):
        return self.next_batch(batch_size, choose_random=True)

    def next_batch(self, batch_size, choose_random=False):
        # batch is a list of [priority, count, transition]
        feasible_batch_size = min(self.size(), batch_size)
        if choose_random:
            batch = self.sample(batch_size=feasible_batch_size)
        else:
            batch = self.peeknsmallest(batch_size=feasible_batch_size)
        states = [b[2].state for b in batch]
        actions = [b[2].action for b in batch]
        next_states = [b[2].next_state for b in batch]
        rewards = [b[2].reward for b in batch]
        dones = [b[2].done for b in batch]
        return (states, actions, next_states, rewards, dones)


class Actor(nn.Module):
    def __init__(self, state_dim, noise_std=0.02, hidden_dim=64):
        super().__init__()
        self.noise_std = noise_std
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(), nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Tanh())
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, state):
        """
        Note that state can be a batch of states
        """
        state = torch.from_numpy(state).float().to(device) if isinstance(
            state, np.ndarray) else state
        action_and_noise = self.block(state) + torch.randn(1) * self.noise_std
        return torch.clamp(action_and_noise, min=-1.0, max=1.0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim + action_dim,
                      out_features=hidden_dim), nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=action_dim))
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, state, action):
        """
        Note that state can be a batch of states
        Note that action can be a batch of actions
        """
        state = torch.from_numpy(state).float().to(device) if isinstance(
            state, np.ndarray) else state
        action = torch.from_numpy(action).float().to(device) if isinstance(
            action, np.ndarray) else action
        state_action = torch.cat((state, action), dim=-1)
        return self.block(state_action)


class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 gamma=0.99,
                 noise_std=0.02,
                 hidden_dim=64,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 verbose=False):
        self.gamma = gamma
        self.tau = 0.01
        self.actor = Actor(state_dim,
                           noise_std=noise_std,
                           hidden_dim=hidden_dim)
        self.actor_target = Actor(state_dim,
                                  noise_std=noise_std,
                                  hidden_dim=hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim)
        self.critic_target = Critic(state_dim,
                                    action_dim,
                                    hidden_dim=hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.buffer = ReplayBuffer(max_size=1e5)
        self.logging_period = 10 if verbose else 1000
        # --- ModelIO ---
        self.modelio = ModelIO(model_path=Path(__file__).resolve().parent /
                               'models')

    def update_target(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data *
                                    (1.0 - self.tau))

    def get_action(self, state):
        """
        used for test time (not training)
        """
        action = self.actor(state).detach()
        env_action = torch.clamp(action, min=-1.0, max=1.0).detach().numpy()
        return env_action

    def train(self, env, episodes, timesteps):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes))
        for i_episode in range(1, episodes + 1):
            state = env.reset()
            for t in range(timesteps):
                # --- choose action
                action = self.actor(state).detach()
                env_action = torch.clamp(action, min=-1.0,
                                         max=1.0).detach().numpy()
                next_state, reward, done, _ = env.step(env_action)

                # --- saving stats
                stats.episode_rewards[i_episode - 1] += reward
                stats.episode_lengths[i_episode - 1] = t

                # --- save the transision
                self.buffer.add_transition(
                    state=torch.from_numpy(state).float().to(device),
                    action=action,
                    next_state=torch.from_numpy(next_state).float().to(device),
                    reward=reward,
                    done=done)

                # --- sample a batch of transitions
                batch = self.buffer.next_random_batch(batch_size=32)

                # --- train
                self.train_batch(batch)

                # --- update target networks
                self.update_target(target=self.actor_target, source=self.actor)
                self.update_target(target=self.critic_target,
                                   source=self.critic)

                if done:
                    break
                state = next_state

            # logging
            if i_episode % self.logging_period == 0:
                print((f"{int(stats.episode_lengths[i_episode - 1])} Steps in"
                       f"Episode {i_episode}/{episodes}. "
                       f"Reward {stats.episode_rewards[i_episode-1]}"))

        return stats

    def train_batch(self, batch):
        states, actions, next_states, rewards, dones = batch
        batch_rewards = torch.FloatTensor(rewards).to(device)
        batch_states = torch.stack(states).to(device)
        batch_actions = torch.stack(actions).to(device)
        batch_next_states = torch.stack(next_states).to(device)

        batch_na = self.actor_target(batch_next_states)
        batch_q_ns_na = self.critic_target(batch_next_states,
                                           batch_na.detach().view(-1, 1))
        update_targets = batch_rewards.view(-1, 1) + self.gamma * batch_q_ns_na
        batch_q_s_a = self.critic(batch_states, batch_actions.view(-1, 1))
        critic_loss = F.mse_loss(batch_q_s_a, update_targets)

        actor_loss = -self.critic(batch_states,
                                  self.actor(batch_states).view(-1, 1)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_models(self, model_name):
        self.modelio.save(model=self.actor,
                          model_name=f'ddpg_c_actor_{model_name}.pt')
        self.modelio.save(model=self.critic,
                          model_name=f'ddpg_c_critic_{model_name}.pt')

    def load_models(self, model_name):
        # if self._model
        self.modelio.load(model=self.actor,
                          model_name=f'ddpg_c_actor_{model_name}.pt')
        self.modelio.load(model=self.actor_target,
                          model_name=f'ddpg_c_actor_{model_name}.pt')
        self.modelio.load(model=self.critic,
                          model_name=f'ddpg_c_critic_{model_name}.pt')
        self.modelio.load(model=self.critic_target,
                          model_name=f'ddpg_c_critic_{model_name}.pt')
