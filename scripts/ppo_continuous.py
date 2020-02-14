from pathlib import Path
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from utils import ModelIO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(), nn.Linear(in_features=hidden_dim, out_features=1))
        self._init_weights()

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(device) if isinstance(
            state, np.ndarray) else state
        state_value = self.block(state)
        return torch.squeeze(state_value)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_std=0.2, hidden_dim=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=action_dim),
            nn.Tanh())
        self._init_weights()
        self.variance = torch.full((action_dim, ), action_std**2).to(device)

    def _init_weights(self):
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(device)
        mu = self.block(state)
        cov = torch.diag(self.variance).to(device)
        dist = MultivariateNormal(mu, cov)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob

    def evaluate(self, state, action):
        """
        Note that 'state' and 'action' here can be
        batches of states and actions
        """
        mu_batch = self.block(state)
        variance_batch = self.variance.expand_as(mu_batch)
        cov_batch = torch.diag_embed(variance_batch)
        dist = MultivariateNormal(mu_batch, cov_batch)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_std=0.1,
                 gamma=0.99,
                 hidden_dim=64,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 K_epochs=5,
                 eps_clip=0.2,
                 entropy_coeff=0.02,
                 verbose=False):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coeff = entropy_coeff
        self.verbose = verbose

        self.critic = Critic(state_dim, hidden_dim=hidden_dim).to(device)
        self.actor = Actor(state_dim,
                           action_dim,
                           action_std=action_std,
                           hidden_dim=hidden_dim).to(device)
        self.actor_old = Actor(state_dim,
                               action_dim,
                               action_std=action_std,
                               hidden_dim=hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer = Buffer()
        # --- ModelIO ---
        self.modelio = ModelIO(model_path=Path(__file__).resolve().parent /
                               'models')

    def get_action(self, state):
        """
        used for test time (not training)
        """
        action, _ = self.actor_old(state)
        action = torch.clamp(action, min=-1.0, max=1.0).detach().numpy()
        return action

    def train(self, env, episodes, timesteps, update_timestep):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes))
        timestep = 0
        for i_episode in range(1, episodes + 1):
            state = env.reset()
            for t in range(timesteps):
                timestep += 1

                # Running policy_old:
                action, action_logprob = self.actor_old(state)
                env_action = torch.clamp(action, min=-1.0,
                                         max=1.0).detach().numpy()
                next_state, reward, done, _ = env.step(env_action)

                # saving stats
                stats.episode_rewards[i_episode - 1] += reward
                stats.episode_lengths[i_episode - 1] = t

                # Saving the experience in buffer:
                self.buffer.states.append(
                    torch.from_numpy(state).float().to(device))
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    self.update()
                    self.buffer.clear_buffer()
                    timestep = 0

                if done:
                    break
                state = next_state

            # logging
            if self.verbose:
                if i_episode % 10 == 0:
                    print((
                        f"{int(stats.episode_lengths[i_episode - 1])} Steps in"
                        f"Episode {i_episode}/{episodes}. "
                        f"Reward {stats.episode_rewards[i_episode-1]}"))
            else:
                if i_episode % 1000 == 0:
                    print((f"{int(stats.episode_lengths[i_episode - 1])} Steps in"
                        f"Episode {i_episode}/{episodes}. "
                        f"Reward {stats.episode_rewards[i_episode-1]}"))

        return stats

    def update(self):
        # Monte Carlo estimate of the return over all steps
        # (possibly across episodes):
        rewards = np.zeros_like(self.buffer.rewards, dtype=np.float32)
        discounted_reward = 0
        for i, (reward, is_terminal) in enumerate(
                zip(reversed(self.buffer.rewards),
                    reversed(self.buffer.is_terminals))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards[-(i + 1)] = discounted_reward

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states).to(device).detach()
        old_actions = torch.stack(self.buffer.actions).to(device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, dist_entropy = self.actor.evaluate(
                old_states, old_actions)

            # getting the state_values from the critic
            state_values = self.critic(old_states)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1,
                                    surr2) - self.entropy_coeff * dist_entropy
            critic_loss = 0.5 * F.mse_loss(state_values, rewards)

            # take gradient step (actor)
            self.actor_optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            # take gradient step (critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save_models(self, model_name):
        self.modelio.save(model=self.actor,
                          model_name=f'ppo_c_actor_{model_name}.pt')
        self.modelio.save(model=self.critic,
                          model_name=f'ppo_c_critic_{model_name}.pt')

    def load_models(self, model_name):
        # if self._model
        self.modelio.load(model=self.actor,
                          model_name=f'ppo_c_actor_{model_name}.pt')
        self.modelio.load(model=self.actor_old,
                          model_name=f'ppo_c_actor_{model_name}.pt')
        self.modelio.load(model=self.critic,
                          model_name=f'ppo_c_critic_{model_name}.pt')
