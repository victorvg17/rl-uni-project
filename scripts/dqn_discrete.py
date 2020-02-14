from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
from utils import ModelIO, PriorityQueue, Transition

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def tt(ndarray, grad=False):
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=grad)


def soft_update(target, source, tau):
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.copy_((1.0 - tau) * p_t.data + tau * p_s.data)


def hard_update(target, source):
    soft_update(target, source, 1.0)


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


class ReplayBuffer:
    def __init__(self, max_size):
        self._buffer = PriorityQueue(max_size=max_size)

    def add_transition(self, state, action, next_state, reward, done,
                       priority):
        trans = Transition(state, action, next_state, reward, done)
        self._buffer.push(item=trans, priority=priority)

    def next_batch(self, batch_size):
        # batch is a list of [priority, count, transition]
        batch = self._buffer.peeknsmallest(batch_size)
        states = [b[2].state for b in batch]
        actions = [b[2].action for b in batch]
        next_states = [b[2].next_state for b in batch]
        rewards = [b[2].reward for b in batch]
        dones = [b[2].done for b in batch]
        return (states, actions, next_states, rewards, dones)


class DQN:
    def __init__(self, state_dim, action_dim, gamma, d2c=None):
        self._q = Q(state_dim, action_dim)
        self._q_target = Q(state_dim, action_dim)

        # self._q.cuda()
        # self._q_target.cuda()

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.0001)
        self._action_dim = action_dim
        self._replay_buffer = ReplayBuffer(5000)
        self._d2c = d2c
        # --- ModelIO ---
        self._modelio = ModelIO(model_path=Path(__file__).resolve().parent /
                                'models')
        self._q_model_name = 'q.pt'
        self._target_model_name = 'target.pt'

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            u = np.random.randint(self._action_dim)
        if self._d2c:
            u = self._d2c(u)
        return u

    def train(self, env, episodes, time_steps, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes))

        for i_episode in range(1, episodes + 1):
            state = env.reset()
            for t in range(time_steps):
                action = self.get_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)

                stats.episode_rewards[i_episode - 1] += reward
                stats.episode_lengths[i_episode - 1] = t
                # calculate priority of the transition (td-error)
                q_s_a = self._q(tt(state)).cpu().detach().numpy()[int(
                    np.squeeze(action))]
                target = reward + self._gamma * np.max(
                    self._q_target(tt(next_state)).cpu().detach().numpy())
                priority = -abs(target - q_s_a) + np.random.randn() * 1e-2
                # add the experience into the buffer
                self._replay_buffer.add_transition(state=state,
                                                   action=action,
                                                   next_state=next_state,
                                                   reward=reward,
                                                   done=done,
                                                   priority=priority)
                # sample a batch of experiences
                samples = self._replay_buffer.next_batch(batch_size=64)

                # update q network parameters
                self._train_batch(samples)
                # update target network periodically/slowly
                soft_update(target=self._q_target, source=self._q, tau=0.01)

                if done:
                    break
                state = next_state
            print(
                f"{int(stats.episode_lengths[i_episode-1])} Steps in Episode {i_episode}/{episodes}. Reward {stats.episode_rewards[i_episode-1]}"
            )

        return stats

    def _train_batch(self, batch):

        states, actions, next_states, rewards, dones = batch
        batch_size = len(rewards)
        # calculating q(s,a)
        batch_actions = np.array(actions).squeeze()
        batch_qs = self._q(tt(np.array(states)))
        batch_qs = batch_qs[np.arange(batch_size), batch_actions]
        # calculating r + gamma * max_a' q(s', a')
        targets = tt(np.array(rewards))
        non_terminal_idx = np.array(dones)
        batch_next_qs = self._q_target(tt(np.array(next_states)))
        batch_max_next_qs, batch_argmax_next_qs = batch_next_qs.max(1)
        targets[non_terminal_idx] = targets[
            non_terminal_idx] + self._gamma * batch_max_next_qs[
                non_terminal_idx]

        self._q_optimizer.zero_grad()
        loss = self._loss_function(batch_qs, targets)
        loss.backward()
        self._q_optimizer.step()

    def save_models(self):
        self._modelio.save(model=self._q, model_name=self._q_model_name)
        self._modelio.save(model=self._q_target,
                           model_name=self._target_model_name)

    def load_models(self):
        # if self._model
        self._modelio.load(model=self._q, model_name=self._q_model_name)
        self._modelio.load(model=self._q_target,
                           model_name=self._target_model_name)
