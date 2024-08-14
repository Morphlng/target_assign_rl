import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, initial_state, joint_action, reward, final_state, done):
        self.buffer.append((initial_state, joint_action, reward, final_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class IQLAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        _limit=3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self._limit = _limit
        self._current_allocation = np.zeros(20)

    def save_checkpoint(self, episode, path="checkpoints"):
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }

        checkpoint_path = os.path.join(path, f"checkpoint_episode_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]

        print(f"Checkpoint loaded from episode {checkpoint['episode']}")
        return checkpoint["episode"]

    def predict(self, state, action_mask):
        threat_levels, pre_allocation, current_allocation = state.reshape([3, -1])
        if np.sum(self._current_allocation) == np.sum(pre_allocation):
            self._current_allocation = np.zeros(len(threat_levels))

        if self._limit is not None:
            redundant_mask = self._current_allocation >= self._limit
            action_mask = action_mask & ~redundant_mask

        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state)).numpy()
            q_values[~action_mask] = -np.inf
            action = np.argmax(q_values)
            self._current_allocation[action] += 1
            return action

    def select_action(self, state, action_mask):
        if random.random() > self.epsilon:
            return self.predict(state, action_mask)
        else:
            valid_actions = np.where(action_mask)[0]
            return np.random.choice(valid_actions)

    def update(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class RuleAgent:
    def __init__(self, num_threats=20):
        self.max_threats = num_threats
        self.current_allocation = np.zeros(num_threats)
        self.pre_allocation = None
        self.index = 0

    def predict(self, state, action_mask=None):
        threat_levels, pre_allocation, current_allocation = state.reshape([3, -1])
        if not np.array_equal(self.pre_allocation, pre_allocation) or np.array_equal(
            self.current_allocation, pre_allocation
        ):
            self.reset(pre_allocation)

        while self.index < self.max_threats:
            if self.current_allocation[self.index] < self.pre_allocation[self.index]:
                self.current_allocation[self.index] += 1
                return self.index
            self.index += 1

    def reset(self, allocation):
        self.pre_allocation = allocation.copy()
        self.current_allocation = np.zeros(self.max_threats)
        self.index = 0


class RandomAgent:
    def __init__(self, num_threats=20, seed=None):
        self.num_threats = num_threats
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def predict(self, state, action_mask):
        return np.random.choice(np.where(action_mask)[0])
