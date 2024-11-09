"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utils.helpers import hyper_params, AttrDict
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal
import pdb

       
class ReplayBuffer():
    def __init__(self, episode_length=200, episodes=10000):
        self.action_buf = np.zeros((episodes, episode_length, 8), dtype=np.float32)
        self.speed_buf = np.zeros((episodes, episode_length, 1), dtype=np.float32)

        self.ptr, self.max_steps = 0, episode_length
        self.eps, self.max_episodes = 0, episodes
    
    def add(self, transitions):
        if transitions is not None:
            for transition in transitions:
                action = transition[0]
                speed = np.clip(transition[1], -4, 4) # Recall distance is in dm (decimeters)
                
                self.action_buf[self.eps, self.ptr] = action
                self.speed_buf[self.eps, self.ptr] = speed
                
                self.ptr = (self.ptr + 1) % self.max_steps
                if self.ptr == self.max_steps - 1:
                    self.eps += 1

    def sample(self, batch_size=128):
        idxs = np.random.randint(1, self.max_steps - 1, size=batch_size)
        idxs = idxs[:, np.newaxis]
        eps = np.random.randint(0, self.eps, size=batch_size)
        eps = eps[:, np.newaxis]

        target_vel = 0.10

        vel = self.speed_buf[eps, idxs+1, :]
        
        reward = np.where((target_vel<=vel) & (vel<= 2 *target_vel), 1, 0)
        neg_vel_reward = np.where((-target_vel < vel) & (vel < target_vel),
                                  1 - np.abs(target_vel-vel)/(2 * target_vel),
                                  0)
        pos_vel_reward = np.where((2 * target_vel < vel) & (vel < 4 * target_vel),
                                  1 - np.abs(2 * target_vel - vel) / (2 *target_vel),
                                  0)

        reward = reward + neg_vel_reward + pos_vel_reward
        reward = np.array(reward, dtype=np.float32)

        fall_reward = np.where(np.abs(vel) > 1.5, -.5, 0)

        reward = reward + fall_reward
        
        batch = AttrDict(action=self.action_buf[eps, idxs, :].squeeze(),
                         prev_action=self.action_buf[eps, idxs - 1, :].squeeze(),
                         speed=self.speed_buf[eps, idxs, :].squeeze(),
                         next_speed=self.speed_buf[eps, idxs + 1, :].squeeze(),
                         reward=reward.squeeze()
                         )

        return batch

    def load_saved_data(self):
        pass
        # TO DO

    def save_replay_buffer(self):
        pass

        
