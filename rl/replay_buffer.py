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
    def __init__(self, episode_length=100, episodes=10000):
        self.joints_buf = np.zeros((episodes, episode_length, 8), dtype=np.float32)
        self.dist_buf = np.zeros((episodes, episode_length, 1), dtype=np.float32)
        self.a_buf = np.zeros((episodes, episode_length, 4), dtype=np.float32)
        #self.rew_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)

        self.ptr, self.max_steps = 0, episode_length
        self.eps, self.max_episodes = 0, episodes
    
    def add(self, transitions):
        if transitions is not None:
            for transition in transitions:
                joints = transition[0]
                dist = transition[1] # Recall distance is in dm (decimeters)
                dist = np.clip(dist, -4, 4)
                a = transition[2]
                # temporary line to reshape action
                if np.abs(dist) > 2.5:
                    continue
                
                self.joints_buf[self.eps, self.ptr] = joints
                self.dist_buf[self.eps, self.ptr] = dist
                self.a_buf[self.eps, self.ptr] = a
                
                self.ptr = (self.ptr + 1) % self.max_steps
                if self.ptr == self.max_steps - 1:
                    self.eps += 1

    def sample(self, batch_size=128):
        idxs = np.random.randint(1, self.max_steps - 1, size=batch_size)
        idxs = idxs[:, np.newaxis]
        eps = np.random.randint(0, self.eps, size=batch_size)
        eps = eps[:, np.newaxis]

        target_vel = 0.09

        vel = self.dist_buf[eps, idxs+1, :]
        
        reward = np.where((target_vel<=vel) & (vel<= 2 *target_vel), 1, 0)
        neg_vel_reward = np.where((-target_vel < vel) & (vel < target_vel),
                                  1 - np.abs(target_vel-vel)/(2 * target_vel),
                                  0)
        pos_vel_reward = np.where((2 * target_vel < vel) & (vel < 4 * target_vel),
                                  1 - np.abs(2 * target_vel - vel) / (2 *target_vel),
                                  0)

        reward = reward + neg_vel_reward + pos_vel_reward
        reward = np.array(reward, dtype=np.float32)
        
        batch = AttrDict(joints=self.joints_buf[eps, idxs, :].squeeze(),
                         dist=self.dist_buf[eps, idxs, :].squeeze(axis=1),
                         next_joints=self.joints_buf[eps, idxs+1, :].squeeze(),
                         next_dist=self.dist_buf[eps, idxs+1, :].squeeze(axis=1),
                         a=self.a_buf[eps, idxs, :].squeeze(),
                         prev_a=self.a_buf[eps, idxs - 1, :].squeeze(),
                         rew=reward.squeeze())

        return batch

    def load_saved_data(self):
        pass
        # TO DO

    def save_replay_buffer(self):
        pass

        
