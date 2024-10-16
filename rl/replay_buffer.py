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
        self.a_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)
        #self.rew_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)

        self.ptr, self.max_steps = 0, episode_length
        self.eps, self.max_episodes = 0, episodes
    
    def add(self, transitions):
        if transitions is not None:
            for transition in transitions:
                joints = transition['arr_0']
                dist = transition['arr_1'] # Recall distance is in dm (decimeters)
                a = transition['arr_2']
                # temporary line to reshape action
                a = a.reshape(8, 8)
                
                self.joints_buf[self.eps, self.ptr] = joints
                self.dist_buf[self.eps, self.ptr] = dist
                self.a_buf[self.eps, self.ptr] = a
            
                self.ptr = (self.ptr + 1) % self.max_steps
                if self.ptr == self.max_steps - 1:
                    self.eps += 1
        

    def sample(self, batch_size=128):
        idxs = np.random.randint(0, self.max_steps - 1, size=batch_size)
        idxs = idxs[:, np.newaxis]
        eps = np.random.randint(0, self.eps, size=batch_size)
        eps = eps[:, np.newaxis]
        reward = -np.abs(self.dist_buf[eps, idxs + 1, :] - 2)

        batch = AttrDict(joints=self.joints_buf[eps, idxs, :].squeeze(),
                         dist=self.dist_buf[eps, idxs, :].squeeze(axis=1),
                         next_joints=self.joints_buf[eps, idxs+1, :].squeeze(),
                         next_dist=self.dist_buf[eps, idxs+1, :].squeeze(axis=1),
                         a=self.a_buf[eps, idxs, :].squeeze(),
                         rew=reward.squeeze())

        return batch

    def load_saved_data(self):
        pass
        # TO DO

    def save_replay_buffer(self):
        pass

        
