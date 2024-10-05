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
    def __init__(self, episode_length=100, episodes=50):
        self.images_buf = np.zeros((episodes, episode_length, 240, 320, 4), dtype=np.float32)
        self.joints_buf = np.zeros((episodes, episode_length, 8), dtype=np.float32)
        self.dist_buf = np.zeros((episodes, episode_length, 1), dtype=np.float32)
        self.a_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)
        #self.rew_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)

        self.ptr, self.max_steps = 0, episode_length
        self.eps, self.max_episodes = 0, episodes
    
    def add(self, transitions):
        for transition in transitions:
            image = transition['arr_0']
            image = (image - 125) / 125
            dist = transition['arr_1'] / 100
            joints = transition['arr_2']
            a = transition['arr_3']
            
            self.images_buf[self.eps, self.ptr] = np.array(image, dtype=np.float32)
            self.joints_buf[self.eps, self.ptr] = joints
            self.dist_buf[self.eps, self.ptr] = dist
            self.a_buf[self.eps, self.ptr] = a
        
            self.ptr = (self.ptr + 1) % self.max_steps
            if self.ptr == self.max_steps - 1:
                self.eps += 1
        

    def sample(self, batch_size=32):
        end_idxs = np.random.randint(8, self.max_steps - 1, size=batch_size)
        eps = np.random.randint(0, self.eps, size=batch_size)
        start_idxs = end_idxs - 8

        slice_idxs = np.array([np.arange(start, end) for start, end in zip(start_idxs, end_idxs)])
        eps = eps[:, np.newaxis]

        reward = -np.abs(self.dist_buf[eps, end_idxs[:, np.newaxis] + 1, :] / 10 -10)

        batch = AttrDict(image=self.images_buf[eps, slice_idxs, :],
                         joints=self.joints_buf[eps, slice_idxs, :],
                         dist=self.dist_buf[eps, slice_idxs, :],
                         next_image=self.images_buf[eps, slice_idxs+1, :, :],
                         next_joints=self.joints_buf[eps, slice_idxs+1, :],
                         next_dist=self.dist_buf[eps, slice_idxs+1, :],
                         a=self.a_buf[eps, end_idxs[:, np.newaxis], :].squeeze(),
                         rew=reward.squeeze())

        return batch

    def load_saved_data(self):
        pass
        # TO DO

    def save_replay_buffer(self):
        pass

        
