"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal


       
class ReplayBuffer():
    def __init__(self, episode_length=1000, episodes=10000):
        self.images_buf = np.zeros((episodes, episode_length, 240, 320, 4), dtype=np.float32)
        self.joints_buf = np.zeros((episodes, episode_length, 8), dtype=np.float32)
        self.dist_buf = np.zeros((episodes, episode_length, 1), dtype=np.float32)
        self.a_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)
        self.rew_buf = np.zeros((episodes, episode_length, 8, 8), dtype=np.float32)

        self.ptr, self.max_steps = 0, episode_length
        self.eps, self.max_episodes = 0, episodes
    
    def add(self, transition):
        image, joint, dist, a = transition
        rew = -np.abs(dist/10 - 10) 

        image = (image - 125) / 125
        dist = dist / 100
        
        self.images_buf[self.eps, self.prt] = image
        self.joints_buf[self.eps, self.prt] = joint
        self.dist_buf[self.eps, self.prt] = dist
        self.a_buf[self.eps, self.prt] = a
        self.rew_buf[self.eps, self.prt] = rew
        
        self.ptr = (self.ptr + 1) % self.max_steps
        if self.ptr == self.max_steps - 1:
            self.eps += 1


    def sample(self, batch_size=32):
        idxs = np.random.randint(8, 999, size=batch_size)
        eps = np.random.randint(0, self.eps - 1, size=batch_size)
        
        
        batch = AttrDict(image=self.images_buf[eps, idxs-8:idxs, :, :],
                         joints=self.joints_buf[eps, idxs-8:idxs, :],
                         dist=self.dist_buf[eps, idxs-8:idxs, :],
                         next_image=self.images_buf[eps, idxs-8+1:idxs+1, :, :],
                         next_joints=self.joints_buf[eps, idxs-8+1:idxs+1, :],
                         next_dist=self.dist_buf[eps, idxs-8+1:idxs+1, :],
                         a=self.a_buf[eps, idxs-8:idxs, :, :],
                         rew=self.rew_buf[eps, idxs-8,idxs, :])

        return batch

    def load_saved_data(self):
        pass
        # TO DO

    def save_replay_buffer(self):
        pass

        
