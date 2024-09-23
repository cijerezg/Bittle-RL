"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict, compute_cum_rewards
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal

       
class ReplayBuffer(hyper_params):
    def __init__(self, size, env, lat_dim, reset_ratio, args):
        super().__init__(args)

        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.cum_reward = np.zeros((size, 1), dtype=np.float32)
        self.norm_cum_reward = np.zeros((size, 1), dtype=np.float32)        
        self.ptr, self.size, self.max_size = 0, 0, size
    
    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tracker[self.ptr] = True
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs],
                         cum_reward=self.cum_reward[idxs],
                         norm_cum_reward=self.norm_cum_reward[idxs])
        return batch
