import torch
from models.architectures import Critic, Policy
import copy
from torch.func import functional_call
import numpy as np
from stable_baselines3.common.utils import polyak_update


class Actor():
    def __init__(self, im_size, action_range):
        self.policy = Policy(im_size, action_range)
        self.max_angle = 125

    def run_policy(self, params, x):
        sample, density, mu, std = functional_call(self.policy, params, x)
        return sample, density, mu, std

    def robot_action(self, sample):
        r_action = [8, 0, 0, 1] # 8 frames; 0 pitch; 0 roll; angle multiplier is 1

        sample = sample.cpu().detach().numpy()
        sample = 25 * sample # The action range was set to -5 and 5, and the angle range -125 to 125
        sample = list(int(sample))
        r_action.extend(sample)
        
        return r_action

class BittleRL():
    
    def __init__(self, env, experience_buffer, actor, critic):             
        super().__init__()

        self.env = env
        self.experience_buffer = experience_buffer
        self.actor = actor
        self.critic = critic
        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = []
        self.log_data = 0
        POINTS = 128
        self.log_data_freq = 512 # data is logged every 512 steps

    def training():


    def training_iteration():

    def losses():

    def eval_critic():
        
        
        
