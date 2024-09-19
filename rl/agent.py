import torch
from models.architectures import Critic, Policy
import copy
from torch.func import functional_call


class Actor():
    def __init__(self, im_size, action_range):
        self.policy = Policy(im_size, action_range)
        self.max_angle = 125

    def policy(self, params, x):
        sample, density, mu, std = functional_call(self.policy, params, x)
        return sample, density, mu, std

    def robot_action(self, sample):
        # Check with robot
