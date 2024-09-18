import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Critic(nn.Module):
    def __init__(self, im_size, hidden_dim=256):
        super().__init__()
        self.im_size = im_size
        
        self.embed1_im = nn.Conv2d(4, 64, 3, stride=5, dilation=5)
        self.embed2_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed3_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed_im = nn.Linear(-1, hidden_dim) # need to calculate dimension here
        
        self.embed_joints = nn.Linear(9, hidden_dim) # Joints are also the servos
        self.embed_dist = nn.Linear(1, hidden_dim)
        
        self.embed_action = nn.Linear(9, hidden_dim) # There are 9 servos

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        
        
        


class Policy(nn.Module):
  
