import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils.helpers import LimitedQueue
import pdb
import math
from utils.helpers import get_params
import time

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_actions = nn.Conv1d(8, 64, 3)

        self.layer1 = nn.Conv1d(64, 16, 3)
        self.layer2 = nn.Conv1d(16, 8, 3)

        self.mu_hidden = nn.Linear(16, 16)
        self.log_std_hidden = nn.Linear(16, 16)
        
        self.mu = nn.Linear(16, 8)
        self.log_std = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.embed_actions(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        x = x.reshape(-1, 16)

        mu = F.relu(self.mu_hidden(x))
        mu = self.mu(mu)

        log_std = F.relu(self.log_std_hidden(x))
        log_std = self.log_std(log_std)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))

        density = Normal(mu, std)
        sample = density.rsample()

        return sample, density, mu, std


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(8, 32)

        self.deconv1 = nn.ConvTranspose1d(16, 32, 3)
        self.deconv2 = nn.ConvTranspose1d(32, 32, 3)
        self.deconv3 = nn.ConvTranspose1d(32, 8, 3, groups=8)

    def forward(self, x, obs):
        x = F.relu(self.layer1(x))
        x = x.reshape(-1, 16, 2)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)

        x[:, 0, :] = obs * .75 + x[:, 0, :] * .25

        for i in range(1, 8):
            x[:, i, :] = x[:, i-1, :] * .75 + x[:, i, :] * .25

        x = 4 * torch.tanh(x / 4)
            
        return x


class Critic(nn.Module):
    def __init__(self, device, hidden_dim=256):
        super().__init__()
        
        # Joints and distance
        self.embed_action = nn.Linear(8, hidden_dim) # Joints are the servos
        self.embed_prev_action = nn.Linear(8, hidden_dim)
        self.embed_speed = nn.Linear(1, hidden_dim)

        
        # Actions
        self.deep_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.preout_linear = nn.Linear(hidden_dim, 64)
        self.out_linear = nn.Linear(64, 1)

        # Normalization layers
        self.embed_action_n = nn.LayerNorm(hidden_dim)
        self.embed_prev_action_n = nn.LayerNorm(hidden_dim)
        self.embed_speed_n = nn.LayerNorm(hidden_dim)
       
        self.deep_layer1_n = nn.LayerNorm(hidden_dim)
        self.deep_layer2_n = nn.LayerNorm(hidden_dim)
        self.deep_layer3_n = nn.LayerNorm(hidden_dim)
        self.deep_layer4_n = nn.LayerNorm(hidden_dim)

        self.preout_linear_n = nn.LayerNorm(64)
        
        

    def forward(self, action, prev_action, speed):
        embedded_action = self.embed_action_n(F.relu(self.embed_action(action)))
        embedded_prev_action = self.embed_prev_action_n(F.relu(self.embed_prev_action(prev_action)))
        embedded_speed = self.embed_speed_n(F.relu(self.embed_speed(speed)))
        
        x = embedded_action + embedded_prev_action + embedded_speed

        x = self.deep_layer1_n(F.relu(self.deep_layer1(x)))
        x = self.deep_layer2_n(F.relu(self.deep_layer2(x)))
        x = self.deep_layer3_n(F.relu(self.deep_layer3(x)))
        x = self.deep_layer4_n(F.relu(self.deep_layer4(x)))

        x = self.preout_linear_n(F.relu(self.preout_linear(x)))
        x = self.out_linear(x)

        return x


class Policy(nn.Module):
    def __init__(self, device, action_range=5, hidden_dim=24):
        super().__init__()

        # Joints and distance
        self.embed_prev_action = nn.Linear(8, hidden_dim)
        self.embed_speed = nn.Linear(1, hidden_dim)

        self.deep_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.deep_mu = nn.Linear(hidden_dim, 16)
        self.mu = nn.Linear(hidden_dim, 8)
        self.log_std = nn.Linear(hidden_dim, 8)

        self.action_range = action_range

    def forward(self, prev_action, speed):
        embedded_prev_action = F.relu(self.embed_prev_action(prev_action))
        speed = F.relu(self.embed_speed(speed))
        
        x = embedded_prev_action + speed

        x = F.relu(self.deep_layer1(x))
        x = F.relu(self.deep_layer2(x))

        mu = F.relu(self.deep_mu(x))
        mu = self.mu(x)

        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        sample = self.action_range * torch.tanh(sample / self.action_range)

        return sample, density, mu, std
