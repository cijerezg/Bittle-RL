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




class Critic(nn.Module):
    def __init__(self, device, hidden_dim=256):
        super().__init__()
        
        # Joints and distance
        self.embed_joints = nn.Linear(8, hidden_dim) # Joints are the servos
        self.embed_dist = nn.Linear(1, hidden_dim)
        
        # Actions
        self.embed_actions_conv1 = nn.Conv1d(8, 64, 3)
        self.embed_actions_conv2 = nn.Conv1d(64, 64, 3)
        
        self.embed_actions = nn.Linear(256, hidden_dim)

        self.deep_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.preout_linear = nn.Linear(hidden_dim, 64)
        self.out_linear = nn.Linear(64, 1)

    def forward(self, joints, dist, actions):
        joints = F.relu(self.embed_joints(joints))
        dist = F.relu(self.embed_dist(dist))

        actions = F.relu(self.embed_actions_conv1(actions))
        actions = F.relu(self.embed_actions_conv2(actions))
        actions = actions.reshape(actions.shape[0], -1)        
        actions = F.relu(self.embed_actions(actions))
        
        x = joints + dist + actions

        x = F.relu(self.deep_layer1(x))
        x = F.relu(self.deep_layer2(x))
        x = F.relu(self.deep_layer3(x))
        x = F.relu(self.deep_layer4(x))

        x = F.relu(self.preout_linear(x))
        x = self.out_linear(x)

        return x



class Policy(nn.Module):
    def __init__(self, device, action_range=5, hidden_dim=128):
        super().__init__()

        # Joints and distance
        self.embed_joints = nn.Linear(8, hidden_dim)
        self.embed_dist = nn.Linear(1, hidden_dim)


        self.deep_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.deep_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.out_mu = nn.ConvTranspose1d(64, 8, 3) # 16 channels because 4 * 64 = 256
        self.mu = nn.ConvTranspose1d(8, 8, 3, groups=8)
        self.log_std = nn.Linear(hidden_dim, 64) # 8 frames and each action vector is 8        

        self.action_range = action_range


    def forward(self, joints, dist):
        joints = F.relu(self.embed_joints(joints))
        dist = F.relu(self.embed_dist(dist))

        x = joints + dist

        x = F.relu(self.deep_layer1(x))
        x = F.relu(self.deep_layer2(x))

        mu = x.reshape(-1, 64, 4)
        mu = F.relu(self.out_mu(mu))
        mu = self.mu(mu)

        log_std = self.log_std(x).reshape(-1, 8, 8)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        sample = self.action_range * torch.tanh(sample / self.action_range)

        return sample, density, mu, std


# device = torch.device('cpu')
    
# # model = Critic(device)

# images = torch.rand(1, 8, 240, 320, 3).to(device) # original sizes were 480 ad 640, but that seems too big.
# joints = torch.rand(1, 8, 8).to(device)
# dist = torch.rand(1, 8, 1).to(device)
# # action = torch.rand(1, 8, 8).to(device)



# model = Policy(device)
# model = model.to(device)


# # model = model.to(device)
# # name = ['Policy']


# # params = get_params([model], name, [None])

# # pdb.set_trace()



# # now = time.time()
# # val = model(images, joints, dist, action)



# # now = time.time()
# out = model(images, joints, dist)



# # print(time.time()-now)
