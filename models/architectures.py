import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class Critic(nn.Module):
    def __init__(self, im_size, hidden_dim=256):
        super().__init__()
        
        self.embed1_im = nn.Conv2d(4, 64, 3, stride=5, dilation=5)
        self.embed2_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed3_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed_im = nn.Linear(-1, hidden_dim) # need to calculate dimension here
        
        self.embed_joints = nn.Linear(9, hidden_dim) # Joints are the servos
        self.embed_dist = nn.Linear(1, hidden_dim)
        
        self.embed_action = nn.Linear(9, hidden_dim) # There are 9 servos

        self.attention1 = Self

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, image, joints, dist, action):
        im = self.embed1_im(image)
        im = self.embed2_im(im)
        im = self.embed3_im(im)
        pdb.set_trace()

        joints = self.embed_joints(joints)
        dist = self.embed_dist(dist)
        action = self.embed_action(action)

        state = im + joints + dist + action

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        return self.out(x)


class Policy(nn.Module):
    def __init__(self, im_size, action_range, hidden_dim=128):
        super().__init__()
        self.action_range = action_range
        
        self.embed1_im = nn.Conv2d(4, 64, 3, stride=5, dilation=5)
        self.embed2_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed3_im = nn.Conv2d(64, 64, 3, stride=5, dilation=5)
        self.embed_im = nn.Linear(-1, hidden_dim) # need to calculate dimension here

        self.embed_joints = nn.Linear(9, hidden_dim)
        self.embed_dist = nn.Linear(1, hidden_dim)

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, 9)
        self.log_std = nn.Linear(hidden_dim, 9)
        

    def forward(self, image, joints, dist):
        im = self.embed1_im(image)
        im = self.embed2_im(im)
        im = self.embed3_im(im)
        pdb.set_trace()

        joints = self.embed_joints(joints)
        dist = self.embed_dist(dist)
        
        x= im + joints + dist

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        sample = self.action_range * torch.tanh(sample / self.action_range)

        return sample, density, mu, std
