import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils.helpers import LimitedQueue


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        self.pe = torch.zeros(max_len, 1, dim_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]


class AttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, out_dim=128):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)

        self.queries = nn.Linear(d_model, d_model)
        self.keys = nn.Linear(d_model, d_model)
        self.values = nn.Linear(d_model, d_model)

        self.attention_layer = nn.MultiheadAttention(embed_dim=d_model, nheads=n_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)

        self.feed_forward = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = x + self.pos_enc(x)

        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)

        out, att_weights = self.attention_layer(Q, K, V)

        out = self.layernorm(out + x)
        out = self.feed_forward(out)

        return out, att_weights

class ConvBlock(nn.Module):
    def __init__(
        self, im_width=640, im_height=480, 
        in_ch=4, hidden_ch=32, out_ch=4, 
        filter_size=5, length=4, embed_dim=128, stride=2
        ):
        super().__init__()

        self.embed_im = nn.Conv2d(in_ch, hidden_ch, filter_size)
        self.embed_im_deep = nn.ModuleList([nn.Conv2d(hidden_ch, hidden_ch, filter_size, stride=stride)] * length)
        self.embed_im_out = nn.Conv2d(hidden_ch, out_ch, filter_size)
        self.embed_im_linear = nn.Linear(20, embed_dim)

    def forward(self, image):
        a, b, c, d, e = image.shape
        image = image.reshape(a * b, c, d, e)

        im = self.embed1_im(image)
        for conv_layer in self.embed_im_deep:
            im = F.relu(conv_layer(im))
        im = F.relu(self.embed_im_out(im))
        
        im_flat = im.reshape(im.shape[0], -1)
        im_flat = im_flat.reshape(a, b, -1)

        im_out = self.embed_im_linear(im_flat)
        return im_out


class Critic(nn.Module):
    def __init__(self, action_frames=8, hidden_dim=256, action_hidden_dim=64):
        super().__init__()

        # Image
        self.conv_block = ConvBlock(hidden_ch=64, embed_dim=256) # all other values are default
        
        # Joints and distance
        self.embed_joints = nn.Linear(8, hidden_dim) # Joints are the servos
        self.embed_dist = nn.Linear(1, hidden_dim)
        
        # Actions
        self.embed_actions = nn.Linear(8, hidden_dim)
        self.action_attn_block = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)

        # Rest of the network
        self.main_attn_block1 = AttentionBlock(d_model=hidden_dim, n_heads=16, dim_feedforward=hidden_dim)
        self.main_attn_block2 = AttentionBlock(d_model=hidden_dim, n_heads=16, dim_feedforward=hidden_dim)

        self.deep_linear1 = nn.Linear(20, 128)
        self.deep_linear2 = nn.Linear(128, 32)
        self.out_linear = nn.Linear(32, 1)

        self.action_frames = action_frames

    def forward(self, image, joints, dist, action):
        im = self.conv_block(image)

        joints = self.embed_joints(joints)
        dist = self.embed_dist(dist)

        actions = self.embed_actions(actions)
        for i in range(self.action_frames):
            actions[:, i, :, :] = self.action_attn_block(actions[:, i, :, :])

        x = im + joints + dist + actions

        x = self.main_attn_block1(x)
        x = self.main_attn_block2(x)

        x = F.relu(self.deep_linear1(x))
        x = F.relu(self.deep_linear2(x))
        x = self.out_linear(x)

        return x



class Policy(nn.Module):
    def __init__(self, action_range=4, frames=8, hidden_dim=128):
        super().__init__()

        # Image
        self.conv_block = ConvBlock(hidden_ch=32, embed_dim=hidden_dim)

        # Joints and distance
        self.embed_actions = nn.Linear(8, hidden_dim)
        self.action_attn_block = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)

        # Rest of the network
        self.main_attn_block1 = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)
        self.main_attn_block2 = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)

        # Embed to create every action frame
        self.embed_action_frame = nn.Linear(20, 64) # it is 64 because 8 frames and the action vector is 8 (8 joints)
        self.act_attn_block1 = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)
        self.act_attn_block2 = AttentionBlock(d_model=hidden_dim, n_heads=8, dim_feedforward=hidden_dim)

        self.mu = nn.Linear(64, 64) # 8 frames and each action vector is 8        
        self.log_std = nn.Linear(64, 64) # 8 frames and each action vector is 8

        self.action_range = action_range        

    def forward(self, image, joints, dist):
        im = self.conv_block(image)

        joints = self.embed_joints(joints)
        dist = self.embed_dist(dist)

        x = im + joints + dist
        x = self.embed_action_frame(x)

        # need to reshape x correctly
        x = act_attn_block1(x)
        x = act_attn_block2(x)

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        sample = self.action_range * torch.tanh(sample / self.action_range)

        return sample, density, mu, std


model = Critic()

images = torch.rand(256, 8, 480, 640, 4)
joints = torch.rand(256, 8, 8)
dist = torch.rand(256, 8, 1)
action = torch.rand(256, 8, 8, 8)


val = model(image, joints, dist, action)