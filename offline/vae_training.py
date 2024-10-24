import sys
sys.path.insert(0, '../')


import torch
from torch.func import functional_call
from torch.distributions import Normal
from utils import Adam
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from models.architectures import Encoder, Decoder
import numpy as np
from data import *


class OfflineTraining():
    def __init__(self):        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.skill_length = 8
        self.N = Normal(0, 1)

    def create_dataset(self):
        data = []

        for skill in skills:
            new_skill = self.process_skill(skill)
            data.append(skill)

        self.data = np.concatenate(data, axis=0)
        self.indexes = np.arange(self.data.shape[0])

    def dataset_loader(self):
        self.create_dataset()

        dset = Drivedata(self.indexes)
        self.loader = DataLoader(dset, shuffle=True, num_works=8,
                                 batch_size=128)
        
                
    def process_skill(self, skill):
        add = skill % self.skill_length
        skill.extend(skill[0:add])        
        skill = np.array(skill).reshape(-1, 8)

        offset = np.array([40, 40, 40, 40, 20, 20, 20, 20])
        offset = offset[np.newaxis, :]

        skill = skill - offset
        skill /= 15

        indices = np.arange(skill.shape[0] - 7)[:, np.newaxis] + np.arange(8)

        skill = skill[indices]

        return skill

    def vae_loss(self, skill, params):
        z, pdf, mu, std = functional_call(self.encoder, params['Encoder'], skill)
        rec = functional_call(self.decoder, params['Decoder'], z)

        rec_loss = -Normal(rec, 1).log_prob(skill).sum(axis=-1).mean(1)

        kl_loss = kl_div
        

    

class Drivedata(Dataset):
    """Dataset loader."""

    def __init__(self, indexes, transform=None):
        """Dataset init."""
        self.xs = indexes

    def __getitem__(self, index):
        """Get given item."""
        return self.xs[index]

    def __len__(self):
        """Compute length."""
        return len(self.xs)
