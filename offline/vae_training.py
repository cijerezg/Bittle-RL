import sys
sys.path.insert(0, '../')


import torch
from torch.func import functional_call
from torch.distributions import Normal
from utils.optimization import Adam_update, set_optimizers
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from models.architectures import Encoder, Decoder
from utils.helpers import get_params
import numpy as np
from skills_library import *
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d


class OfflineTraining():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.skill_length = 8
        self.N = Normal(0, 1)
        self.dataset_loader()



    def create_dataset(self):
        data = []
        init_states = []

        for skill in skills:
            new_skill, init_state = self.process_skill(skill)
            data.append(new_skill)
            init_states.append(init_state)

        self.data = np.concatenate(data, axis=0)
        self.init_states = np.concatenate(init_states, axis=0)
        self.indexes = np.arange(self.data.shape[0])

    def dataset_loader(self):
        self.create_dataset()

        dset = Drivedata(self.indexes)
        self.loader = DataLoader(dset, shuffle=True, num_workers=8,
                                 batch_size=128)
        
                
    def process_skill(self, skill):
        skill = np.array(skill, dtype=np.float32).reshape(-1, 8)


        offset = np.array([40, 40, 40, 40, 20, 20, 20, 20], dtype=np.float32)
        offset = offset[np.newaxis, :]

        skill = skill - offset
        skill = skill / 8

        seed1 = np.random.RandomState(12345)
        seed2 = np.random.RandomState(12345654)
        prev_random_inits = 7 * (seed1.rand(4 * skill.shape[0], 8) - .5)
        random_inits = 7 * (seed2.rand(4 * skill.shape[0], 8) - .5)
        random_inits = prev_random_inits * .8 + random_inits * .2
        
        closest_idxs = np.argmin(np.linalg.norm(random_inits[:, np.newaxis] - skill, axis=2), axis=1)
        closest_idxs = closest_idxs[:, np.newaxis] + np.arange(3)
        closest_idxs = closest_idxs % skill.shape[0]
        
        skills_vals = skill[closest_idxs, :]

        x = np.array([0, 1, 6, 7, 8])
        
        skills_vals = np.concatenate((prev_random_inits[:, np.newaxis, :],
                                      random_inits[:, np.newaxis, :],
                                      skills_vals), axis=1)
        
        f = interp1d(x, skills_vals, axis=1, kind='linear')

        x = np.arange(9)        
        new_skills = f(x)

        skill = np.tile(skill, (8, 1))
        cutoff = skill.shape[0] // self.skill_length
        skill = skill[:8*cutoff, :]

        
        indices = np.arange(skill.shape[0] - 7)[:, np.newaxis] + np.arange(8)
        init_state = skill[indices[:-1, 0]]
        init_state = np.concatenate((init_state[0,:][None, :], init_state), axis=0)

        skill = skill[indices]

        skill = np.concatenate((skill, new_skills[:, 1:, :]), axis=0)
        init_state = np.concatenate((init_state, new_skills[:, 0, :]), axis=0)

        skill = np.array(skill, dtype=np.float32)
        init_state = np.array(init_state, dtype=np.float32)
        

        return skill, init_state

    def vae_loss(self, skill, inits, params, plot):
        z, pdf, mu, std = functional_call(self.encoder, params['Encoder'], skill)
        rec = functional_call(self.decoder, params['Decoder'], (z, inits))

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            sns.heatmap(skill[0,:].detach().cpu().numpy(), ax=axes[0], cmap="viridis", cbar=True, annot=True)
            axes[0].set_title('Skill')

            sns.heatmap(rec[0,:].detach().cpu().numpy(), ax=axes[1], cmap="viridis", cbar=True, annot=True)
            axes[1].set_title('Reconstruction')
            print(z[0, :])

            plt.tight_layout()
            plt.show()
        
        rec_loss = -Normal(rec, 1).log_prob(skill).sum(axis=-1).mean()
        kl_loss = kl_divergence(pdf, self.N).mean()

        return rec_loss, kl_loss
        

    def train(self, params, optimizers, beta, i):

        plot = False
        if i % 100 == 0:
            plot = True

        for idx in self.loader:
            batch = torch.from_numpy(self.data[idx]).to(self.device)
            inits = torch.from_numpy(self.init_states[idx]).to(self.device)
            rec_loss, kl_loss = self.vae_loss(batch, inits, params, plot)
            plot = False
            loss = rec_loss + beta * kl_loss
            losses = [loss]
            keys = ['VAE']
            params = Adam_update(params, losses, keys, optimizers)

        print(f'Reconstuction loss is {rec_loss}, and kl loss is {kl_loss}')


        return params
        

    
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



training = OfflineTraining()

models = [training.encoder, training.decoder]
names = ['Encoder', 'Decoder']
pretrained_models = [None, None]


params = get_params(models, names, pretrained_models)

keys_optimizers = ['VAE']
optimizers = set_optimizers(params, keys_optimizers, 3e-4)

for i in range(800):
    params = training.train(params, optimizers, 0.05, i)


torch.save(params['Decoder'], '../offline_models/decoder.pt')
