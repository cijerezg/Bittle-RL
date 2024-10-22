import sys
sys.path.insert(0, '../')

import torch
from models.architectures import Critic, Policy
import copy
from torch.func import functional_call
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.utils import polyak_update
import wandb
from utils.helpers import hyper_params
from torch.optim import Adam
from utils.optimization import Adam_update
import pdb
import time
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from data.skill_library import *
import matplotlib.pyplot as plt


INIT_LOG_ALPHA = 0
MAX_ENTROPY = 100

class Actor():
    def __init__(self, device):
        self.policy = Policy(device).to(device)
        self.max_angle = 125

    def run_policy(self, params, x):
        sample, density, mu, std, smooth_sample = functional_call(self.policy, params['Policy'], x)
        return sample, density, mu, std, smooth_sample

    def robot_action(self, sample):
        r_action = [48, 0, 0, 1]
        sample = sample.cpu().detach().numpy()
        sample = sample.squeeze()        
        sample = 12 * sample # The action range was set to -5 and 5, and the angle range -125 to 125
        offset = np.array([40, 40, 40, 40, 20, 20, 20, 20])
        offset = offset[np.newaxis, :]
        sample = sample + offset
        sample = np.pad(sample, ((0, 40), (0, 0)), mode='edge') # This is to maintain the last joint position before executing new skill
        sample = sample.flatten().astype(np.int32).tolist()
        r_action.extend(sample)
        
        return r_action

class BittleRL(hyper_params):
    def __init__(self, experience_buffer, actor, critic, args):             
        super().__init__(args)

        # Need to define additional params
        # Self max iterations
        self.experience_buffer = experience_buffer
        self.actor = actor
        self.critic = critic
        self.log_data_freq = 200 # data is logged every 512 steps

        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True, device=self.device)

        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.learning_rate)
        self.prior = Normal(0, 1)
        

    def training_iteration(self, params, optimizers, transition, iterations):
        self.experience_buffer.add(transition)

        log_data = True if iterations % self.log_data_freq == 0 else False

        if self.experience_buffer.eps >= 1:
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False
                policy_loss, critic_loss = self.losses(params, log_data, iterations)
                losses = [policy_loss, critic_loss]
                keys = ['Policy', 'Critic']
                params = Adam_update(params, losses, keys, optimizers)
                polyak_update(params['Critic'].values(),
                              params['Target_critic'].values(), 0.005)

        return params


    def losses(self, params, log_data, iterations):
        batch = self.experience_buffer.sample(batch_size=256)

        dist = torch.from_numpy(batch.dist).to(self.device)
        joints = torch.from_numpy(batch.joints).to(self.device)
        next_dist = torch.from_numpy(batch.next_dist).to(self.device)
        next_joints = torch.from_numpy(batch.next_joints).to(self.device)
        a = torch.from_numpy(batch.a).to(self.device)
        rew = torch.from_numpy(batch.rew).to(self.device)

        with torch.no_grad():
            next_sample, _, _, _, _ = self.actor.run_policy(params, (next_joints, next_dist))

        target_critic_arg = (next_joints, next_dist, next_sample)
        critic_arg = (joints, dist, a)

        with torch.no_grad():
            q_target = self.eval_critic(target_critic_arg, params,
                                        target_critic=True)
        
        q_target = rew + (self.discount * q_target.squeeze())
        q_target = torch.clamp(q_target, min=-100, max=100)

        q = self.eval_critic(critic_arg, params)

        critic_loss = F.mse_loss(q.squeeze(), q_target.squeeze())

        # Policy loss
        sample, pdf, mu, std, _ = self.actor.run_policy(params, (joints, dist))

        q_pi_arg = (joints, dist, sample)
        q_pi = self.eval_critic(q_pi_arg, params)

        entropy_term = torch.clamp(kl_divergence(pdf, self.prior), max=MAX_ENTROPY).mean()

        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        entropy_loss = alpha_skill * entropy_term

        policy_loss = -q_pi.mean() + entropy_loss.mean()

        if log_data:
            current_eps = self.experience_buffer.eps
            last_return = -np.abs(self.experience_buffer.dist_buf[current_eps - 1, :] - 3) / 5 + .5
            wandb.log({'Last_return': last_return.mean()}, step=iterations)
            
            
            wandb.log(
                {
                    'Sampled_reward': rew.mean().detach().cpu(),
                    'Sampled_reward_dist': wandb.Histogram(rew.detach().cpu()),
                    'Entropy_term': entropy_term.detach().cpu(),

                    'Critic/Q_values': wandb.Histogram(q[torch.abs(q) < 50].detach().cpu()),
                    'Critic/Mean_Q_value': q.mean().detach().cpu(),
                    'Critic/Critic_loss': critic_loss.detach().cpu(),
                    'Critic/Q_values_std': q[torch.abs(q) < 50].std().detach().cpu(),

                    'Policy/q_pi': q_pi.mean().detach().cpu(),
                    'Policy/mu_over_time': wandb.Histogram(sample[:,:,0].detach().cpu()),
                    'Policy/mu_over_joints': wandb.Histogram(sample[:,0,:].detach().cpu()),
                    'Policy/std': std.mean().detach().cpu(),
                    'Policy/alpha': alpha_skill.detach().cpu(),                    
                }
            )

            svd = self.compute_svd(params)
            for log_name, log_val in svd.items():
                wandb.log({log_name: wandb.Histogram(log_val['S'])})

            
        self.update_log_alpha(entropy_term)

        return policy_loss, critic_loss


    def eval_critic(self, arg, params, target_critic=False):
        name = 'Target_critic' if target_critic else 'Critic'

        return functional_call(self.critic, params[name], arg)

    def update_log_alpha(self, entropy_term):
        loss = torch.exp(self.log_alpha_skill) * \
            (self.delta_entropy - entropy_term).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss.backward()
        self.optimizer_alpha_skill.step()


    def compute_svd(self, params):
        models = ['Critic', 'Policy']

        svd = {}
        
        with torch.no_grad():
            for name in models:
                for key, param in params[name].items():
                    if len(param.shape) < 2:
                        continue
                    U, S, Vh = torch.linalg.svd(param)
                    svd_dict = {'U': U.cpu(), 'S': S.cpu(), 'Vh': Vh.cpu()}
                    svd[f'{name}/{key}-svd'] = svd_dict

        return svd
        
        

        
        
        
