import sys
sys.path.insert(0, '../')

import torch
from models.architectures import Critic, Policy, Decoder
import copy
from torch.func import functional_call
import torch.nn.functional as F
import torch.nn as nn
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
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA


INIT_LOG_ALPHA = 0
MAX_ENTROPY = 100

class Actor():
    def __init__(self, device):
        self.policy = Policy(device).to(device)
        self.max_angle = 125
        self.decoder = Decoder().to(device)

    def run_policy(self, params, x):
        sample, density, mu, std = functional_call(self.policy, params['Policy'], x)
        return sample, density, mu, std

    def robot_action(self, sample, previous_sample, params):
        #sample = functional_call(self.decoder, params['Decoder'], (sample, joints))
        delta = sample - previous_sample
        points = np.arange(0, 1, 0.125) + 0.125
        interpolated_point = delta[np.newaxis, :] * points[:, np.newaxis]
        sample = previous_sample[np.newaxis, :] + interpolated_point
                        
        r_action = [12, 0, 0, 1]
        sample = sample.squeeze()        
        sample = 10 * sample # The action range was set to -5 and 5, and the angle range -125 to 125
        offset = np.array([30, 30, 30, 30, 25, 25, 25, 25])
        offset = offset[np.newaxis, :]
        sample = sample + offset
        sample = np.pad(sample, ((0, 4), (0, 0)), mode='edge') # This is to maintain the last joint position before executing new skill
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
        self.log_data_freq = 20 # data is logged every 512 steps

        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True, device=self.device)

        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.learning_rate)
        self.prior = Normal(0, 1)
        

    def training_iteration(self, params, optimizers, transition, iterations, ref_params):
        self.experience_buffer.add(transition)

        log_data = True if iterations % self.log_data_freq == 0 else False

        if self.experience_buffer.eps >= 1:
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False
                policy_loss, critic_loss = self.losses(params, log_data, iterations, ref_params)
                losses = [policy_loss, critic_loss]
                keys = ['Policy', 'Critic']
                params = Adam_update(params, losses, keys, optimizers)
                polyak_update(params['Critic'].values(),
                              params['Target_critic'].values(), 0.005)

        return params


    def losses(self, params, log_data, iterations, ref_params):
        batch = self.experience_buffer.sample(batch_size=256)

        action = torch.from_numpy(batch.action).to(self.device)
        prev_action = torch.from_numpy(batch.prev_action).to(self.device)
        speed = torch.from_numpy(batch.speed).to(self.device)
        next_speed = torch.from_numpy(batch.next_speed).to(self.device)
        reward = torch.from_numpy(batch.reward).to(self.device)

        speed = speed.reshape(-1, 1)
        next_speed = next_speed.reshape(-1, 1)

        with torch.no_grad():
            next_sample, _, _, _ = self.actor.run_policy(params, (action, next_speed))

        target_critic_arg = (next_sample, action, next_speed)
        critic_arg = (action, prev_action, speed)

        with torch.no_grad():
            q_target = self.eval_critic(target_critic_arg, params,
                                        target_critic=True)
        
        q_target = reward + (self.discount * q_target.squeeze())
        q_target = torch.clamp(q_target, min=-100, max=100)

        q = self.eval_critic(critic_arg, params)

        critic_loss = F.mse_loss(q.squeeze(), q_target.squeeze())

        # Policy loss
        sample, pdf, mu, std = self.actor.run_policy(params, (prev_action, speed))

        q_pi_arg = (sample, prev_action, speed)
        q_pi = self.eval_critic(q_pi_arg, params)

        entropy_term = torch.clamp(kl_divergence(pdf, self.prior), max=MAX_ENTROPY).mean()

        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        entropy_loss = alpha_skill * entropy_term

        policy_loss = -q_pi.mean() + entropy_loss.mean()

        if log_data:
            last_eps = self.experience_buffer.eps - 1
            
            last_return = self.experience_buffer.speed_buf[last_eps, :].mean()

            actions = self.experience_buffer.action_buf[last_eps, :, :].squeeze()

            j_pca = PCA(n_components=3)
            traj_joints = j_pca.fit_transform(actions)

                                    
            wandb.log({'Average speed': last_return.mean()}, step=iterations)

            q_output = self.log_scatter_3d(q.squeeze(), q_target.squeeze(), reward.squeeze(), next_speed.squeeze(),
                                           'Q', 'Q target', 'Reward', 'Speed')

            q_improv_pi = self.log_scatter_3d(q_pi.squeeze() - q.squeeze(), q_pi.squeeze(), reward.squeeze(), next_speed.squeeze(),
                                              'Q delta', 'Q pi', 'Reward', 'Speed')
            
            joints_traj = self.log_scatter_3d(traj_joints[:, 0], traj_joints[:, 1], traj_joints[:, 2], np.arange(200),
                                              'Dim 1', 'Dim 2', 'Dim 3', 'Step', torch_tensor=False)
            
            q_dist = self.log_histogram_2d(q.squeeze(), q_target.squeeze(), 'Q vals', 'Q target')

            dist_critic = self.distance_to_params(params, ref_params, 'Critic', 'Critic')
            dist_policy = self.distance_to_params(params, ref_params, 'Policy', 'Policy')

            
            wandb.log(
                {
                    'Sampled_reward': reward.mean().detach().cpu(),
                    'Sampled_reward_dist': wandb.Histogram(reward.detach().cpu()),
                    'Entropy_term': entropy_term.detach().cpu(),

                    'Critic/Q_values': wandb.Histogram(q[torch.abs(q) < 100].detach().cpu()),
                    'Critic/Mean_Q_value': q.mean().detach().cpu(),
                    'Critic/Critic_loss': critic_loss.detach().cpu(),
                    'Critic/Q_values_std': q[torch.abs(q) < 100].std().detach().cpu(),
                    'Critic/Q_3D': q_output,
                    'Critic/Q_distribution': q_dist,
                    'Critic/Distance_to_init': dist_critic,

                    'Policy/Joints trajectory': joints_traj,
                    'Policy/Distance_to_init': dist_policy,
                    'Policy/q_pi': q_pi.mean().detach().cpu(),
                    'Policy/mu_dist': wandb.Histogram(sample.detach().cpu()),
                    'Policy/mu_mean_across_samples': sample.std(0).mean().detach().cpu(),
                    'Policy/std': std.mean().detach().cpu(),
                    'Policy/alpha': alpha_skill.detach().cpu(),
                    'Policy/q_improv_pi': q_improv_pi                    
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
        
        
    def log_scatter_3d(self, x, y, z, color, xlabel, ylabel, zlabel, color_label, torch_tensor=True):
        if torch_tensor:
            x = x.detach().cpu().numpy()[:, None]
            y = y.detach().cpu().numpy()[:, None]
            z = z.detach().cpu().numpy()[:, None]
            color = color.detach().cpu().numpy()[:, None]
        else:
            x = x[:, None]
            y = y[:, None]
            z = z[:, None]
            color = color[:, None]
            

        data = np.concatenate([x, y, z, color], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel, zlabel, color_label])
        
        fig_scatter = px.scatter_3d(df, x=xlabel, y=ylabel,
                                    z=zlabel, color=color_label)
        fig_scatter.update_layout(scene=dict(aspectmode='cube'))

        return fig_scatter
                    
    def log_histogram_2d(self, x, y, xlabel, ylabel):
        x = x.detach().cpu().numpy()[:, None]
        y = y.detach().cpu().numpy()[:, None]

        data = np.concatenate([x, y], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel])

        fig_heatmap = px.density_heatmap(df, x=xlabel, y=ylabel,
                                         marginal_x='histogram',
                                         marginal_y='histogram',
                                         nbinsx=60,
                                         nbinsy=60)

        return fig_heatmap


    def distance_to_params(self, params1, params2, name1, name2):
        with torch.no_grad():
            vec1 = nn.utils.parameters_to_vector(params1[name1].values())
            target_vec1 = nn.utils.parameters_to_vector(params2[name2].values())
        return torch.norm(vec1 - target_vec1)
