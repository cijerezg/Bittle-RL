import torch
from models.architectures import Critic, Policy
import copy
from torch.func import functional_call
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.utils import polyak_update
import wandb


class Actor():
    def __init__(self, im_size, action_range):
        self.policy = Policy(im_size, action_range)
        self.max_angle = 125

    def run_policy(self, params, x):
        sample, density, mu, std = functional_call(self.policy, params, x)
        return sample, density, mu, std

    def robot_action(self, sample):
        r_action = [8, 0, 0, 1] # 8 frames; 0 pitch; 0 roll; angle multiplier is 1

        sample = sample.cpu().detach().numpy()
        sample = 25 * sample # The action range was set to -5 and 5, and the angle range -125 to 125
        sample = list(int(sample))
        r_action.extend(sample)
        
        return r_action

class BittleRL():
    
    def __init__(self, env, experience_buffer, actor, critic):             
        super().__init__()

        # Need to define additional params
        # Self max iterations

        self.env = env
        self.experience_buffer = experience_buffer
        self.actor = actor
        self.critic = critic
        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = []
        self.log_data = 0
        self.log_data_freq = 512 # data is logged every 512 steps

    def training():
        self.iterations = 0

        while self.iterations < self.max_iterations:
            params = self.training_iteration(params, optimizers)

            if self.iterations % self.log_data_freq ==0:
                wandb.log({'Iterations': self.iterations})

            self.iterations += 1

            if self.iterations % self.reset_frequency == 0:
                params, optimizers = reset_params(params, optimizers, self.learning_rate) # To do in utils

                self.log_alpha_skill = torch.tensor(self.log_alpha_skill.item(), dtype=torch.float16,
                                                    requires_grad=True, device=self.device)
                self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=self.learning_rate)

        return params


    def training_iteration(self, params, optimizers, transition):
        obs, next_obs, a, reward, done = transition
        next_a, _, _, _ = self.actor.run_policy(params, next_obs)

        self.experience_buffer.add(obs, next_obs, a, next_a, reward, done)

        log_data = True if self.log_data & self.log_data_freq == 0 else False

        if self.experience_buffer.size >= self.batch_size:
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False
                policy_loss, critic_loss = self.losses(params, log_data)
                params = Adam_update(params, losses, optimizers)
                polyak_update(params['Critic'].values(),
                              params['Target_critic'].values(), 0.005)

        return params


    def losses(self, params, log_data):
        batch = self.experience_buffer.sample(batch_size=self.batch_size)

        # Need to change obs to consider the separate obs

        obs = torch.from_numpy(batch.observations).to(self.device)
        next_obs = torch.from_numpy(batch.next_observations).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        next_z = torch.from_numpy(batch.next_z).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        cum_reward = torch.from_numpy(batch.cum_reward).to(self.device)
        norm_cum_reward = torch.from_numpy(batch.norm_cum_reward).to(self.device)
        
        
        target_critic_arg = torch.cat([next_obs, next_z], dim=1)
        critic_arg = torch.cat([obs, z], dim=1)

        with torch.no_grad():
            q_target = self.eval_critic(target_critic_arg, params,
                                        target_critic=True)

        q_target = rew + (self.discount * q_target) * (1 - dones)

        q = self.eval_critic(critic_arg, params)

        critic_loss = F.mse_loss(q.squeeze(), q_target.squeeze())

        # Policy loss
        sample, pdf, mu, std = self.actor.run_policy(params, obs)

        q_pi_arg = torch.cat([obs, sample], dim=1)
        q_pi = self.eval_critic(params, q_pi_arg)

        entropy_term = -torch.clamp(pdf.entropy(), max=MAX_ENTROPY).mean()
        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        entropy_loss = alpha_skill * entropy_term

        policy_loss = q_pi + entropy_loss

        update_log_alpha(entropy_term)

        return policy_loss, critic_loss


    def eval_critic(self, arg, params, target_critic=False):
        name = 'Target_critic' if target_critic else 'Critic'

        return functional_call(self.critic, params[name], arg)

        
        
        
