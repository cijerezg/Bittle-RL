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


INIT_LOG_ALPHA = 0
MAX_ENTROPY = 40

class Actor():
    def __init__(self, device):
        self.policy = Policy(device)
        self.max_angle = 125

    def run_policy(self, params, x):
        sample, density, mu, std = functional_call(self.policy, params['Policy'], x)
        return sample, density, mu, std

    def robot_action(self, sample):
        r_action = [8, 0, 0, 1] # 8 frames; 0 pitch; 0 roll; angle multiplier is 1

        sample = sample.cpu().detach().numpy()
        sample = 25 * sample # The action range was set to -5 and 5, and the angle range -125 to 125
        sample = sample.flatten().astype(np.float32).tolist()
        r_action.extend(sample)
        
        return r_action

class BittleRL(hyper_params):
    def __init__(self, env, experience_buffer, actor, critic, args):             
        super().__init__(args)

        # Need to define additional params
        # Self max iterations
        self.experience_buffer = experience_buffer
        self.actor = actor
        self.critic = critic
        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = []
        self.log_data = 0
        self.log_data_freq = 512 # data is logged every 512 steps

        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True, device=self.device)

        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.learning_rate)
        

    def training_iteration(self, params, optimizers, transition):
        self.experience_buffer.add(transition)

        log_data = True if self.log_data & self.log_data_freq == 0 else False

        if self.experience_buffer.size >= self.batch_size:
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False
                policy_loss, critic_loss = self.losses(params, log_data)
                losses = [policy_loss, critic_loss]
                keys = ['Policy', 'Critic']
                params = Adam_update(params, losses, keys, optimizers)
                polyak_update(params['Critic'].values(),
                              params['Target_critic'].values(), 0.005)

        return params


    def losses(self, params, log_data):
        batch = self.experience_buffer.sample(batch_size=self.batch_size)

        image = torch.from_numpy(batch.image).to(self.device)
        dist = torch.from_numpy(batch.dist).to(self.device)
        joints = torch.from_numpy(batch.joints).to(self.device)
        next_image = torch.from_numpy(batch.image).to(self.device)
        next_dist = torch.from_numpy(batch.distance).to(self.device)
        next_joints = torch.from_numpy(batch.joints).to(self.device)
        a = torch.from_numpy(batch.z).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)

        
        with torch.no_grad():
            next_a = self.actor.run_policy(params, (next_image, next_joints, next_dist))

        
        target_critic_arg = (next_image, next_joints, next_dist, next_a)
        critic_arg = (image, joints, dist, a)

        with torch.no_grad():
            q_target = self.eval_critic(target_critic_arg, params,
                                        target_critic=True)

        q_target = rew + (self.discount * q_target)

        q = self.eval_critic(critic_arg, params)

        critic_loss = F.mse_loss(q.squeeze(), q_target.squeeze())

        # Policy loss
        sample, pdf, mu, std = self.actor.run_policy(params, (image, joints, dist))

        q_pi_arg = (image, joints, dist, sample)
        q_pi = self.eval_critic(params, q_pi_arg)

        entropy_term = -torch.clamp(pdf.entropy(), max=MAX_ENTROPY).mean()
        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        entropy_loss = alpha_skill * entropy_term

        policy_loss = q_pi + entropy_loss

        self.update_log_alpha(entropy_term)

        return policy_loss, critic_loss


    def eval_critic(self, arg, params, target_critic=False):
        name = 'Target_critic' if target_critic else 'Critic'

        return functional_call(self.critic, params[name], arg)

    def update_log_alpha(self, entropy_term):
        loss = torch.exp(self.log_alpha_skill) * \
            (entropy_term - self.delta_skill).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss.backward()
        self.optimizer_alpha_skill.step()
        

        
        
        
