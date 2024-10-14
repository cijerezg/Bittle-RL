import socket
from rl.agent import BittleRL
from rl.replay_buffer import ReplayBuffer
from models.architectures import Policy, Critic
from rl.agent import Actor
from utils.helpers import get_params, save_params, load_experiences, create_dir
from utils.optimization import reset_params, set_optimizers
import wandb
import os
import torch
import pickle
import numpy as np
import time
import pdb

# test

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

os.environ['WANDB_SILENT'] = "true"
wandb.login()


data_folder = 'Data'
policy_folder = 'checkpoints'
path_exp = 'experiences'
path_init_exp = 'init_experiences'


config = {
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'hidden_dim_critic': 256,
    'hidden_dim_policy': 128,

    'batch_size': 256,
    'action_range': 4,
    'learning_rate': 3e-4,
    'discount': 0.97,
    'gradient_steps': 8,

    'reset_frequency': 2000,
    'delta_entropy': 25,
    'load_pretrained_models': False,
    'max_iterations': 10000
}

    

def main(config=None):
    with wandb.init(project='BittleRL', config=config):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        create_dir(policy_folder)
        create_dir(path_exp)
        
        config = wandb.config

        actor = Actor(device)
        critic = Critic(device).to(device)

        experience_buffer = ReplayBuffer()
        bittle_rl = BittleRL(experience_buffer, actor, critic, config)

        models = [actor.policy, critic, critic]
        names = ['Policy', 'Critic', 'Target_critic']
        pretrained_models = [None, None, None]

        params = get_params(models, names, pretrained_models)

        keys_optimizers = ['Critic', 'Policy']
        optimizers = set_optimizers(params, keys_optimizers, config.learning_rate)
            
        iterations = 0
        
        init_transitions = load_experiences(path_init_exp, delete=False)
        bittle_rl.experience_buffer.add(init_transitions)
                                        
        while iterations < config.max_iterations:
            transitions = load_experiences(path_exp)
    
            params = bittle_rl.training_iteration(params, optimizers, transitions)
            
            iterations += 1
            print(iterations)

            if iterations % config.reset_frequency == 0:
                keys = ['Policy', 'Critic']
                params, optimizers, agent = reset_params(params, names, optimizers, keys, config.learning_rate)

            save_params(policy_folder, params['Policy'])
            
#if __name__ = "__main__":
#    val = main()

main(config)
  
  
