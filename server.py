import socket
from rl.agent import BittleRL
from rl.replay_buffer import ReplayBuffer
from models.architectures import Policy, Critic
from rl.agent import Actor
from utils.helpers import get_data, send_data
from utils.optimizers import reset_params
import wandb
import os
import torch
import pickle
import numpy as np

# test

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

os.environ['WANDB_SILENT'] = "true"
wandb.login()


data_folder = 'Data'
policy_folder = 'Checkpoints'

config = {
    'hidden_dim_critic': 256,
    'hidden_dim_policy': 128,

    'batch_size': 256,
    'action_range': 4,
    'learning_rate': 3e-4,
    'discount': 0.97,
    'gradient_steps': 4,

    'max_iterations': int(6.4e4) - 1,
    'buffer_size': int(6.4e6) - 1,
    'reset_frequency': 2000,
    'delta_entropy': 25,
    'load_pretrained_models': False
}

    

def main(config=None):
    with wandb.init(project='BittleRL', config=config):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    

        config = wandb.config

        actor = Actor(device)
        critic = Critic(device)

        experience_buffer = ReplayBuffer(int(1e8))
        bittle_rl = BittleRL(experience_buffer, actor, critic, config)

        models = [actor.policy, critic, critic]
        names = ['Policy', 'Critic', 'Target_critic']
        pretrained_models = [None, None, None]

        params = get_params(models, names, pretrained_models)

        keys_optimizers = ['Critic', 'Policy']
        optimizers = set_optimizers(params, keys_optimizers, config.learning_rate)
            
        # Set up socket communication
        server_address = ('', 12345)
  
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(server_address)
        server_socket.listen(1)
        conn, addr = server_socket.accept()

        iterations = 0


        while iterations < config.max_iterations:
            transition = get_data(conn)
    
            params = bittle_rl.training_iteration(params, optimizers, transition)

            if iterations % config.reset_frequency == 0:
                params, optimizers, agent = reset_params(params, names, optimizers, config.learning_rate)

            send_data(server_socket, (params))
            iterations += 1
            
            
if __name__ = "__main__":
    val = main()
    
  
  
