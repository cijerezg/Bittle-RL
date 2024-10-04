import socket
from rl.agent import BittleRL
from rl.replay_buffer import ReplayBuffer
from models.architectures import Policy, Critic
from rl.agent import Actor
from utils.helpers import get_data, send_data, get_params
from utils.optimization import reset_params, set_optimizers
import wandb
import os
import torch
import pickle
import numpy as np
import time

# test

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

os.environ['WANDB_SILENT'] = "true"
wandb.login()


data_folder = 'Data'
policy_folder = 'Checkpoints'

config = {
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'hidden_dim_critic': 256,
    'hidden_dim_policy': 128,

    'batch_size': 256,
    'action_range': 4,
    'learning_rate': 3e-4,
    'discount': 0.97,
    'gradient_steps': 4,

    'reset_frequency': 2000,
    'delta_entropy': 25,
    'load_pretrained_models': False,
    'max_iterations': 10000
}

    

def main(config=None):
    with wandb.init(project='BittleRL', config=config):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            
        # Set up socket communication
        # server_address = ('', 12345)
  
        # server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # server_socket.bind(server_address)
        # server_socket.listen(1)
        # conn, addr = server_socket.accept()

        iterations = 0


        while iterations < config.max_iterations:
            #transition = get_data(conn)

            transition = (np.zeros((240, 320, 4), dtype=np.float32), np.zeros(8, dtype=np.float32),
                          np.zeros(1, dtype=np.float32), np.zeros((8, 8), dtype=np.float32))
            now = time.time()
    
            params = bittle_rl.training_iteration(params, optimizers, transition)
            
            iterations += 1

            if iterations % config.reset_frequency == 0:
                keys = ['Policy', 'Critic']
                params, optimizers, agent = reset_params(params, names, optimizers, keys, config.learning_rate)

            #send_data(server_socket, (params))

            
            
#if __name__ = "__main__":
#    val = main()

main(config)
  
  
