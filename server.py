import socket
from rl.agent import BittleRL
from rl.replay_buffer import ReplayBuffer
from models.architectures import Policy, Critic
from rl.agent import Actor
import wandb
import os
import torch
import pickle
import numpy as np


torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

os.environ['WANDB_SILENT'] = "true"
wandb.login()


data_folder = 'Data'
policy_folder = 'Checkpoints'

config = {
    'device': 'cuda',
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
        
        config = wandb.config

        policy = Policy(#enter arguments)
        critic = Critic(#enter arguments)
        actor = Actor(#enter arguments)

        experience_buffer = ReplayBuffer(# enter arguments)

        bittle_rl = BittleRL(# enter arguments)

        models = [# enter arguments]

        pretrained_models = # enter this
        names = # enter this

        params = get_params(models, pretrained_models)

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
            




    with conn:
        data = conn.recv(2048)
        if data:
            val = data.decode()
        else:
            print('No data')
            val = None
    
    return val

if __name__ = "__main__":
    val = main()
    
  
  
