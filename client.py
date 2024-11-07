import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, save_experiences, load_params, create_dir
import torch
import time
import os
import pdb


MAX_STEPS = 400
FRAMES = 8
ACTION_DIM = 8

path_exp = 'experiences'
path_params = 'checkpoints'


def main():
    create_dir(path_exp)
    create_dir(path_params)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    actor = Actor(device)

    model, name = [actor.policy], ['Policy']
    
    pretrained_params = [None]
    params = get_params(model, name, pretrained_params)

    bittle = Robot(actor)

    step = 0
    time.sleep(1)

    updated_policy = load_params(path_params)
    if updated_policy:
        params['Policy'] = updated_policy

    speed = 0
    prev_action = np.zeros((1, 8), dtype=np.float32)
        
    while step < MAX_STEPS:
        dist = bittle.compute_distance()

        if step > 0:
            speed = old_dist - dist

        speed = np.array(speed, dtype=np.float32)

        action, sample_action = bittle.get_action(params, (prev_action, speed))
        sample_action = sample_action.detach().numpy().squeeze()
        
        save_experiences(path_exp, (sample_action, speed), step)

        prev_action = sample_action        

        bittle.execute_action(action)
        step += 1
        
        print(f'Step is :{step}; speed is {speed}; distance is {dist})

        old_dist = dist
               
        if step % 5 == 0:
            updated_policy = load_params(path_params)
        
            if updated_policy:
                params['Policy'] = updated_policy
    bittle.closeAll()
        
    
if __name__ == "__main__":
    main()
    
    
    
