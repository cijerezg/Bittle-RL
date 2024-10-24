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

    model, name = [actor.policy, actor.decoder], ['Policy', 'Decoder']
    decoder_params = torch.load('offline_models/decoder.pt',
                                weights_only=True, map_location=device)
    
    pretrained_params = [None, decoder_params]
    params = get_params(model, name, pretrained_params)

    distance_points = 5
    bittle = Robot(actor)

    step = 0
    time.sleep(1)

    updated_policy = load_params(path_params)
    if updated_policy:
        params['Policy'] = updated_policy

    speed = 0
        
    while step < MAX_STEPS:
        dist = 0
        for i in range(distance_points):
            measured_dist = bittle.compute_distance()
            dist += measured_dist

        dist /= distance_points                
        if step > 0:
            speed = dist - old_dist
            joints = np.zeros((1, 8), dtype=np.float32)

        speed = np.array(speed, dtype=np.float32)

        
        action, sample_action, joints = bittle.get_action(params, (joints, speed))
        sample_action = sample_action.detach().numpy().squeeze()
        joints = joints.detach().numpy()
        
        save_experiences(path_exp, (joints, speed, sample_action), step) 

        print(f'Step is :{step}; speed is {speed}')
        
        bittle.execute_action(action)
        step += 1

        old_dist = dist
               
        if step % 20 == 0:
            updated_policy = load_params(path_params)
        
            if updated_policy:
                params['Policy'] = updated_policy
    bittle.closeAll()
        
    
if __name__ == "__main__":
    main()
    
    
    
