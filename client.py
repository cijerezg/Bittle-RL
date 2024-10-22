import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, save_experiences, load_params, create_dir
import torch
import time
import os
from data.skill_library import *
import pdb

#host_ip = '10.56.136.219' # Liz
host_ip = '10.1.207.51' # UWM IP


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

    model, name, pretrained_params = [actor.policy], ['Policy'], [None]
    params = get_params(model, name, pretrained_params)

    distance_points = 5
    bittle = Robot(actor)
    prefix_action = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    action = np.zeros(8)
    inc = np.array([10, 10, 10, 10, 5, 5, 5, 5])
    
    for i in range(5):
        aux_act = action.tolist()
        bittle.execute_action(prefix_action.extend(aux_act))
        time.sleep(.6)
        action += inc

    step = 0
    sample_action = np.zeros((1, 8))

    time.sleep(1)
    
    while step < MAX_STEPS:

        dist = 0
        for i in range(distance_points):
            measured_dist = np.array(bittle.compute_distance(), dtype=np.float32)
            dist += measured_dist

        dist /= distance_points            
        joints = np.array(sample_action[-1, :], dtype=np.float32)

        action, sample_action = bittle.get_action(params, (joints, dist))
        
                                                
        # if step < MAX_STEPS:
        #     idx += 1
        #     idx = idx % len(skills)
            
        #     action_s = skills[idx]
        #     action = [8, 0, 0, 1]
        #     action.extend(action_s)
        #     sample_action = np.array(action_s, dtype=np.float32)
        #     sample_action = sample_action / 25
        # else:
        #     action, sample_action = bittle.get_action(params, (joints, dist))
        sample_action = sample_action.detach().numpy().squeeze()
            
        save_experiences(path_exp, (joints, dist, sample_action), step) 

        print(f'Step is :{step}')
        
        bittle.execute_action(action)
        step += 1

        if step % 20 == 0:
            updated_policy = load_params(path_params)
        
            if updated_policy:
                params['Policy'] = updated_policy
    bittle.closeAll()
        
    
if __name__ == "__main__":
    main()
    
    
    
