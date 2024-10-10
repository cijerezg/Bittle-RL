import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, LimitedQueue, save_experiences, load_params, create_dir
import torch
import time
import os
from data.skill_library import *

#host_ip = '10.56.136.219' # Liz
host_ip = '10.1.207.51' # UWM IP


MAX_STEPS = 1000
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

    bittle = Robot(actor)
    action = [FRAMES, 0, 0, 1]
    init_joints = [0] * (FRAMES * ACTION_DIM) # because 8 frames are being used
    action.extend(init_joints)
    bittle.execute_action(action)   
    
    dist_queue = LimitedQueue((1))
    joints_queue = LimitedQueue((8))
    
    step = 0

    while step < MAX_STEPS:
        dist = bittle.compute_distance()
        joints = np.array(action[-8:], dtype=np.float32)

        dist_queue.add(np.array([dist], dtype=np.float32))
        joints_queue.add(joints)

        action, sample_action = bittle.get_action(params, (joints_queue.get_items(),
                                                           dist_queue.get_items()))

        if step < 240:
            idx = np.random.randint(0, len(skills)) # skills come from the skill library
            action_s = skills[idx]
            action = [8, 0, 0, 1]
            action.extend(action_s)
            sample_action = np.array(action_s, dtype=np.float32)
            sample_action = sample_action / 25
        else:
            action, sample_action = bittle.get_action(params, (joints_queue.get_items(),
                                                               dist_queue.get_items()))
            sample_action = sample_action.detach().numpy()
            
        save_experiences(path_exp, (joints, dist, sample_action), step) 

        print(step)
        
        bittle.execute_action(action)
        step += 1

        if step % 50 == 0:
            updated_policy = load_params(path_params)
        
            if updated_policy:
                params['Policy'] = updated_policy
    
    
if __name__ == "__main__":
    main()
    
    
    
