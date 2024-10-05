import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, LimitedQueue, save_experiences, load_params
import torch
import time


host_ip = '10.56.136.219'

MAX_STEPS = 1000
FRAMES = 8
ACTION_DIM = 8

path_exp = 'experiences'
path_params = 'checkpoints'


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    actor = Actor(device)

    model, name, pretrained_params  = [actor.policy], ['Policy'], [None]
    params = get_params(model, name, pretrained_params)

    bittle = Robot(actor)
    action = [FRAMES, 0, 0, 1]
    init_joints = [0] * (FRAMES * ACTION_DIM) # because 8 frames are being used
    action.extend(init_joints)
    bittle.execute_action(action)   
    
    image_queue = LimitedQueue((240, 320, 4))
    dist_queue = LimitedQueue((1))
    joints_queue = LimitedQueue((8))
    
    step = 0

    while step < MAX_STEPS:
        image = bittle.capture_image()
        dist = bittle.compute_distance()

        image_queue.add(image)
        dist_queue.add(np.array([dist], dtype=np.float32))
        joints_queue.add(np.array(action[-8:], dtype=np.float32))

        action, sample_action = bittle.get_action(params, (image_queue.get_items(),
                                                           joints_queue.get_items(),
                                                           dist_queue.get_items()))

        save_experiences(path_exp, (image, dist, sample_action), step) 
        
        bittle.execute_action(action)
        step += 1
        
        params = load_params(path_params)
    
    
if __name__ == "__main__":
    main()
    
    
    
