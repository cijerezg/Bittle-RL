import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, LimitedQueue, send_data, get_data
import torch
import time


host_ip = '10.56.136.219'
port = 12345
MAX_STEPS = 1000
FRAMES = 8
ACTION_DIM = 8

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
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))

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

        send_data(client_socket, (image, dist, sample_action)) 
        
        bittle.execute_action(action)
        step += 1

        params = get_data(client_socket)
    
    
if __name__ == "__main__":
    main()
    
    
    
