import socket
from run_policy import Robot
import pickle
import numpy as np
from rl.agent import Actor
from utils.helpers import get_params, LimitedQueue
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
    action = torch.tensor(action)

    image_queue = LimitedQueue((240, 320, 4))
    dist_queue = LimitedQueue((1))
    joints_queue = LimitedQueue((8))
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect((host_ip, port))

    step = 0

    while step < MAX_STEPS:
        print(step)
        now = time.time()
        image = bittle.capture_image()
        dist = bittle.compute_distance()

        image_queue.add(image)
        dist_queue.add(dist)
        joints_queue.add(action)
        

        action = bittle.get_action(params, (image_queue.get_items(),
                                            dist_queue.get_items(),
                                            joints_queue.get_items()))
        
        bittle.execute_action(action)
        print(time.time() - now)
        step += 1
        
    
#    client_socket.sendall(image)
#    client_socket.sendall(dist)
#    client_socket.sendall(action)
    
    
if __name__ == "__main__":
    main()
    
    
    
