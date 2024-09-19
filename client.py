import socket
from run_policy import Robot
import pickle
import numpy as np

host_ip = '10.56.136.219'
port = 12345

def main(policy):
    bittle = Robot(policy)
    action = np.zeros(16)
    bittle.execute_action(action)
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))
    
    image = bittle.capture_image()
    dist = bittle.compute_distance()
    #action = policy(action, image, dist)

    bittle.execute_action(action)
    
    image = image.tostring()
    dist = dist.tostring()
    action = action.tostring()

    client_socket.sendall(image)
    client_socket.sendall(dist)
    client_socket.sendall(action)
    
    
if __name__ == "__main__":
    main(None)
