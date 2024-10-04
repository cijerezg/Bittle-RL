import socket
import time
from utils.helpers import send_data, get_data
import numpy as np

host_ip = '10.56.136.219'
port = 12345


def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))

    i = 0
    
    while i < 1000:
        time.sleep(.1)
        arr1 = np.zeros((240, 320, 4))
        joints = np.zeros((8))
        dist = np.zeros((1))

        send_data(client_socket, (arr1, joints, dist))

        time.sleep(.2)

        params = get_data(client_socket)
        i += 1
