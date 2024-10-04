import socket
import time
from utils.helpers import send_data, get_data
import numpy as np



def main():
    server_addr = ('', 12345)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_addr)
    server_socket.listen(1)
    conn, addr = server_socket.accept()

    i = 0

    while i < 1000:
        data = get_data(conn)

        time.sleep(4)

        arr = {'val1': np.zeros(30,30), 'val1': np.zeros(30,30),
               'val1': np.zeros(30,30), 'val1': np.zeros(30,30)}

        send_data(conn, (arr))
