import socket
import time
from utils.helpers import send_data, get_data
import numpy as np
import pickle


def main():
    server_addr = ('', 12345)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_addr)
    server_socket.listen(1)
    conn, addr = server_socket.accept()

    i = 0

    while i < 1000:

        data = []
        while True:
            packet = conn.recv(4096)
            if not packet: break
            data.append(packet)

        print('done receiving data')
        r_dat = pickle.loads(b"".join(data))
                
        time.sleep(4)

        arr = {'val1': np.zeros(30,30), 'val4': np.zeros(30,30),
               'val2': np.zeros(30,30), 'val3': np.zeros(30,30)}


        ser_dict = pickle.dumps(arr)
        server_socket.sendall(ser_dict)
        print(i)
        i += 1


main()
