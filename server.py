import socket

def main():
    server_address = ('', 12345)
  
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_address)
    server_socket.listen(1)

    conn, addr = server_socket.accept()

    with conn:
        data = conn.recv(2048)
        if data:
            val = data.decode()
        else:
            print('No data')
            val = None
    
    return val

if __name__ = "__main__":
    val = main()
    
  
  
