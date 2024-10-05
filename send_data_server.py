import subprocess
import time
import os



def shell_cmd(ip, path, file):
    return f'scp {file} carlos@{ip}:{path}'
    

def main():
    ip = '192.168.0.241'
    local_path = 'checkpoints'
    remote_path = '/home/carlos/Documents/Bittle-RL/'
    

    while True:
        if os.listdir(local_path):
            for file in os.listdir(local_path):
                local_file = os.path.join(local_path, file)
                remote_file = os.path.join(remote_path, local_file)
                cmd = shell_cmd(ip, remote_file, local_file)
                subprocess.run(cmd, shell=True)
                subprocess.run(f'rm {local_file}', shell=True)                
        else:
            time.sleep(5)
                        
        
main()
