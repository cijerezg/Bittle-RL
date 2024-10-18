import subprocess
import time
import os
import pdb


def shell_cmd(ip, path, file):
    return f'scp {file} carlos@{ip}:{path}'
    

def main():
    ip = '192.168.0.102' #Liz ip
    #ip = '10.22.134.36'
    local_path = 'checkpoints'
    remote_path = '/home/carlos/Documents/Bittle-RL/'

    time.sleep(3)

    while True:
        if os.listdir(local_path):
            files = os.listdir(local_path)
            files.sort()            
            #for file in os.listdir(local_path):
            for file in files:
                local_file = os.path.join(local_path, file)
                remote_file = os.path.join(remote_path, local_file)
                cmd = shell_cmd(ip, remote_file, local_file)
                subprocess.run(cmd, shell=True)
                subprocess.run(f'rm {local_file}', shell=True)

        time.sleep(10)
        
main()
