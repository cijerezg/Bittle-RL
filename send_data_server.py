import subprocess
import time
import os



def shell_cmd(ip, path, file):
    return f'scp {file} carlos@{ip}:{path}/{file}'
    

def main():
    ip = '192.168.0.155'
    path = 'home/carlos/Documents/Research/Petoi/Bittle-RL/experiences'
    

    while True:
        if os.listdir(path):
            for file in os.listdir(path):
                cmd = shell_cmd(ip, path, file)
                subprocess.run(cmd, shell=True) 
                subprocess.run(f'rm {file}', shell=True)                
        else:
            time.sleep(4)
                
        
        



