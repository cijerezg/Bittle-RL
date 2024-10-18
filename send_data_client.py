import subprocess
import time
import os

# you may need to run
# ssh-agent bash
# ssh-add
# to be able to send data without asking for password


def shell_cmd(ip, path, file):
    return f'scp {file} carlos@{ip}:{path}'
    

def main():
    #ip = '192.168.0.155' # Liz ip
    ip = '10.1.207.51'# UWM ip
    local_path = 'experiences'
    remote_path = '/home/carlos/Documents/Research/Petoi/Bittle-RL/'
    
    time.sleep(3)
    
    while True:
        if os.listdir(local_path):            
            files = os.listdir(local_path)
            files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            for file in files:
                local_file = os.path.join(local_path, file)
                remote_file = os.path.join(remote_path, local_file)
                cmd = shell_cmd(ip, remote_file, local_file)
                subprocess.run(cmd, shell=True) 
                subprocess.run(f'rm {local_file}', shell=True)
        time.sleep(.1)
                
        
main()
