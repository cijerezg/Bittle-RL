import VL53L0X # dist sensor
import time
from ardSerial import *
import threading
import torch

class Robot():
    def __init__(self, actor):
        self.actor = actor

        # Initialize robot
        model = 'Bittle'
        postureTable = postureDict[model]
        self.goodPorts = {}
        connectPort(self.goodPorts)
        self.t=threading.Thread(target=keepCheckingPort, args=(self.goodPorts,))
        self.t.start()

        # Initialize distance sensor
        self.tof = VL53L0X.VL53L0X(i2c_bus=1, i2c_address=0x29)
        self.tof.open()
        self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
        self.timing = self.tof.get_timing()
        self.idx = 0

    def compute_distance(self):
        return self.tof.get_distance() / 100 # Original distance is in mm, but since it is divided 100, it becomes dm (decimeters)

    def get_action(self, params, state):
        # Add the actions here using the skills that are created
        
        joints, dist = state
        joints = torch.tensor(joints).unsqueeze(0)
        dist = torch.tensor(dist).unsqueeze(0)
        state = (joints, dist)
        
        sample, density, mu, std = self.actor.run_policy(params, state)
        r_action, out_joints = self.actor.robot_action(sample, params, joints)
        
        return r_action, sample, out_joints

    def execute_action(self, action):
        task = ['K', action, .2] # 0.16 This is the time it runs the action for 
        send(self.goodPorts, task)

    def closeAll(self):
        closeAllSerial(self.goodPorts)
        self.tof.stop_ranging()
        self.tof.close()
        
