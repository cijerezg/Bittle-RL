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

    def compute_distance(self):
        return self.tof.get_distance() # distance in mm

    def get_action(self, params, state):
        joints, dist = state
        joints = torch.tensor(joints).unsqueeze(0)
        dist = torch.tensor(dist).unsqueeze(0)
        state = (joints, dist)
        
        sample, density, mu, std, smooth_sample = self.actor.run_policy(params, state)
        r_action = self.actor.robot_action(smooth_sample)
        return r_action, sample

    def execute_action(self, action):
        task = ['K', action, 1]
        send(self.goodPorts, task)

    def closeAll(self):
        closeAllSerial(self.goodPorts)
        self.tof.stop_ranging()
        self.tof.close()
        
