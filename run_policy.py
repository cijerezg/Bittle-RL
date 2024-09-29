from picamera2 import Picamera2
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
        connectPort(goodPorts)
        t=threading.Thread(target=keepCheckingPort, args=(goodPorts,))
        t.start()

        # Initialize camera
        self.cam = Picamera2()
        camera_config = self.cam.create_preview_configuration()
        self.cam.configure(camera_config)
        self.cam.start()

        # Initialize distance sensor
        self.tof = VL53L0X.VL53L0X(i2c_bus=1, i2c_address=0x29)
        self.tof.open()
        self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
        self.timing = self.tof.get_timing()

    def capture_image(self):
        return self.cam.capture_array()

    def compute_distance(self):
        return self.tof.get_distance() # distance in mm

    def get_action(self, params, state):
        image, dist, joints = state
        image = torch.tensor(image).unsqueeze(0)
        dist = torch.tensor(dist).unsqueeze(0)
        joints = torch.tensor(joints).unsqueeze(0)
        state = (image, dist, joints)
        
        sample, density, mu, std = self.actor.run_policy(params, state)
        r_action = self.actor.robot_action(sample)
        return r_action

    def execute_action(self, action):
        task = ['K', action, 0.001]
        send(self.goodPorts, task)

    def closeAll(self):
        closeAllSerial(self.goodPorts)
        self.tof.stop_ranging()
        self.tof.close()
        self.cam.stop_preview()
        self.cam.close()

        
        
