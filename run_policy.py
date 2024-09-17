from picamera2 import Picamera2
import V53L0X # dist sensor
import time
from ardSerial import *

class Robot():
    def __init__(self, policy):
        self.policy = policy

        # Initialize robot
        model = 'Bittle'
        postureTable = postureDict[model]
        self.goodPorts = {}
        connectPort(goodPorts)
        t=threading.Thread(target=keepCheckingPort, args=(goodPorts,))
        t.start()

        # Initialize camera
        self.cam = Picamera2()
        camera_config = cam.create_preview_configuration()
        self.cam.configure(camera_config)
        self.cam.start()

        # Initialize distance sensor
        tof = VL53L0X.Vl53L0X(i2c_bus=1, i2c_address=0x29)
        tof.open()
        tof.start_ranging(VL53L0X.Vl53l0XAccuracyMode.BETTER)
        self.timing = tof.get_timing()

    def capture_image(self):
        return self.cam.capture_array()

    def compute_distance(self):
        return self.tof.get_distance() # distance in mm

    def get_action(self, state):
        a, mu = policy(state)
        return a, mu

    def execute_action(self, action):
        task = ['I', action]
        send(self.goodPorts, task)

    def closeAll(self):
        closeAllSerial(self.goodPorts)
        self.tof.stop_ranging()
        self.tof.close()
        self.cam.stop_preview()
        self.cam.close()

        
        
