"""
Single drone position reading via Lighthouse using callbacks.
Place drone in Lighthouse field of view, run script to confirm
position estimates are sensible before moving to flight scripts.
"""

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
import numpy as np
import math

# Change this to match your drone
URI = 'radio://0/80/2M/E7E7E7E701'
SURVEY_ALT = 0.5

#class formationController:
#    def __init__(self, URIs):
#        for URI in URIs:
#            drones.append(droneState(URI))

class DroneState:
    def __init__(self, scf):
        self.scf = scf
        
        # Establishes Position Logging
        self.log_pos = LogConfig(name='Position', period_in_ms=100)
        self.log_pos.add_variable('stateEstimate.x', 'float')
        self.log_pos.add_variable('stateEstimate.y', 'float')
        self.log_pos.add_variable('stateEstimate.z', 'float')

        # Establishes Attitude Logging
        self.log_att = LogConfig(name='Attitude', period_in_ms=100)
        self.log_att.add_variable('stabilizer.roll', 'float')
        self.log_att.add_variable('stabilizer.pitch', 'float')
        self.log_att.add_variable('stabilizer.yaw', 'float')
    
        # Variable Establishment
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.attitude_received = False
        self.position_received = False
        self.current_setpoint = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

    def start_logging(self):
        self.scf.cf.log.add_config(self.log_pos)
        self.log_pos.data_received_cb.add_callback(self.position_callback)
        self.log_pos.start()

        self.scf.cf.log.add_config(self.log_att)
        self.log_att.data_received_cb.add_callback(self.attitude_callback)
        self.log_att.start()

    def stop_logging(self):
        self.log_pos.stop()
        self.log_att.stop()

    def wait_for_data(self):
        while not (self.position_received and self.attitude_received):
            time.sleep(0.01)

    def position_callback(self, timestamp, msg_in, logconf):
        self.x = msg_in['stateEstimate.x']
        self.y = msg_in['stateEstimate.y']
        self.z = msg_in['stateEstimate.z']
        self.position_received = True

    def attitude_callback(self, timestamp, msg_in, logconf):
        self.roll = msg_in['stabilizer.roll']
        self.pitch = msg_in['stabilizer.pitch']
        self.yaw = msg_in['stabilizer.yaw']
        self.attitude_received = True

    def takeoff(self):
        self.current_setpoint = {'x': self.x, 'y': self.y, 'z': SURVEY_ALT, 'yaw': self.yaw}
        self.fly_to_setpoint()

    def fly_to_setpoint(self):
        while not self.waypoint_reached_check():
            self.scf.cf.commander.send_position_setpoint(
                self.current_setpoint['x'],
                self.current_setpoint['y'],
                self.current_setpoint['z'],
                self.current_setpoint['yaw'])
            time.sleep(0.05)

    def land(self):
        heights = np.arange(self.current_setpoint['z'], 0.05, -0.05)
        for altitude in heights:
            self.current_setpoint = {'x': self.x, 'y': self.y, 'z': altitude, 'yaw': self.yaw}
            self.fly_to_setpoint()
        self.scf.cf.commander.send_stop_setpoint()

    def orient(self, desired_yaw):
        self.current_setpoint = {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': desired_yaw}
        self.fly_to_setpoint()

    def hover(self, hover_time):
        start = time.time()
        while time.time() - start < hover_time:
            self.scf.cf.commander.send_position_setpoint(
                self.x,
                self.y,
                self.z,
                self.yaw)
            time.sleep(0.05)

    def waypoint_reached_check(self):
        dis_err = math.sqrt(
            (self.x - self.current_setpoint['x'])**2 +
            (self.y - self.current_setpoint['y'])**2 +
            (self.z - self.current_setpoint['z'])**2
        )
        yaw_err = abs(self.yaw - self.current_setpoint['yaw']) % 360
        if yaw_err > 180:
            yaw_err = 360 - yaw_err
        if dis_err < 0.05 and yaw_err < 5.0:
            return True
        return False


def main():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print(f"Connected to {URI}")

        drone = DroneState(scf)
        drone.start_logging()
        drone.wait_for_data()

        try:
            drone.takeoff()
            drone.hover(0.5)
            drone.orient(0.0)
            drone.hover(0.5)
            drone.land()
        except KeyboardInterrupt:
            drone.land()
            print("\nStopping...")
        finally:
            drone.stop_logging()


if __name__ == '__main__':
    main()
