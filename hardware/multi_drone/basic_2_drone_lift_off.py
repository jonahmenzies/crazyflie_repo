"""
Multi-drone formation control using Lighthouse positioning.
Uses FormationController to coordinate all drones and
DroneState to manage individual drone data and callbacks.

Supports both:
    - Individual drone control: controller.drone_states[URI_1].takeoff()
    - Formation-wide control: controller.takeoff()
"""

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.log import LogConfig
import numpy as np
import math

# Change these to match your drones
URI_1 = 'radio://0/10/2M/E7E7E7E7E7'
URI_2 = 'radio://0/60/2M/E7E7E7E7E7'
URI_3 = 'radio://0/90/2M/E7E7E7E7E7'
SURVEY_ALT = 0.5
uris = [URI_1, URI_2, URI_3]


class FormationController:
    def __init__(self, swarm):
        self.swarm = swarm
        self.drone_states = {}

    def setup(self, scf):
        drone = DroneState(scf)
        drone.start_logging()
        drone.wait_for_data()
        self.drone_states[scf.cf.link_uri] = drone

    def initialise(self):
        self.swarm.parallel_safe(self.setup)

    # ==================== FORMATION COMMANDS ====================

    def takeoff(self):
        for uri, drone in self.drone_states.items():
            drone.current_setpoint = {'x': drone.x, 'y': drone.y, 'z': SURVEY_ALT, 'yaw': drone.yaw}
        self.fly_to_setpoints()

    def hover(self, hover_time):
        for uri, drone in self.drone_states.items():
            drone.current_setpoint = {'x': drone.x, 'y': drone.y, 'z': drone.z, 'yaw': drone.yaw}
        start = time.time()
        while time.time() - start < hover_time:
            self.send_setpoints()
            time.sleep(0.05)

    def orient(self, desired_yaw):
        for uri, drone in self.drone_states.items():
            drone.current_setpoint = {'x': drone.x, 'y': drone.y, 'z': drone.z, 'yaw': desired_yaw}
        self.fly_to_setpoints()

    def land(self):
        max_z = max(drone.current_setpoint['z'] for drone in self.drone_states.values())
        heights = np.arange(max_z, 0.05, -0.05)
        for altitude in heights:
            for uri, drone in self.drone_states.items():
                drone.current_setpoint = {'x': drone.x, 'y': drone.y, 'z': altitude, 'yaw': drone.yaw}
            self.fly_to_setpoints()
        for uri, drone in self.drone_states.items():
            drone.scf.cf.commander.send_stop_setpoint()

    # ==================== FORMATION CORE ====================

    def fly_to_setpoints(self):
        while not self.all_reached():
            self.send_setpoints()
            time.sleep(0.05)

    def send_setpoints(self):
        for uri, drone in self.drone_states.items():
            drone.scf.cf.commander.send_position_setpoint(
                drone.current_setpoint['x'],
                drone.current_setpoint['y'],
                drone.current_setpoint['z'],
                drone.current_setpoint['yaw'])

    def all_reached(self):
        for uri, drone in self.drone_states.items():
            if not drone.waypoint_reached_check():
                return False
        return True

    def stop_logging(self):
        for uri, drone in self.drone_states.items():
            drone.stop_logging()

    # ==================== FORMATION OFFSETS ====================

    def offset_from_target(self, leader, follower, dx, dy, dz):
        self.drone_states[follower].current_setpoint['x'] = self.drone_states[leader].current_setpoint['x'] + dx
        self.drone_states[follower].current_setpoint['y'] = self.drone_states[leader].current_setpoint['y'] + dy
        self.drone_states[follower].current_setpoint['z'] = self.drone_states[leader].current_setpoint['z'] + dz

    def offset_from_actual(self, leader, follower, dx, dy, dz):
        self.drone_states[follower].current_setpoint['x'] = self.drone_states[leader].x + dx
        self.drone_states[follower].current_setpoint['y'] = self.drone_states[leader].y + dy
        self.drone_states[follower].current_setpoint['z'] = self.drone_states[leader].z + dz


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

    # ==================== LOGGING ====================

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

    # ==================== CALLBACKS ====================

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

    # ==================== INDIVIDUAL DRONE COMMANDS ====================

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
        self.current_setpoint = {'x': self.x, 'y': self.y, 'z': self.z, 'yaw': self.yaw}
        start = time.time()
        while time.time() - start < hover_time:
            self.scf.cf.commander.send_position_setpoint(
                self.current_setpoint['x'],
                self.current_setpoint['y'],
                self.current_setpoint['z'],
                self.current_setpoint['yaw'])
            time.sleep(0.05)

    # ==================== CHECKS ====================

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
    factory = CachedCfFactory(rw_cache='./cache')

    with Swarm(uris, factory=factory) as swarm:
        controller = FormationController(swarm)
        controller.initialise()

        try:
            # Formation-wide control:
            controller.takeoff()
            controller.hover(3.0)
            controller.orient(0.0)
            controller.hover(3.0)
            controller.land()

            # Individual drone control example:
            # controller.drone_states[URI_1].takeoff()
            # controller.drone_states[URI_1].hover(3.0)
            # controller.drone_states[URI_1].land()

            # Formation offset example:
            # controller.drone_states[URI_1].current_setpoint = {'x': 0.0, 'y': 0.0, 'z': 0.5, 'yaw': 0.0}
            # controller.offset_from_target(URI_1, URI_2, 0.5, 0.0, 0.0)
            # controller.offset_from_target(URI_1, URI_3, -0.5, 0.0, 0.0)
            # controller.fly_to_setpoints()

        except KeyboardInterrupt:
            controller.land()
            print("\nStopping...")
        finally:
            controller.stop_logging()


if __name__ == '__main__':
    main()
