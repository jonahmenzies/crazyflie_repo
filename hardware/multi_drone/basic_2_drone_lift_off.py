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
from cflib.crazyflie.console import Console
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.log import LogConfig
import numpy as np
import math


# --- silence cflib boot-console decode noise -------------------------------
def _tolerant_incoming(self, packet):
    try:
        text = packet.data.decode('UTF-8', errors='replace')
    except Exception:
        return
    self.receivedChar.call(text)


Console._incoming = _tolerant_incoming
# ---------------------------------------------------------------------------


# Change these to match your drones
URI_1 = 'radio://0/10/2M/E7E7E7E701'
URI_2 = 'radio://0/60/2M/E7E7E7E706'
URI_3 = 'radio://1/90/2M/E7E7E7E709'
SURVEY_ALT = 0.5
uris = [URI_1, URI_2, URI_3]

# supervisor.info bit positions — VERIFY against your firmware version
CAN_FLY    = 1 << 3
IS_TUMBLED = 1 << 5
IS_LOCKED  = 1 << 6


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

    # ==================== PREFLIGHT ====================

    def preflight_check(self, timeout=5.0):
        """Refuse to fly if any drone reports locked/tumbled/cannot-fly.

        After an emergency stop or a tumble the drone latches into a locked
        state that only clears on power cycle. Ask the drone directly rather
        than tracking reboots by hand.
        """
        results = {}

        def check(scf):
            uri = scf.cf.link_uri
            lg = LogConfig(name='Supervisor', period_in_ms=100)
            lg.add_variable('supervisor.info', 'uint16_t')
            state = {'info': None}

            def cb(ts, msg, conf):
                state['info'] = msg['supervisor.info']

            try:
                scf.cf.log.add_config(lg)
                lg.data_received_cb.add_callback(cb)
                lg.start()
                start = time.time()
                while state['info'] is None and time.time() - start < timeout:
                    time.sleep(0.05)
                lg.stop()
            except Exception as ex:
                print(f'  {uri}  supervisor log failed ({type(ex).__name__})')

            results[uri] = state['info']

        self.swarm.parallel_safe(check)

        print('\nPreflight check:')
        ok = True
        for uri in uris:
            info = results.get(uri)
            if info is None:
                print(f'  {uri}  NO SUPERVISOR DATA')
                ok = False
                continue
            problems = []
            if info & IS_LOCKED:
                problems.append('LOCKED')
            if info & IS_TUMBLED:
                problems.append('TUMBLED')
            if not (info & CAN_FLY):
                problems.append('CANNOT FLY')
            if problems:
                print(f'  {uri}  {", ".join(problems)} — power cycle required')
                ok = False
            else:
                print(f'  {uri}  ready')

        if not ok:
            raise RuntimeError('preflight check failed — power cycle affected drones')
        print('All drones ready.\n')

    def reset_estimators(self):
        for uri, drone in self.drone_states.items():
            drone.scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.2)
        for uri, drone in self.drone_states.items():
            drone.scf.cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

    def emergency_stop(self):
        for uri, drone in self.drone_states.items():
            try:
                drone.scf.cf.commander.send_stop_setpoint()
            except Exception:
                pass

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
        # Freeze x/y at current position so a drifting drone doesn't chase
        # its own drift on the way down.
        hold = {uri: (d.x, d.y, d.yaw) for uri, d in self.drone_states.items()}
        max_z = max(drone.current_setpoint['z'] for drone in self.drone_states.values())
        heights = np.arange(max_z, 0.05, -0.05)
        for altitude in heights:
            for uri, drone in self.drone_states.items():
                hx, hy, hyaw = hold[uri]
                drone.current_setpoint = {'x': hx, 'y': hy, 'z': altitude, 'yaw': hyaw}
            self.fly_to_setpoints(timeout=3.0)
        for uri, drone in self.drone_states.items():
            drone.scf.cf.commander.send_stop_setpoint()

    # ==================== FORMATION CORE ====================

    def fly_to_setpoints(self, timeout=10.0):
        start = time.time()
        while not self.all_reached():
            if time.time() - start > timeout:
                print('  timeout waiting for setpoint convergence')
                return
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

    def fly_to_setpoint(self, timeout=10.0):
        start = time.time()
        while not self.waypoint_reached_check():
            if time.time() - start > timeout:
                print('  timeout waiting for setpoint convergence')
                return
            self.scf.cf.commander.send_position_setpoint(
                self.current_setpoint['x'],
                self.current_setpoint['y'],
                self.current_setpoint['z'],
                self.current_setpoint['yaw'])
            time.sleep(0.05)

    def land(self):
        hx, hy, hyaw = self.x, self.y, self.yaw
        heights = np.arange(self.current_setpoint['z'], 0.05, -0.05)
        for altitude in heights:
            self.current_setpoint = {'x': hx, 'y': hy, 'z': altitude, 'yaw': hyaw}
            self.fly_to_setpoint(timeout=3.0)
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

        # Refuse to arm if any drone is still latched from a previous stop.
        controller.preflight_check()

        controller.reset_estimators()

        try:
            controller.takeoff()
            controller.hover(3.0)
            controller.orient(0.0)
            controller.hover(3.0)
            controller.land()

        except KeyboardInterrupt:
            print('\nEmergency stop')
            controller.emergency_stop()
        finally:
            controller.stop_logging()


if __name__ == '__main__':
    main()
