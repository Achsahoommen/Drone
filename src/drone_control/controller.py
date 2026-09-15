import time
import threading
from typing import Optional, Callable
from pathlib import Path
from enum import Enum

from utils.config import CONFIG, DRONE_CFG, setup_logging

logger = setup_logging("drone_control")


class DroneState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    STANDBY = "standby"
    ARMING = "arming"
    TAKEOFF = "takeoff"
    FLYING = "flying"
    SPRAYING = "spraying"
    RETURNING = "returning"
    LANDING = "landing"
    LANDED = "landed"
    ERROR = "error"


class DroneController:
    def __init__(self, connection_string=None, simulate=True):
        self.connection_string = connection_string or DRONE_CFG["connection_string"]
        self.simulate = simulate
        self.state = DroneState.DISCONNECTED
        self.vehicle = None
        self.current_altitude = 0.0
        self.target_altitude = DRONE_CFG.get("takeoff_altitude", 5.0)
        self.ground_speed = DRONE_CFG.get("ground_speed", 3.0)
        self._state_callbacks = []
        self._is_armed = False

        logger.info("DroneController initialized | Sim: {} | Conn: {}".format(
            simulate, self.connection_string
        ))

    def register_state_callback(self, callback: Callable):
        self._state_callbacks.append(callback)

    def _update_state(self, new_state: DroneState):
        old = self.state
        self.state = new_state
        logger.info("State: {} -> {}".format(old.value, new_state.value))
        for cb in self._state_callbacks:
            cb(old, new_state)

    def connect(self, timeout=30):
        self._update_state(DroneState.CONNECTING)

        if self.simulate:
            logger.info("SIM: Connecting to drone...")
            time.sleep(1)
            self._update_state(DroneState.STANDBY)
            return True

        try:
            from dronekit import connect
            self.vehicle = connect(
                self.connection_string,
                baud=DRONE_CFG.get("baud_rate", 57600),
                timeout=timeout,
            )
            self._update_state(DroneState.STANDBY)
            logger.info("Connected to drone")
            return True
        except Exception as e:
            logger.error("Connection failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def arm(self):
        if self.state != DroneState.STANDBY:
            logger.error("Cannot arm: current state is {}".format(self.state.value))
            return False

        self._update_state(DroneState.ARMING)

        if self.simulate:
            logger.info("SIM: Arming motors...")
            time.sleep(1)
            self._is_armed = True
            return True

        try:
            self.vehicle.armed = True
            while not self.vehicle.armed:
                time.sleep(0.5)
            self._is_armed = True
            logger.info("Motors armed")
            return True
        except Exception as e:
            logger.error("Arming failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def takeoff(self, altitude=None):
        if altitude is None:
            altitude = self.target_altitude

        if not self._is_armed:
            logger.error("Cannot takeoff: not armed")
            return False

        self._update_state(DroneState.TAKEOFF)
        logger.info("Taking off to {}m".format(altitude))

        if self.simulate:
            for alt in [float(i) for i in range(1, int(altitude) + 1)]:
                self.current_altitude = alt
                logger.info("  Altitude: {}m".format(alt))
                time.sleep(0.5)
            self.current_altitude = altitude
            self._update_state(DroneState.FLYING)
            return True

        try:
            self.vehicle.simple_takeoff(altitude)
            while True:
                alt = self.vehicle.location.global_relative_frame.alt
                self.current_altitude = alt
                if alt >= altitude * 0.95:
                    break
                time.sleep(0.5)
            self._update_state(DroneState.FLYING)
            return True
        except Exception as e:
            logger.error("Takeoff failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def goto_position(self, lat, lon, alt=None):
        if alt is None:
            alt = self.current_altitude

        if self.simulate:
            logger.info("SIM: Going to ({}, {}, {})".format(lat, lon, alt))
            time.sleep(1)
            return True

        try:
            from dronekit import LocationGlobalRelative
            target = LocationGlobalRelative(lat, lon, alt)
            self.vehicle.simple_goto(target, groundspeed=self.ground_speed)
            return True
        except Exception as e:
            logger.error("Navigation failed: {}".format(e))
            return False

    def set_altitude(self, altitude):
        if self.simulate:
            self.current_altitude = altitude
            logger.info("SIM: Altitude set to {}m".format(altitude))
            return True
        return False

    def activate_spray(self, duration_sec, flow_rate_lpm=1.0):
        self._update_state(DroneState.SPRAYING)
        logger.info(
            "SPRAY ON: {}s @ {} LPM".format(duration_sec, flow_rate_lpm)
        )

        if self.simulate:
            time.sleep(min(duration_sec, 2))
            logger.info("SIM: Spray complete")
            self._update_state(DroneState.FLYING)
            return True

        try:
            time.sleep(duration_sec)
            logger.info("Spray complete")
            self._update_state(DroneState.FLYING)
            return True
        except Exception as e:
            logger.error("Spray failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def land(self):
        self._update_state(DroneState.LANDING)
        logger.info("Initiating landing")

        if self.simulate:
            for alt in range(int(self.current_altitude), 0, -1):
                self.current_altitude = float(alt)
                time.sleep(0.3)
            self.current_altitude = 0
            self._is_armed = False
            self._update_state(DroneState.LANDED)
            return True

        try:
            self.vehicle.mode = "LAND"
            while self.current_altitude > 0.5:
                time.sleep(1)
            self._is_armed = False
            self._update_state(DroneState.LANDED)
            return True
        except Exception as e:
            logger.error("Landing failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def return_to_launch(self):
        self._update_state(DroneState.RETURNING)
        logger.info("Returning to launch")

        if self.simulate:
            time.sleep(2)
            return self.land()

        try:
            self.vehicle.mode = "RTL"
            return True
        except Exception as e:
            logger.error("RTL failed: {}".format(e))
            self._update_state(DroneState.ERROR)
            return False

    def emergency_stop(self):
        logger.warning("EMERGENCY STOP")
        self._is_armed = False
        if self.simulate:
            self._update_state(DroneState.LANDED)
            return True

        try:
            self.vehicle.armed = False
            self.vehicle.mode = "LAND"
            return True
        except Exception as e:
            logger.error("Emergency stop failed: {}".format(e))
            return False

    def disconnect(self):
        if self.vehicle:
            try:
                self.vehicle.close()
            except Exception:
                pass
        self._update_state(DroneState.DISCONNECTED)
        logger.info("Disconnected")


if __name__ == "__main__":
    drone = DroneController(simulate=True)

    drone.register_state_callback(
        lambda old, new: print("  State: {} -> {}".format(old.value, new.value))
    )

    if drone.connect():
        if drone.arm():
            if drone.takeoff(3):
                time.sleep(2)
                drone.activate_spray(3, flow_rate_lpm=1.0)
                time.sleep(1)
                drone.land()
    drone.disconnect()
