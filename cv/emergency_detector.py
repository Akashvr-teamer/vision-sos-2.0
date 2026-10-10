import os
import sys
import time

try:
    from . import fall_detection
    from .car_accident_detection import CarAccidentDetector
    from .robbery_detection import RobberyDetector
    from .fire_detection import FireDetector
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import fall_detection
    from car_accident_detection import CarAccidentDetector
    from robbery_detection import RobberyDetector
    from fire_detection import FireDetector


class UnifiedEmergencyDetector:
    """
    Unified Multi-Hazard AI Emergency Detector.
    Concurrently analyzes incoming frames across:
    1. Falls & Medical Collapses
    2. Vehicle Collisions & Traffic Accidents
    3. Robbery & Hostage / Violent Threat Postures
    4. Fire, Smoke & Thermal Combustion
    """

    def __init__(self):
        self.car_detector = CarAccidentDetector()
        self.robbery_detector = RobberyDetector()
        self.fire_detector = FireDetector()
        self.reset()

    def reset(self):
        fall_detection.reset()
        self.car_detector.reset()
        self.robbery_detector.reset()
        self.fire_detector.reset()
        self.current_state = "NORMAL"
        self.active_event_type = None

    def update(self, landmarks, frame):
        # 1. Fall / Collapse
        fall_event, fall_status = fall_detection.update(landmarks, frame)

        # 2. Car Accident (context-aware: disabled when foreground human is present)
        car_event, car_status = self.car_detector.update(frame, landmarks)

        # 3. Robbery / Violence
        robbery_event, robbery_status = self.robbery_detector.update(landmarks, frame)

        # 4. Fire / Combustion (masks out human body so skin and clothes are never evaluated as flame)
        fire_event, fire_status = self.fire_detector.update(frame, landmarks)

        event = None
        state = "NORMAL"
        countdown = None

        # Determine priority: Emergency > Suspected > Normal
        if fire_status["state"] == "EMERGENCY":
            state = "FIRE_EMERGENCY"
            event = fire_event
        elif fall_status["state"] == "EMERGENCY":
            state = "FALL_EMERGENCY"
            event = fall_event
        elif car_status["state"] == "EMERGENCY":
            state = "ACCIDENT_EMERGENCY"
            event = car_event
        elif robbery_status["state"] == "EMERGENCY":
            state = "ROBBERY_EMERGENCY"
            event = robbery_event
        elif fire_status["state"] == "FIRE_SUSPECTED":
            state = "FIRE_SUSPECTED"
            countdown = fire_status["countdown"]
        elif fall_status["state"] == "FALL_SUSPECTED":
            state = "FALL_SUSPECTED"
            countdown = fall_status["countdown"]
        elif car_status["state"] == "ACCIDENT_SUSPECTED":
            state = "ACCIDENT_SUSPECTED"
            countdown = car_status["countdown"]
        elif robbery_status["state"] == "SUSPICIOUS_ACTIVITY":
            state = "SUSPICIOUS_ACTIVITY"
            countdown = robbery_status["countdown"]

        self.current_state = state

        return event, {
            "state": state,
            "countdown": countdown,
            "fall_state": fall_status["state"],
            "car_state": car_status["state"],
            "robbery_state": robbery_status["state"],
            "fire_state": fire_status["state"]
        }


detector_instance = UnifiedEmergencyDetector()


def reset():
    detector_instance.reset()


def update(landmarks, frame):
    return detector_instance.update(landmarks, frame)
