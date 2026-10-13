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
    Supports either unified concurrent inference across all 4 pipelines,
    or isolated single-category execution to eliminate pipeline cross-clashes:
    1. Falls & Medical Collapses
    2. Vehicle Collisions & Traffic Accidents
    3. Robbery & Hostage / Violent Threat Postures
    4. Fire, Smoke & Thermal Combustion
    """

    def __init__(self):
        self.car_detector = CarAccidentDetector()
        self.robbery_detector = RobberyDetector()
        self.fire_detector = FireDetector()
        self.active_category = "Unified"
        self.reset()

    def set_active_category(self, category):
        self.active_category = category or "Unified"

    def reset(self):
        fall_detection.reset()
        self.car_detector.reset()
        self.robbery_detector.reset()
        self.fire_detector.reset()
        self.current_state = "NORMAL"
        self.active_event_type = None

    def update(self, landmarks, frame, category=None):
        if category is not None:
            self.active_category = category

        cat = (self.active_category or "Unified").strip().lower()
        is_fall = any(k in cat for k in ["fall", "collapse"])
        is_car = any(k in cat for k in ["car", "accident"])
        is_robbery = any(k in cat for k in ["robbery", "threat"])
        is_fire = any(k in cat for k in ["fire", "smoke", "combustion"])
        is_unified = (not is_fall and not is_car and not is_robbery and not is_fire) or cat == "unified"

        default_status = {"state": "NORMAL", "countdown": None}
        fall_event, fall_status = None, dict(default_status)
        car_event, car_status = None, dict(default_status)
        robbery_event, robbery_status = None, dict(default_status)
        fire_event, fire_status = None, dict(default_status)

        # 1. Fall / Collapse (runs only if Unified or Fall selected)
        if is_unified or is_fall:
            fall_event, fall_status = fall_detection.update(landmarks, frame)

        # 2. Car Accident (runs only if Unified or Car Accident selected)
        if is_unified or is_car:
            car_event, car_status = self.car_detector.update(frame, landmarks)

        # 3. Robbery / Violence (runs only if Unified or Robbery selected)
        if is_unified or is_robbery:
            robbery_event, robbery_status = self.robbery_detector.update(landmarks, frame)

        # 4. Fire / Combustion (runs only if Unified or Fire selected)
        if is_unified or is_fire:
            fire_event, fire_status = self.fire_detector.update(frame, landmarks)

        event = None
        state = "NORMAL"
        countdown = None

        # Determine priority based on active category
        if fire_status["state"] == "EMERGENCY":
            state = "FIRE_EMERGENCY"
            event = fire_event
        elif car_status["state"] == "EMERGENCY":
            state = "ACCIDENT_EMERGENCY"
            event = car_event
        elif fall_status["state"] == "EMERGENCY":
            state = "FALL_EMERGENCY"
            event = fall_event
        elif robbery_status["state"] == "EMERGENCY":
            state = "ROBBERY_EMERGENCY"
            event = robbery_event
        elif car_status["state"] == "ACCIDENT_SUSPECTED":
            state = "ACCIDENT_SUSPECTED"
            countdown = car_status["countdown"]
        elif fire_status["state"] == "FIRE_SUSPECTED":
            state = "FIRE_SUSPECTED"
            countdown = fire_status["countdown"]
        elif fall_status["state"] == "FALL_SUSPECTED":
            state = "FALL_SUSPECTED"
            countdown = fall_status["countdown"]
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
            "fire_state": fire_status["state"],
            "active_category": self.active_category,
            # Full robbery sub-status so callers can read confirmed signal labels
            "robbery_status": robbery_status,
        }


detector_instance = UnifiedEmergencyDetector()
car_detector = detector_instance.car_detector
fire_detector = detector_instance.fire_detector


def reset():
    detector_instance.reset()


def set_active_category(category):
    detector_instance.set_active_category(category)


def update(landmarks, frame, category=None):
    return detector_instance.update(landmarks, frame, category=category)


def draw(frame):
    """Draws active vehicle and fire detections on frame."""
    frame = detector_instance.car_detector.draw(frame)
    frame = detector_instance.fire_detector.draw(frame)
    return frame

