import os
import sys
import time
import cv2
import numpy as np


class CarAccidentDetector:
    """
    Kinetic Collision & Car Accident Detector.
    Strictly context-aware:
    - If a human subject is present in the foreground (landmarks active),
      vehicular collision detection is disabled (prevents human movement from triggering car accidents).
    - On vehicular/traffic video feeds, detects extreme kinetic optical flow surges,
      multi-blob collision vectors, and sudden impact deceleration.
    """

    def __init__(self):
        self.reset()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self.prev_gray = None

    def reset(self):
        self.state = "NORMAL"
        self.motion_history = []
        self.impact_frames = 0
        self.sustained_crash_frames = 0
        self.fired = False
        self.trigger_reason = None
        self.prev_gray = None

    def update(self, frame, landmarks=None):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        # -----------------------------------------------------------
        # CONTEXT-AWARE FILTER:
        # If a person is present in the frame (MediaPipe pose detected),
        # this is an indoor/pedestrian scene. A human shifting or leaning
        # in front of a camera can NEVER be a vehicular accident.
        # -----------------------------------------------------------
        if landmarks is not None and hasattr(landmarks, "landmark") and len(landmarks.landmark) > 0:
            self.reset()
            return None, {"state": "NORMAL", "countdown": None}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB if len(frame.shape) == 3 else frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize for fast, robust processing
        h, w = gray.shape[:2]
        scale = 320.0 / max(w, h)
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        motion_energy = 0.0
        impact_detected = False

        if self.prev_gray is not None and self.prev_gray.shape == small_gray.shape:
            # Dense Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy = float(np.mean(magnitude))
            max_motion = float(np.max(magnitude))
            motion_std = float(np.std(magnitude))

            # Foreground moving vehicle blobs
            fg_mask = self.bg_subtractor.apply(small_gray)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            large_blobs = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 900:
                    large_blobs += 1

            self.motion_history.append(motion_energy)
            if len(self.motion_history) > 30:
                self.motion_history.pop(0)

            avg_recent_motion = np.mean(self.motion_history[:-1]) if len(self.motion_history) > 1 else 1.0

            # Massive catastrophic collision kinematics on traffic feed:
            # 1. Sudden surge > 4.5x normal baseline with peak velocity > 55.0
            # 2. Multiple colliding vehicle blobs with chaotic directional variance
            if (motion_energy > (avg_recent_motion * 4.5 + 4.5) and max_motion > 55.0):
                impact_detected = True
                self.trigger_reason = "high_kinetic_impact"
            elif large_blobs >= 2 and motion_std > 6.5 and motion_energy > 5.5:
                impact_detected = True
                self.trigger_reason = "vehicle_collision_vector"

        self.prev_gray = small_gray

        event = None

        # State Machine Logic
        if self.state == "NORMAL":
            if impact_detected:
                self.impact_frames += 1
                if self.impact_frames >= 4:
                    self.state = "ACCIDENT_SUSPECTED"
                    self.sustained_crash_frames = 0
            else:
                self.impact_frames = max(0, self.impact_frames - 1)

        elif self.state == "ACCIDENT_SUSPECTED":
            self.sustained_crash_frames += 1

            if self.sustained_crash_frames >= 18: # ~0.7s confirmation
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "car_accident",
                        "confidence": 0.90,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": 0.0,
                        "motion_energy": motion_energy,
                        "fall_velocity": motion_energy * 10.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "sudden_impact_deceleration"
                    }
            elif not impact_detected and self.sustained_crash_frames > 28:
                self.state = "NORMAL"
                self.impact_frames = 0
                self.fired = False

        elif self.state == "EMERGENCY":
            if not impact_detected:
                self.impact_frames = max(0, self.impact_frames - 1)
                if self.impact_frames == 0:
                    self.state = "NORMAL"
                    self.fired = False

        return event, {
            "state": self.state,
            "countdown": max(0.0, 1.0 - (self.sustained_crash_frames / 28.0)) if self.state == "ACCIDENT_SUSPECTED" else None
        }
