import os
import sys
import time
import cv2
import numpy as np

try:
    from . import movement
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import movement


class RobberyDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "NORMAL"
        self.suspicious_frames = 0
        self.fired = False
        self.trigger_reason = None
        self.prev_points = None
        self.prev_wrists = None

    def check_surrender_posture(self, landmarks):
        if landmarks is None:
            return False

        try:
            nose = movement.get_point(landmarks, movement.PoseLandmark.NOSE)
            left_wrist = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_shoulder = movement.get_point(landmarks, movement.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_SHOULDER)
            left_elbow = movement.get_point(landmarks, movement.PoseLandmark.LEFT_ELBOW)
            right_elbow = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_ELBOW)

            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            nose_y = nose[1]

            # 1. Both wrists must be clearly above nose level
            hands_above_head = (left_wrist[1] < (nose_y - 0.04)) and (right_wrist[1] < (nose_y - 0.04))
            
            # 2. Elbows must also be raised up towards shoulder level (hands up posture)
            elbows_up = (left_elbow[1] < (shoulder_y + 0.08)) and (right_elbow[1] < (shoulder_y + 0.08))

            # 3. Hands must be spread apart (open surrender stance, NOT touching face / phone / chin)
            hands_spread = abs(left_wrist[0] - right_wrist[0]) > 0.22
            
            # 4. Hands must not be resting on face/chin
            not_touching_face = (abs(left_wrist[0] - nose[0]) > 0.08) and (abs(right_wrist[0] - nose[0]) > 0.08)

            torso_angle_val = movement.torso_angle(landmarks)

            if hands_above_head and elbows_up and hands_spread and not_touching_face and torso_angle_val < 40:
                return True
        except Exception:
            pass
        return False

    def check_aggressive_confrontation(self, landmarks, motion):
        if landmarks is None:
            return False

        try:
            left_wrist = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            current_wrists = np.array([left_wrist, right_wrist])

            wrist_speed = 0.0
            if self.prev_wrists is not None:
                wrist_speed = float(np.mean(np.linalg.norm(current_wrists - self.prev_wrists, axis=1)))
            self.prev_wrists = current_wrists

            # Violent physical struggle requires extreme erratic arm velocities (not subtle seated motion)
            if wrist_speed > 0.14 and motion > 0.08:
                return True
        except Exception:
            pass
        return False

    def update(self, landmarks, frame):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        surrender = False
        aggression = False
        motion = 0.0

        if landmarks is not None:
            surrender = self.check_surrender_posture(landmarks)
            motion, self.prev_points = movement.motion_energy(landmarks, self.prev_points)
            aggression = self.check_aggressive_confrontation(landmarks, motion)
        else:
            self.prev_points = None
            self.prev_wrists = None

        event = None

        if surrender:
            self.trigger_reason = "surrender_hands_up"
            self.suspicious_frames += 1
        elif aggression:
            self.trigger_reason = "violent_confrontation"
            self.suspicious_frames += 1
        else:
            # Decay quickly when normal posture resumes
            self.suspicious_frames = max(0, self.suspicious_frames - 2)

        # State Machine Logic (calibrated ~30 fps: 15 frames = ~0.5s alert, 35 frames = ~1.2s confirm)
        if self.state == "NORMAL":
            if self.suspicious_frames >= 15:
                self.state = "SUSPICIOUS_ACTIVITY"

        elif self.state == "SUSPICIOUS_ACTIVITY":
            if self.suspicious_frames >= 35:
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "robbery",
                        "confidence": 0.92,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": movement.torso_angle(landmarks) if landmarks else 0.0,
                        "motion_energy": motion,
                        "fall_velocity": 0.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "robbery_hostage_posture"
                    }
            elif self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        elif self.state == "EMERGENCY":
            if self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        return event, {
            "state": self.state,
            "countdown": max(0.0, 1.0 - (self.suspicious_frames / 35.0)) if self.state == "SUSPICIOUS_ACTIVITY" else None
        }
