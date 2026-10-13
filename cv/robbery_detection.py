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
    """
    Real-World Robbery & Violent Threat Detector.

    Detects multiple real robbery patterns observed in CCTV footage:
    1. Classic surrender: both hands above head (spread or not)
    2. Single-arm raise: one hand forced up (partial surrender / being grabbed)
    3. Aggressive confrontation: high-speed erratic arm motion (struggle, pushing, hitting)
    4. Victim crouching under duress: sudden body collapse with arms reaching forward
    5. Sustained erratic body motion: violent struggle over multiple frames

    Much broader than a strict "movie hands-up" posture check.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "NORMAL"
        self.suspicious_frames = 0
        self.fired = False
        self.trigger_reason = None
        self.prev_points = None
        self.prev_wrists = None
        self.wrist_speed_history = []   # rolling window for smoothing
        self.motion_history = []        # rolling window for sustained motion

    # ------------------------------------------------------------------
    # POSTURE CHECKS
    # ------------------------------------------------------------------

    def check_surrender_posture(self, landmarks):
        """
        Classic surrender: at least ONE wrist clearly above the nose.
        (Real victims often only manage to raise one arm when grabbed.)
        Both wrists above nose is a stronger signal and scored higher.
        """
        if landmarks is None:
            return 0   # 0 = not detected, 1 = partial, 2 = strong

        try:
            nose          = movement.get_point(landmarks, movement.PoseLandmark.NOSE)
            left_wrist    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_shoulder = movement.get_point(landmarks, movement.PoseLandmark.LEFT_SHOULDER)
            right_shoulder= movement.get_point(landmarks, movement.PoseLandmark.RIGHT_SHOULDER)
            left_elbow    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_ELBOW)
            right_elbow   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_ELBOW)

            nose_y      = nose[1]
            shoulder_y  = (left_shoulder[1] + right_shoulder[1]) / 2.0

            lw_above_nose = left_wrist[1]  < (nose_y - 0.03)
            rw_above_nose = right_wrist[1] < (nose_y - 0.03)

            # Elbow raised above shoulder level
            le_up = left_elbow[1]  < (shoulder_y + 0.10)
            re_up = right_elbow[1] < (shoulder_y + 0.10)

            both_above = lw_above_nose and rw_above_nose
            one_above  = lw_above_nose or rw_above_nose

            if both_above and (le_up or re_up):
                return 2   # strong surrender signal
            if one_above:
                return 1   # partial raise (grabbed arm, being pushed against surface)
        except Exception:
            pass
        return 0

    def check_aggressive_confrontation(self, landmarks, wrist_speed, motion):
        """
        Violent physical struggle:
        - Fast arm movement (wrist speed) over several consecutive frames
        - Combined with elevated whole-body motion
        Thresholds are lower than before to catch real ATM / street attacks.
        """
        if landmarks is None:
            return False
        try:
            # Smooth wrist speed over a rolling window to reject one-frame jitter
            self.wrist_speed_history.append(wrist_speed)
            if len(self.wrist_speed_history) > 6:
                self.wrist_speed_history.pop(0)
            avg_wrist_speed = float(np.mean(self.wrist_speed_history))

            self.motion_history.append(motion)
            if len(self.motion_history) > 6:
                self.motion_history.pop(0)
            avg_motion = float(np.mean(self.motion_history))

            # Lowered thresholds: wrist speed 0.07 (was 0.14), motion 0.04 (was 0.08)
            if avg_wrist_speed > 0.07 and avg_motion > 0.04:
                return True
        except Exception:
            pass
        return False

    def check_forward_lean_grab(self, landmarks):
        """
        Victim being pushed forward / against a wall / ATM:
        Torso leaning significantly (>30°) while arms are forward and elevated.
        Common in ATM robbery footage where the attacker shoves the victim.
        """
        if landmarks is None:
            return False
        try:
            torso_ang = movement.torso_angle(landmarks)

            left_wrist  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_hip    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_HIP)
            right_hip   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_HIP)
            hip_y       = (left_hip[1] + right_hip[1]) / 2.0

            # Both wrists at or above hip level while body is leaning forward
            wrists_forward = (left_wrist[1] <= hip_y + 0.05) and (right_wrist[1] <= hip_y + 0.05)

            if torso_ang > 30 and wrists_forward:
                return True
        except Exception:
            pass
        return False

    def check_crouching_duress(self, landmarks):
        """
        Victim crouching / cowering under threat:
        Head-to-ankle gap compresses significantly (body collapses downward)
        while arms are still raised or extended (not a calm sit-down).
        """
        if landmarks is None:
            return False
        try:
            gap = movement.head_ankle_gap(landmarks)

            left_wrist  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_hip    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_HIP)
            right_hip   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_HIP)
            hip_y       = (left_hip[1] + right_hip[1]) / 2.0

            # Compressed vertical body span + at least one arm up near/above waist
            arms_extended = (left_wrist[1] < hip_y + 0.10) or (right_wrist[1] < hip_y + 0.10)

            if gap < 0.38 and arms_extended:
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # MAIN UPDATE
    # ------------------------------------------------------------------

    def update(self, landmarks, frame):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        motion     = 0.0
        wrist_speed = 0.0

        if landmarks is not None:
            motion, self.prev_points = movement.motion_energy(landmarks, self.prev_points)

            # Compute wrist speed independently
            try:
                lw = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
                rw = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
                current_wrists = np.array([lw, rw])
                if self.prev_wrists is not None:
                    wrist_speed = float(
                        np.mean(np.linalg.norm(current_wrists - self.prev_wrists, axis=1))
                    )
                self.prev_wrists = current_wrists
            except Exception:
                self.prev_wrists = None
        else:
            self.prev_points  = None
            self.prev_wrists  = None
            self.wrist_speed_history.clear()
            self.motion_history.clear()

        # Evaluate all posture signals
        surrender_score = self.check_surrender_posture(landmarks)         # 0 / 1 / 2
        aggression      = self.check_aggressive_confrontation(landmarks, wrist_speed, motion)
        lean_grab       = self.check_forward_lean_grab(landmarks)
        crouching       = self.check_crouching_duress(landmarks)

        # Score-based increment: stronger signals advance faster
        increment = 0
        if surrender_score == 2:
            increment = 3
            self.trigger_reason = "surrender_both_hands_up"
        elif surrender_score == 1:
            increment = 2
            self.trigger_reason = "surrender_one_hand_up"
        elif aggression:
            increment = 2
            self.trigger_reason = "violent_confrontation"
        elif lean_grab:
            increment = 2
            self.trigger_reason = "forward_grab_push"
        elif crouching:
            increment = 1
            self.trigger_reason = "victim_cowering_duress"

        event = None

        if increment > 0:
            self.suspicious_frames = min(self.suspicious_frames + increment, 60)
        else:
            # Slower decay so patchy skeleton detection doesn't reset progress
            self.suspicious_frames = max(0, self.suspicious_frames - 1)

        # State Machine (at ~25-30 fps):
        # SUSPICIOUS_ACTIVITY at 12 frames (~0.4s), EMERGENCY at 28 frames (~1s)
        if self.state == "NORMAL":
            if self.suspicious_frames >= 12:
                self.state = "SUSPICIOUS_ACTIVITY"

        elif self.state == "SUSPICIOUS_ACTIVITY":
            if self.suspicious_frames >= 28:
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "robbery",
                        "confidence": 0.91,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": movement.torso_angle(landmarks) if landmarks else 0.0,
                        "motion_energy": motion,
                        "fall_velocity": 0.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "robbery_threat_posture"
                    }
            elif self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        elif self.state == "EMERGENCY":
            if self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        countdown = (
            max(0.0, 1.0 - (self.suspicious_frames / 28.0))
            if self.state == "SUSPICIOUS_ACTIVITY"
            else None
        )

        return event, {
            "state": self.state,
            "countdown": countdown
        }
