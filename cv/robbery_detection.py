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


# Minimum consecutive frames a raw signal must hold before it is "confirmed"
SIGNAL_CONFIRM_FRAMES = 8   # frames a signal must hold before being confirmed (~0.27s at 30fps)


class RobberyDetector:
    """
    Real-World Robbery & Violent Threat Detector.

    Each physical signal (arm raise, violent wrist motion, forward grab, etc.)
    must be sustained for SIGNAL_CONFIRM_FRAMES consecutive frames before it is
    treated as a confirmed cause. Only confirmed causes increment the suspicion
    counter. This prevents single-frame pose noise from triggering an alert.

    Confirmed signals are stored in self.active_signals so the UI / logs can
    display exactly *what movements* caused the escalation.
    """

    # Human-readable label for each signal key
    SIGNAL_LABELS = {
        "surrender_both_hands_up":  "Both hands raised above head",
        "surrender_one_hand_up":    "Single arm raised under duress",
        "violent_confrontation":    "Violent arm struggle detected",
        "forward_grab_push":        "Victim pushed / grabbed forward",
        "victim_cowering_duress":   "Victim crouching under threat",
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.state              = "NORMAL"
        self.suspicious_frames  = 0
        self.fired              = False
        self.trigger_reason     = None

        self.prev_points        = None
        self.prev_wrists        = None
        self.wrist_speed_history = []
        self.motion_history      = []

        # Per-signal raw frame counters (how many consecutive frames signal seen)
        self._raw_counts = {k: 0 for k in self.SIGNAL_LABELS}

        # Confirmed signals that are currently active (survived SIGNAL_CONFIRM_FRAMES)
        self.active_signals: dict[str, int] = {}   # key -> sustained confirmed frames

    # ------------------------------------------------------------------
    # RAW SIGNAL DETECTORS  (return True/False each frame)
    # ------------------------------------------------------------------

    def _raw_surrender_both(self, landmarks):
        """Both wrists clearly above nose, at least one elbow raised."""
        try:
            nose           = movement.get_point(landmarks, movement.PoseLandmark.NOSE)
            left_wrist     = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist    = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_shoulder  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_SHOULDER)
            left_elbow     = movement.get_point(landmarks, movement.PoseLandmark.LEFT_ELBOW)
            right_elbow    = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_ELBOW)

            nose_y     = nose[1]
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

            both_above = (left_wrist[1] < nose_y - 0.03) and (right_wrist[1] < nose_y - 0.03)
            elbow_up   = (left_elbow[1] < shoulder_y + 0.10) or (right_elbow[1] < shoulder_y + 0.10)

            return both_above and elbow_up
        except Exception:
            return False

    def _raw_surrender_one(self, landmarks):
        """At least one wrist above nose (single-arm raise under duress)."""
        try:
            nose        = movement.get_point(landmarks, movement.PoseLandmark.NOSE)
            left_wrist  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            nose_y      = nose[1]

            one_above = (left_wrist[1] < nose_y - 0.03) or (right_wrist[1] < nose_y - 0.03)
            both_above = (left_wrist[1] < nose_y - 0.03) and (right_wrist[1] < nose_y - 0.03)

            # Only count as "one arm" if it's NOT already triggering "both arms"
            return one_above and not both_above
        except Exception:
            return False

    def _raw_violent_confrontation(self, wrist_speed, motion):
        """Sustained fast arm + body motion — averaged over a rolling window."""
        try:
            self.wrist_speed_history.append(wrist_speed)
            if len(self.wrist_speed_history) > 6:
                self.wrist_speed_history.pop(0)

            self.motion_history.append(motion)
            if len(self.motion_history) > 6:
                self.motion_history.pop(0)

            avg_ws  = float(np.mean(self.wrist_speed_history))
            avg_mot = float(np.mean(self.motion_history))

            return avg_ws > 0.07 and avg_mot > 0.04
        except Exception:
            return False

    def _raw_forward_grab(self, landmarks):
        """Torso leaning forward >30° while arms are at or above waist."""
        try:
            torso_ang  = movement.torso_angle(landmarks)
            left_wrist  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_hip    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_HIP)
            right_hip   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_HIP)
            hip_y       = (left_hip[1] + right_hip[1]) / 2.0

            wrists_at_waist = (left_wrist[1] <= hip_y + 0.05) and (right_wrist[1] <= hip_y + 0.05)
            return torso_ang > 30 and wrists_at_waist
        except Exception:
            return False

    def _raw_crouching(self, landmarks):
        """Compressed vertical body span + at least one arm extended."""
        try:
            gap         = movement.head_ankle_gap(landmarks)
            left_wrist  = movement.get_point(landmarks, movement.PoseLandmark.LEFT_WRIST)
            right_wrist = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_WRIST)
            left_hip    = movement.get_point(landmarks, movement.PoseLandmark.LEFT_HIP)
            right_hip   = movement.get_point(landmarks, movement.PoseLandmark.RIGHT_HIP)
            hip_y       = (left_hip[1] + right_hip[1]) / 2.0

            arms_extended = (left_wrist[1] < hip_y + 0.10) or (right_wrist[1] < hip_y + 0.10)
            return gap < 0.38 and arms_extended
        except Exception:
            return False

    # ------------------------------------------------------------------
    # SIGNAL CONFIRMATION ENGINE
    # ------------------------------------------------------------------

    def _update_signals(self, raw_detections: dict) -> dict:
        """
        For each signal:
          - If raw detection is True  → increment raw counter (cap at SIGNAL_CONFIRM_FRAMES)
          - If raw detection is False → decay raw counter by 1 (min 0)
          - Signal is CONFIRMED when raw counter >= SIGNAL_CONFIRM_FRAMES
          - Confirmed signals accumulate in active_signals (key → sustained frames)
          - Signals that drop out of confirmed state are removed from active_signals
        Returns dict of currently confirmed signals.
        """
        confirmed = {}

        for key, detected in raw_detections.items():
            if detected:
                self._raw_counts[key] = min(self._raw_counts[key] + 1, SIGNAL_CONFIRM_FRAMES * 3)
            else:
                self._raw_counts[key] = max(self._raw_counts[key] - 1, 0)

            if self._raw_counts[key] >= SIGNAL_CONFIRM_FRAMES:
                # Signal is confirmed — count how long it has been sustained
                prev = self.active_signals.get(key, 0)
                confirmed[key] = prev + 1
            # Else: not confirmed, will be removed from active_signals

        self.active_signals = confirmed
        return confirmed

    # ------------------------------------------------------------------
    # MAIN UPDATE
    # ------------------------------------------------------------------

    def update(self, landmarks, frame):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        motion      = 0.0
        wrist_speed = 0.0

        if landmarks is not None:
            motion, self.prev_points = movement.motion_energy(landmarks, self.prev_points)

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
            self.prev_points = None
            self.prev_wrists = None
            self.wrist_speed_history.clear()
            self.motion_history.clear()

        # --- Evaluate raw signals this frame ---
        raw = {
            "surrender_both_hands_up": self._raw_surrender_both(landmarks),
            "surrender_one_hand_up":   self._raw_surrender_one(landmarks),
            "violent_confrontation":   self._raw_violent_confrontation(wrist_speed, motion),
            "forward_grab_push":       self._raw_forward_grab(landmarks),
            "victim_cowering_duress":  self._raw_crouching(landmarks),
        }

        # --- Confirm signals (must sustain SIGNAL_CONFIRM_FRAMES frames) ---
        confirmed = self._update_signals(raw)

        # --- Score: each distinct confirmed cause contributes ---
        # Stronger signals score more per frame they remain confirmed
        WEIGHTS = {
            "surrender_both_hands_up": 3,
            "surrender_one_hand_up":   2,
            "violent_confrontation":   2,
            "forward_grab_push":       2,
            "victim_cowering_duress":  1,
        }

        increment = sum(WEIGHTS[k] for k in confirmed)

        event = None

        if increment > 0:
            self.suspicious_frames = min(self.suspicious_frames + increment, 120)
            # Set trigger reason to ALL active confirmed signals
            self.trigger_reason = " + ".join(
                self.SIGNAL_LABELS[k] for k in confirmed
            )
        else:
            # Slow decay — one lost frame doesn't undo confirmed evidence
            self.suspicious_frames = max(0, self.suspicious_frames - 1)

        # --- State Machine ---
        # SUSPICIOUS_ACTIVITY at 22 pts (~0.7s of one strong confirmed signal)
        # EMERGENCY at 55 pts  (~1.8s of one strong or ~0.9s of two stacked signals)
        if self.state == "NORMAL":
            if self.suspicious_frames >= 22:
                self.state = "SUSPICIOUS_ACTIVITY"

        elif self.state == "SUSPICIOUS_ACTIVITY":
            if self.suspicious_frames >= 55:
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
                        "trigger": self.trigger_reason or "robbery_threat_posture",
                        "confirmed_signals": list(confirmed.keys()),
                    }
            elif self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        elif self.state == "EMERGENCY":
            if self.suspicious_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        countdown = (
            max(0.0, 1.0 - (self.suspicious_frames / 55.0))
            if self.state == "SUSPICIOUS_ACTIVITY"
            else None
        )

        return event, {
            "state": self.state,
            "countdown": countdown,
            "active_signals": {k: self.SIGNAL_LABELS[k] for k in confirmed},
            "trigger_reason": self.trigger_reason,
        }
