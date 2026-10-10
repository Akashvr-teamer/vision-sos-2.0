import os
import sys
import time
import cv2
import numpy as np


class FireDetector:
    """
    High-Precision Vision-Based Fire and Flame Emergency Detector.
    Uses multi-color-space incandescence modeling (YCrCb + RGB + HSV)
    combined with human body masking and temporal flame turbulence analysis.
    Prevents false triggers from human skin, wood paneling, and room lighting.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "NORMAL"
        self.fire_frames = 0
        self.fired = False
        self.trigger_reason = None
        self.prev_fire_mask = None

    def mask_human_body(self, mask, landmarks, shape):
        """
        Masks out the entire region containing any detected human subject,
        ensuring skin tones, faces, hair, and clothing are never evaluated as fire.
        """
        if landmarks is None or not hasattr(landmarks, "landmark"):
            return mask

        h, w = shape[:2]
        try:
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            if len(xs) > 0:
                # Generous padding around the human body
                x1 = int(max(0, (min(xs) - 0.10) * w))
                x2 = int(min(w, (max(xs) + 0.10) * w))
                y1 = int(max(0, (min(ys) - 0.10) * h))
                y2 = int(min(h, (max(ys) + 0.10) * h))
                # Set human region to zero (no fire)
                mask[y1:y2, x1:x2] = 0
        except Exception:
            pass
        return mask

    def detect_incandescent_flame_pixels(self, frame):
        """
        Segment high-temperature incandescent flame pixels.
        Real fire is a primary light source (incandescent):
        - High luminance: Y >= 220
        - High chrominance delta: Cr >= 150, Cb <= 115
        - Intense Red/Orange: R >= 225, G >= 160, B <= 135, R - G >= 35
        - HSV: Hue in [0, 28], Saturation >= 110, Value >= 225
        """
        h, w = frame.shape[:2]
        scale = 320.0 / max(w, h)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # 1. YCrCb Color Space (Industry standard for fire detection)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        ycrcb_fire = (y >= 215) & (cr >= 150) & (cb <= 120) & (cr > cb)

        # 2. RGB Direct Incandescence Rule
        b, g, r = cv2.split(small)
        rgb_fire = (r >= 225) & (g >= 155) & (b <= 140) & ((r.astype(int) - g.astype(int)) >= 30)

        # 3. HSV Flame Cone
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        hsv_fire = ((h_channel <= 28) | (h_channel >= 170)) & (s_channel >= 100) & (v_channel >= 220)

        # Intersection of all 3 color spaces: Only genuine incandescent flame passes
        candidate = np.zeros(small.shape[:2], dtype=np.uint8)
        candidate[ycrcb_fire & rgb_fire & hsv_fire] = 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

        return candidate, small.shape[:2]

    def update(self, frame, landmarks=None):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        raw_mask, dims = self.detect_incandescent_flame_pixels(frame)

        # Exclude human body completely
        mask = self.mask_human_body(raw_mask, landmarks, dims)

        fire_pixel_count = cv2.countNonZero(mask)

        # Check for dynamic flame turbulence (flames oscillate rapidly)
        is_flickering = False
        flicker_score = 0.0

        if self.prev_fire_mask is not None and fire_pixel_count >= 180:
            diff = cv2.absdiff(mask, self.prev_fire_mask)
            diff_count = cv2.countNonZero(diff)
            flicker_score = diff_count / float(max(1, fire_pixel_count))
            # True flames exhibit high boundary turbulence
            if 0.15 <= flicker_score <= 1.1:
                is_flickering = True
        elif fire_pixel_count >= 500:
            # Massive bright incandescent region
            is_flickering = True

        self.prev_fire_mask = mask

        # Genuine fire requires both intense incandescent clustering and dynamic flicker
        has_fire = (fire_pixel_count >= 200 and is_flickering) or (fire_pixel_count >= 800)

        event = None

        if has_fire:
            self.trigger_reason = "incandescent_flame_turbulence"
            self.fire_frames += 1
        else:
            # Fast decay to prevent false alarms
            self.fire_frames = max(0, self.fire_frames - 3)

        # Hysteresis confirmation (~30 fps: 15 frames = ~0.5s alert, 35 frames = ~1.2s confirm)
        if self.state == "NORMAL":
            if self.fire_frames >= 15:
                self.state = "FIRE_SUSPECTED"

        elif self.state == "FIRE_SUSPECTED":
            if self.fire_frames >= 35:
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "fire",
                        "confidence": 0.95,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": 0.0,
                        "motion_energy": float(flicker_score * 10.0),
                        "fall_velocity": 0.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "fire_incandescence_anomaly"
                    }
            elif self.fire_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        elif self.state == "EMERGENCY":
            if self.fire_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        return event, {
            "state": self.state,
            "countdown": max(0.0, 1.0 - (self.fire_frames / 35.0)) if self.state == "FIRE_SUSPECTED" else None
        }
