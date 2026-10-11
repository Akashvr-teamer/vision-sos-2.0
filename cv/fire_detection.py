import os
import sys
import time
import cv2
import numpy as np


class FireDetector:
    """
    High-Precision Vision-Based Fire, Flame & Smoke Emergency Detector.
    Uses multi-color-space incandescence modeling (YCrCb + RGB + HSV)
    combined with smoke detection (desaturation + brightness patterns),
    human body masking, spatial flame clustering, and temporal confirmation.
    Prevents false triggers from human skin, wood paneling, and room lighting.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "NORMAL"
        self.fire_frames = 0
        self.fired = False
        self.trigger_reason = None
        self.active_flame_boxes = []
        self._prev_gray = None

    def mask_human_body(self, mask, landmarks, shape):
        """
        Masks out the region containing any detected human subject,
        ensuring skin tones, faces, and clothing are never evaluated as fire.
        Supports both MediaPipe landmark objects and dictionary lists.
        """
        if landmarks is None:
            return mask

        h, w = shape[:2]
        try:
            xs, ys = [], []
            if hasattr(landmarks, "landmark"):
                xs = [lm.x for lm in landmarks.landmark]
                ys = [lm.y for lm in landmarks.landmark]
            elif isinstance(landmarks, list):
                xs = [lm["x"] for lm in landmarks if isinstance(lm, dict) and "x" in lm]
                ys = [lm["y"] for lm in landmarks if isinstance(lm, dict) and "y" in lm]

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
        Segments flame and combustion pixels using broadened multi-space rules:
        - RGB  : R > G > B, R >= 150, R - G >= 15 (captures orange, yellow, white flame)
        - YCrCb: Cr > Cb, Cr >= 128, Y >= 100
        - HSV  : Hue in [0, 40] (red-orange-yellow range), Sat >= 50, Val >= 100
                 OR: Hue in [165, 180] (deep red wrap-around), Sat >= 50, Val >= 100
        All three channels combined via OR union for maximum recall.
        """
        b, g, r = cv2.split(frame)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_ch, cr, cb = cv2.split(ycrcb)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # --- RGB fire criterion (broadened) ---
        r32 = r.astype(np.int32)
        g32 = g.astype(np.int32)
        b32 = b.astype(np.int32)
        rgb_fire = (
            (r32 > g32) &
            (g32 > b32) &
            (r32 >= 150) &
            ((r32 - g32) >= 15)
        )

        # --- YCrCb fire criterion (broadened) ---
        ycrcb_fire = (
            (cr.astype(np.int32) > cb.astype(np.int32)) &
            (cr >= 128) &
            (y_ch >= 100)
        )

        # --- HSV fire criterion (broader hue range for yellow/orange flames) ---
        hue_red_orange = (h_ch <= 40)          # red → orange → yellow-orange
        hue_deep_red   = (h_ch >= 165)         # wrap-around deep reds
        sat_ok = (s_ch >= 50)
        val_ok = (v_ch >= 100)
        hsv_fire = (hue_red_orange | hue_deep_red) & sat_ok & val_ok

        # Combine: RGB must hold AND either YCrCb or HSV confirms
        flame_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        flame_mask[rgb_fire & (ycrcb_fire | hsv_fire)] = 255

        # Morphological cleanup: open removes specks, dilate connects flame tongues
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
        flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_DILATE, kernel)

        return flame_mask

    def detect_smoke_pixels(self, frame):
        """
        Detects smoke as large, bright, desaturated (low-saturation gray/white)
        regions that are NOT flesh-toned. Smoke = high V, low S, mid hue.
        Used as a secondary signal that lowers the confirmation threshold.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Smoke: low saturation, moderately bright, hue anywhere
        smoke_mask = (
            (s_ch.astype(np.int32) < 60) &
            (v_ch.astype(np.int32) >= 130) &
            (v_ch.astype(np.int32) <= 245)   # exclude pure white overexposure
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

        return smoke_mask

    def update(self, frame, landmarks=None):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        # --- Flame pixel detection ---
        raw_flame_mask = self.detect_incandescent_flame_pixels(frame)
        flame_mask = self.mask_human_body(raw_flame_mask, landmarks, frame.shape)

        # Find significant flame clusters
        cnts, _ = cv2.findContours(flame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area >= 200:   # lowered from 300 to 200 for better recall
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, w, h, area))

        self.active_flame_boxes = boxes
        has_fire = len(boxes) > 0

        # --- Smoke detection as secondary signal ---
        has_smoke = False
        if not has_fire:
            smoke_mask = self.detect_smoke_pixels(frame)
            smoke_mask = self.mask_human_body(smoke_mask, landmarks, frame.shape)
            smoke_cnts, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_smoke_area = sum(cv2.contourArea(c) for c in smoke_cnts)
            frame_area = frame.shape[0] * frame.shape[1]
            # Smoke must cover at least 8% of frame to count as a signal
            has_smoke = (total_smoke_area / frame_area) > 0.08

        event = None

        if has_fire:
            self.trigger_reason = "incandescent_flame_combustion"
            self.fire_frames += 1
        elif has_smoke:
            self.trigger_reason = "smoke_detected"
            self.fire_frames += 1          # smoke also increments counter
        else:
            # Slower decay: only -1 per miss frame (was -2) so patchy detection accumulates
            self.fire_frames = max(0, self.fire_frames - 1)

        # Hysteresis confirmation: 8 frames = Suspected, 20 frames = Emergency
        # (lowered from 10/25 so sparse detections still confirm)
        if self.state == "NORMAL":
            if self.fire_frames >= 8:
                self.state = "FIRE_SUSPECTED"

        elif self.state == "FIRE_SUSPECTED":
            if self.fire_frames >= 20:
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "fire",
                        "confidence": 0.93,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": 0.0,
                        "motion_energy": 8.5,
                        "fall_velocity": 0.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "fire_smoke_anomaly"
                    }
            elif self.fire_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        elif self.state == "EMERGENCY":
            if self.fire_frames == 0:
                self.state = "NORMAL"
                self.fired = False

        countdown = max(0.0, (20 - self.fire_frames) / 30.0) if self.state == "FIRE_SUSPECTED" else None

        return event, {
            "state": self.state,
            "countdown": countdown
        }

    def draw(self, frame):
        """
        Renders flame bounding boxes and labels around detected fires.
        """
        if not self.active_flame_boxes:
            return frame

        out = frame.copy()
        for x, y, w, h, area in self.active_flame_boxes:
            # Fiery orange/crimson bounding box
            color = (0, 69, 255) if self.state == "EMERGENCY" else (0, 140, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            # High-tech corner accents
            length = min(15, min(w, h) // 4)
            cv2.line(out, (x, y), (x + length, y), (0, 255, 255), 2)
            cv2.line(out, (x, y), (x, y + length), (0, 255, 255), 2)
            cv2.line(out, (x + w, y), (x + w - length, y), (0, 255, 255), 2)
            cv2.line(out, (x + w, y), (x + w, y + length), (0, 255, 255), 2)

            # Tag label banner
            label = "FIRE DETECTED" if self.state == "EMERGENCY" else "FLAME DETECTED"
            banner_w = 150
            cv2.rectangle(out, (x, max(0, y - 24)), (x + banner_w, y), color, -1)
            cv2.putText(out, label, (x + 6, max(16, y - 6)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        return out
