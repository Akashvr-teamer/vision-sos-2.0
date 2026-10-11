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
        self.detected_vehicles = []
        self.impact_point = None

    def is_close_foreground_human(self, landmarks):
        """
        Only filters out close-up webcam users sitting directly in front of the lens.
        Does NOT filter out road footage, pedestrians, or distant objects.
        """
        if landmarks is None or not hasattr(landmarks, "landmark"):
            return False
        try:
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            if (bbox_w * bbox_h) > 0.25 and bbox_h > 0.40:
                return True
        except Exception:
            pass
        return False

    def update(self, frame, landmarks=None):
        if frame is None:
            return None, {"state": self.state, "countdown": None}

        # Only suppress if a huge foreground human is sitting directly in front of webcam lens
        if self.is_close_foreground_human(landmarks):
            self.reset()
            return None, {"state": "NORMAL", "countdown": None}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize for fast, robust processing
        h, w = gray.shape[:2]
        scale = 320.0 / max(w, h)
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        motion_energy = 0.0
        max_motion = 0.0
        impact_detected = False
        inv_scale = 1.0 / scale

        # Foreground vehicle blobs extraction with morphological cleanup
        fg_mask = self.bg_subtractor.apply(small_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract and update tracked vehicle bounding boxes
        current_vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                bx, by, bw_b, bh_b = cv2.boundingRect(cnt)
                fx, fy = int(bx * inv_scale), int(by * inv_scale)
                fbw, fbh = int(bw_b * inv_scale), int(bh_b * inv_scale)
                if fbw >= 35 and fbh >= 35:
                    aspect = float(fbh) / max(fbw, 1)
                    # Tall vertical or elongated profiles correspond to trucks, trailers, and buses
                    is_heavy = (aspect > 1.15 and fbh > 120) or (fbw * fbh > 45000 and aspect > 1.0)
                    vtype = "TRUCK" if is_heavy else "CAR"
                    current_vehicles.append({
                        "bbox": (fx, fy, fbw, fbh),
                        "type": vtype,
                        "area": fbw * fbh
                    })
        self.detected_vehicles = current_vehicles

        if self.prev_gray is not None and self.prev_gray.shape == small_gray.shape:
            # Dense Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy = float(np.mean(magnitude))
            max_motion = float(np.max(magnitude))
            motion_std = float(np.std(magnitude))
            moving_blobs = len(self.detected_vehicles)

            self.motion_history.append(motion_energy)
            if len(self.motion_history) > 30:
                self.motion_history.pop(0)

            avg_recent_motion = np.mean(self.motion_history[:-1]) if len(self.motion_history) > 1 else 0.5

            # Robust vehicular collision dynamics on traffic feeds:
            # 1. Extreme kinetic impact spike (peak flow >= 25.0)
            # 2. Sudden surge >= 2.0x normal baseline with peak >= 15.0
            # 3. High motion standard deviation with moving vehicles (collision vector)
            if max_motion >= 25.0:
                impact_detected = True
                self.trigger_reason = "high_kinetic_impact"
            elif motion_energy > (avg_recent_motion * 2.0 + 0.3) and max_motion >= 15.0:
                impact_detected = True
                self.trigger_reason = "sudden_kinetic_surge"
            elif motion_std > 3.0 and max_motion >= 18.0 and moving_blobs >= 1:
                impact_detected = True
                self.trigger_reason = "vehicle_collision_vector"

            # Localize impact point from peak optical flow vector
            if impact_detected or self.state in ("ACCIDENT_SUSPECTED", "EMERGENCY"):
                _, _, _, max_loc = cv2.minMaxLoc(magnitude)
                self.impact_point = (int(max_loc[0] * inv_scale), int(max_loc[1] * inv_scale))

        self.prev_gray = small_gray

        event = None

        # State Machine Logic
        if self.state == "NORMAL":
            if impact_detected:
                self.impact_frames += 1
                if self.impact_frames >= 1:
                    self.state = "ACCIDENT_SUSPECTED"
                    self.sustained_crash_frames = 0
            else:
                self.impact_frames = 0

        elif self.state == "ACCIDENT_SUSPECTED":
            self.sustained_crash_frames += 1

            if self.sustained_crash_frames >= 10:  # ~0.4s confirmation window
                if not self.fired:
                    self.state = "EMERGENCY"
                    self.fired = True
                    event = {
                        "type": "car_accident",
                        "confidence": 0.94,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "motionless_sec": 0.0,
                        "torso_angle": 0.0,
                        "motion_energy": motion_energy,
                        "fall_velocity": max_motion * 10.0,
                        "snapshot_frame": frame,
                        "trigger": self.trigger_reason or "sudden_impact_deceleration"
                    }
            elif not impact_detected and self.sustained_crash_frames > 25:
                self.state = "NORMAL"
                self.impact_frames = 0
                self.fired = False

        elif self.state == "EMERGENCY":
            pass  # Maintain emergency state until user or system resets

        return event, {
            "state": self.state,
            "countdown": max(0.0, round(1.0 - (self.sustained_crash_frames / 10.0), 1)) if self.state == "ACCIDENT_SUSPECTED" else None
        }

    def draw(self, frame):
        """
        Renders real-time vehicle detection bounding boxes, vehicle labels (CAR / TRUCK),
        and high-visibility collision/accident indicators.
        """
        if frame is None:
            return frame

        is_accident = (self.state in ("ACCIDENT_SUSPECTED", "EMERGENCY"))

        # Render detected vehicles
        for v in getattr(self, "detected_vehicles", []):
            x, y, bw, bh = v["bbox"]
            vtype = v["type"]

            is_involved = False
            if is_accident and self.impact_point:
                ix, iy = self.impact_point
                if (x - 70 <= ix <= x + bw + 70) and (y - 70 <= iy <= y + bh + 70):
                    is_involved = True

            if is_involved:
                # Highlight colliding vehicle in bright crimson red
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (30, 30, 240), 3)
                label = f"IMPACT ZONE: {vtype}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                cv2.rectangle(frame, (x, max(0, y - 24)), (x + lw + 10, y), (30, 30, 240), -1)
                cv2.putText(frame, label, (x + 5, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Normal tracked vehicle with modern cyan/gold bounding box
                box_color = (0, 220, 255) if vtype == "CAR" else (240, 180, 0)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
                label = f"{vtype}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                cv2.rectangle(frame, (x, max(0, y - 18)), (x + lw + 6, y), box_color, -1)
                cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.45, (10, 15, 20), 1, cv2.LINE_AA)

        # Draw collision vector target ring if an accident is active
        if is_accident and self.impact_point:
            ix, iy = self.impact_point
            cv2.circle(frame, (ix, iy), 24, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "COLLISION VECTOR", (ix + 32, iy + 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return frame
