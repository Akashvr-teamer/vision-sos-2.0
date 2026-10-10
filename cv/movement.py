import mediapipe as mp
import numpy as np

try:
    PoseLandmark = mp.solutions.pose.PoseLandmark
except (AttributeError, Exception):
    from enum import IntEnum
    class PoseLandmark(IntEnum):
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32



# Extracts the normalized [x, y] coordinates of a given landmark name or enum.
def get_point(landmarks, name):
    if isinstance(name, str):
        idx = PoseLandmark[name].value
    elif hasattr(name, "value"):
        idx = name.value
    else:
        idx = int(name)
    landmark = landmarks.landmark[idx]
    return np.array([landmark.x, landmark.y], dtype=float)


# Computes the torso angle in degrees away from vertical (0=standing, 90=flat).
def torso_angle(landmarks):
    left_shoulder = get_point(landmarks, PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_point(landmarks, PoseLandmark.RIGHT_SHOULDER)
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    left_hip = get_point(landmarks, PoseLandmark.LEFT_HIP)
    right_hip = get_point(landmarks, PoseLandmark.RIGHT_HIP)
    hip_mid = (left_hip + right_hip) / 2.0

    dx = abs(shoulder_mid[0] - hip_mid[0])
    dy = abs(shoulder_mid[1] - hip_mid[1])
    angle = np.degrees(np.arctan2(dx, dy))
    return float(abs(angle))


# Computes the absolute vertical distance between the nose and the ankle midpoint.
def head_ankle_gap(landmarks):
    nose = get_point(landmarks, PoseLandmark.NOSE)
    left_ankle = get_point(landmarks, PoseLandmark.LEFT_ANKLE)
    right_ankle = get_point(landmarks, PoseLandmark.RIGHT_ANKLE)
    ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2.0
    return float(abs(nose[1] - ankle_avg_y))


# Computes the mean Euclidean displacement of all 33 landmarks across frames.
def motion_energy(landmarks, prev_points):
    current_points = np.array(
        [[lm.x, lm.y] for lm in landmarks.landmark], dtype=float
    )
    if prev_points is None:
        return (0.0, current_points)
    distances = np.linalg.norm(current_points - prev_points, axis=1)
    return (float(np.mean(distances)), current_points)


if __name__ == "__main__":
    import os
    import sys
    import cv2

    # Import camera and pose from the same folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import camera
    import pose

    if len(sys.argv) > 1:
        source = f"videos/{sys.argv[1]}.mp4"
    else:
        source = 0

    cap = camera.open_source(source)
    cv2.namedWindow("VisionSOS - Movement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionSOS - Movement", 640, 480)
    prev_points = None
    frame_count = 0

    while True:
        frame = camera.read_frame(cap)
        if frame is None:
            break

        frame_count += 1
        landmarks = pose.get_landmarks(frame)
        if landmarks is None:
            print(f"frame {frame_count} | NO LANDMARKS")
            continue

        angle = torso_angle(landmarks)
        gap = head_ankle_gap(landmarks)
        motion, prev_points = motion_energy(landmarks, prev_points)

        print(
            f"frame {frame_count} | angle: {angle:5.1f} | gap: {gap:.3f} | motion: {motion:.4f}"
        )

        frame = pose.draw(frame, landmarks)

        h, w = frame.shape[:2]
        if w > 640:
            new_height = int(h * (640 / w))
            frame = cv2.resize(frame, (640, new_height))

        cv2.imshow("VisionSOS - Movement", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    camera.close(cap)
