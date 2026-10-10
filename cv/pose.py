import os
import cv2
import mediapipe as mp


class LandmarksWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

use_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")

if use_legacy:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
else:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)
    mp_draw = None


def get_landmarks(frame):
    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if use_legacy:
        results = pose_detector.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = pose_detector.detect(mp_image)
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        return LandmarksWrapper(results.pose_landmarks[0])


def draw(frame, landmarks):
    if landmarks is None or frame is None:
        return frame

    if use_legacy and mp_draw:
        mp_draw.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        h, w = frame.shape[:2]
        pts = {}
        for idx, lm in enumerate(landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in pts and end_idx in pts:
                cv2.line(frame, pts[start_idx], pts[end_idx], (255, 255, 0), 2)

    return frame



if __name__ == "__main__":
    import os
    import sys

    # Import camera from the same folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import camera

    if len(sys.argv) > 1:
        source = f"videos/{sys.argv[1]}.mp4"
    else:
        source = 0

    cap = camera.open_source(source)
    cv2.namedWindow("VisionSOS - Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionSOS - Pose", 640, 480)
    frame_count = 0

    while True:
        frame = camera.read_frame(cap)
        if frame is None:
            break

        frame_count += 1
        landmarks = get_landmarks(frame)
        frame = draw(frame, landmarks)

        h, w = frame.shape[:2]
        if w > 640:
            new_height = int(h * (640 / w))
            frame = cv2.resize(frame, (640, new_height))

        cv2.imshow("VisionSOS - Pose", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        if frame_count % 30 == 0:
            if landmarks is not None:
                print(f"frame {frame_count} | landmarks: FOUND")
            else:
                print(f"frame {frame_count} | landmarks: NONE")

    camera.close(cap)
