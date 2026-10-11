import time
import unittest

try:
    import numpy as np
    import cv2
    from cv import fall_detection
    from cv.pose import LandmarkItem, LandmarksWrapper
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


def make_landmarks(torso_angle_deg=0.0, gap=0.6):
    """
    Synthesizes a 33-landmark pose with specified torso angle and head-ankle gap.
    0 deg = standing vertical, 90 deg = horizontal lying down.
    """
    rad = np.radians(torso_angle_deg)
    # Hip at (0.5, 0.6)
    hip_x, hip_y = 0.5, 0.6
    # Shoulder relative to hip
    shoulder_len = 0.25
    shoulder_x = hip_x + shoulder_len * np.sin(rad)
    shoulder_y = hip_y - shoulder_len * np.cos(rad)

    # Nose near shoulder
    nose_y = shoulder_y - 0.08
    # Ankle near hip - gap
    ankle_y = nose_y + gap

    landmarks = []
    for i in range(33):
        landmarks.append(LandmarkItem(x=0.5, y=0.5, z=0.0, visibility=1.0))

    # PoseLandmark.NOSE = 0
    landmarks[0] = LandmarkItem(x=0.5, y=nose_y)
    # PoseLandmark.LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12
    landmarks[11] = LandmarkItem(x=shoulder_x - 0.05, y=shoulder_y)
    landmarks[12] = LandmarkItem(x=shoulder_x + 0.05, y=shoulder_y)
    # PoseLandmark.LEFT_HIP = 23, RIGHT_HIP = 24
    landmarks[23] = LandmarkItem(x=hip_x - 0.05, y=hip_y)
    landmarks[24] = LandmarkItem(x=hip_x + 0.05, y=hip_y)
    # PoseLandmark.LEFT_ANKLE = 27, RIGHT_ANKLE = 28
    landmarks[27] = LandmarkItem(x=0.45, y=ankle_y)
    landmarks[28] = LandmarkItem(x=0.55, y=ankle_y)

    return LandmarksWrapper(landmarks)


class TestFallResilience(unittest.TestCase):
    def setUp(self):
        if not CV_AVAILABLE:
            raise unittest.SkipTest("OpenCV/NumPy not installed in current interpreter")
        fall_detection.reset()

    def test_standing_is_normal(self):
        standing = make_landmarks(torso_angle_deg=5.0, gap=0.7)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event, status = fall_detection.update(standing, frame)
        self.assertEqual(status["state"], "NORMAL")
        self.assertIsNone(status["countdown"])
        self.assertIsNone(event)

    def test_fall_and_dropout_resilience(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Start standing
        standing = make_landmarks(torso_angle_deg=5.0, gap=0.7)
        fall_detection.update(standing, frame)

        # 2. Sudden fall to 75 degrees
        fallen = make_landmarks(torso_angle_deg=75.0, gap=0.15)
        event, status = fall_detection.update(fallen, frame)
        self.assertEqual(status["state"], "FALL_SUSPECTED")
        self.assertIsNotNone(status["countdown"])

        # 3. Simulate landmark dropout (landmarks = None)
        # Verify countdown DOES NOT disappear to None!
        event, status = fall_detection.update(None, frame)
        self.assertEqual(status["state"], "FALL_SUSPECTED")
        self.assertIsNotNone(status["countdown"], "Countdown must NOT be None during landmark dropouts!")

    def test_jitter_does_not_reset_stillness(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        standing = make_landmarks(torso_angle_deg=5.0, gap=0.7)
        fall_detection.update(standing, frame)

        # Sudden fall
        fallen = make_landmarks(torso_angle_deg=75.0, gap=0.15)
        _, status1 = fall_detection.update(fallen, frame)
        self.assertEqual(status1["state"], "FALL_SUSPECTED")
        cd1 = status1["countdown"]

        time.sleep(0.08)
        # Add slight jitter (< 0.02) to landmarks
        jittered = make_landmarks(torso_angle_deg=74.5, gap=0.152)
        _, status2 = fall_detection.update(jittered, frame)

        self.assertEqual(status2["state"], "FALL_SUSPECTED")
        self.assertLessEqual(status2["countdown"], cd1, "Timer countdown should progress despite slight jitter!")


if __name__ == "__main__":
    unittest.main()
