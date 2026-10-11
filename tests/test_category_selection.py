import unittest
import numpy as np
from cv import emergency_detector
from backend import system_state
from backend.app import app


class TestCategorySelection(unittest.TestCase):

    def setUp(self):
        emergency_detector.reset()
        system_state.set_active_category("Unified")

    def test_isolated_car_mode_bypasses_fall(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Synthetic horizontal lying landmarks (which in unified would trigger fall logic)
        fake_landmarks = [
            {"x": 0.5, "y": 0.8, "visibility": 0.9} for _ in range(33)
        ]

        event, status = emergency_detector.update(fake_landmarks, frame, category="car")

        # Fall state should remain NORMAL because it was bypassed
        self.assertEqual(status["fall_state"], "NORMAL")
        self.assertEqual(status["robbery_state"], "NORMAL")
        self.assertEqual(status["fire_state"], "NORMAL")
        self.assertEqual(status["active_category"], "car")

    def test_isolated_fall_mode_bypasses_car_and_fire(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        event, status = emergency_detector.update(None, frame, category="fall")

        self.assertEqual(status["car_state"], "NORMAL")
        self.assertEqual(status["fire_state"], "NORMAL")
        self.assertEqual(status["robbery_state"], "NORMAL")
        self.assertEqual(status["active_category"], "fall")

    def test_isolated_robbery_mode(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        event, status = emergency_detector.update(None, frame, category="robbery")

        self.assertEqual(status["car_state"], "NORMAL")
        self.assertEqual(status["fire_state"], "NORMAL")
        self.assertEqual(status["fall_state"], "NORMAL")
        self.assertEqual(status["active_category"], "robbery")

    def test_isolated_fire_mode(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        event, status = emergency_detector.update(None, frame, category="fire")

        self.assertEqual(status["car_state"], "NORMAL")
        self.assertEqual(status["fall_state"], "NORMAL")
        self.assertEqual(status["robbery_state"], "NORMAL")
        self.assertEqual(status["active_category"], "fire")

    def test_system_state_category_tracking(self):
        system_state.request_new_test("test_video.mp4", category="car")
        source, cat = system_state.check_for_new_test()

        self.assertEqual(source, "test_video.mp4")
        self.assertEqual(cat, "car")

        state = system_state.read_state()
        self.assertEqual(state["active_category"], "car")

    def test_api_set_category_endpoint(self):
        client = app.test_client()
        res = client.post("/api/set_category", json={"category": "car"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["active_category"], "car")

        # Check system state reflects this
        state = system_state.read_state()
        self.assertEqual(state["active_category"], "car")


if __name__ == "__main__":
    unittest.main()
