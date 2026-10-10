import unittest
from backend.situation_engine import compute_severity, get_severity_label


class TestRiskScore(unittest.TestCase):

    def test_severity_cap(self):
        # Even with extreme values, severity should not exceed 100
        event = {
            "type": "seizure",
            "motionless_sec": 10.0,
            "fall_velocity": 1000,
            "trigger": "sudden_drop",
            "torso_angle": 90,
            "timestamp": "2026-10-04 23:30:00"
        }
        score = compute_severity(event)
        self.assertEqual(score, 100)
        self.assertEqual(get_severity_label(score), "CRITICAL")

    def test_severity_thresholds(self):
        self.assertEqual(get_severity_label(90), "CRITICAL")
        self.assertEqual(get_severity_label(70), "HIGH")
        self.assertEqual(get_severity_label(50), "MEDIUM")
        self.assertEqual(get_severity_label(20), "LOW")


if __name__ == "__main__":
    unittest.main()
