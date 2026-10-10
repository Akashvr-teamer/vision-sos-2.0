import unittest
from backend.situation_engine import compute_severity, get_severity_label, process_event


class TestSituationEngine(unittest.TestCase):

    def test_collapse_severity(self):
        event = {
            "type": "collapse",
            "motionless_sec": 4.0,
            "fall_velocity": 300,
            "trigger": "sudden_drop",
            "torso_angle": 70,
            "timestamp": "2026-10-04 23:00:00"
        }
        severity = compute_severity(event)
        self.assertGreaterEqual(severity, 85)
        self.assertEqual(get_severity_label(severity), "CRITICAL")

    def test_car_accident_severity(self):
        event = {
            "type": "car_accident",
            "motion_energy": 6.5,
            "trigger": "high_kinetic_impact",
            "timestamp": "2026-10-04 20:00:00"
        }
        severity = compute_severity(event)
        self.assertGreaterEqual(severity, 85)
        self.assertEqual(get_severity_label(severity), "CRITICAL")

    def test_robbery_severity(self):
        event = {
            "type": "robbery",
            "motion_energy": 1.2,
            "trigger": "surrender_hands_up",
            "timestamp": "2026-10-04 21:00:00"
        }
        severity = compute_severity(event)
        self.assertGreaterEqual(severity, 70)
        self.assertIn(get_severity_label(severity), ["HIGH", "CRITICAL"])

    def test_normal_severity(self):
        event = {
            "type": "normal",
            "motionless_sec": 0,
            "fall_velocity": 10,
            "trigger": "",
            "torso_angle": 10,
            "timestamp": "2026-10-04 12:00:00"
        }
        severity = compute_severity(event)
        self.assertLess(severity, 40)
        self.assertEqual(get_severity_label(severity), "LOW")

    def test_process_event(self):
        event = {
            "type": "car_accident",
            "motion_energy": 4.0,
            "trigger": "high_kinetic_impact",
            "timestamp": "2026-10-04 14:00:00"
        }
        result = process_event(event)
        self.assertIn("severity", result)
        self.assertIn("label", result)
        self.assertIn("alert", result)


if __name__ == "__main__":
    unittest.main()
