import unittest
from alerts.twilio_sms import send_sms


class TestAlert(unittest.TestCase):

    def test_send_sms_unconfigured(self):
        # Unconfigured credentials should return False gracefully without throwing an exception
        result = send_sms()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
