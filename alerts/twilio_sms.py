try:
    from twilio.rest import Client
except ImportError:
    Client = None

try:
    from config.secrets import (
        TWILIO_ACCOUNT_SID,
        TWILIO_API_KEY,
        TWILIO_API_SECRET,
        TWILIO_PHONE_NUMBER,
        RESPONDER_PHONE_NUMBER,
    )
except ImportError:
    TWILIO_ACCOUNT_SID = None
    TWILIO_API_KEY = None
    TWILIO_API_SECRET = None
    TWILIO_PHONE_NUMBER = None
    RESPONDER_PHONE_NUMBER = None


def send_sms():
    if not Client:
        print("[SMS] twilio package not installed. Skipping SMS alert.")
        return False

    if not TWILIO_ACCOUNT_SID or not TWILIO_API_KEY or str(TWILIO_ACCOUNT_SID).startswith("YOUR_"):
        print("[SMS] Twilio credentials not configured in config/secrets.py. Skipping SMS alert.")
        return False

    try:
        client = Client(
            TWILIO_API_KEY,
            TWILIO_API_SECRET,
            account_sid=TWILIO_ACCOUNT_SID
        )

        sms = client.messages.create(
            from_=TWILIO_PHONE_NUMBER,
            to=RESPONDER_PHONE_NUMBER,
            body="[VisionSOS] Emergency Alert: Potential fall or collapse detected!"
        )

        print(f"[SMS] Alert sent successfully: {sms.sid}")
        return True

    except Exception as e:
        print(f"[SMS] Failed to send alert: {e}")
        return False