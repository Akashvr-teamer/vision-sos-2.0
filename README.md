# VisionSOS

AI-powered emergency detection and response system.

VisionSOS monitors video feeds (webcam or any input video files) in real time using computer vision, MediaPipe pose estimation, and optical flow vector dynamics to automatically detect:
1. 🚶‍♂️💥 **Human Falls & Collapses**
2. 🚗💥 **Car Accidents & Vehicular Collisions**
3. 🚨🔫 **Robberies, Surrender Postures & Mugging Attacks**

When an emergency is detected, it calculates a situation severity score, records snapshot evidence, logs the event to SQLite database (`events.db`), broadcasts real-time telemetry to a web dashboard, and dispatches SMS alerts via Twilio.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
Activate your virtual environment and install dependencies:

```bash
# Using existing virtual environment:
..\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎬 How to Run with Any Random Input Video

### Run with Live Webcam (Default)
```bash
python main.py
```

### Run with Any Video File
You can provide any random video file (`.mp4`, `.mov`, `.avi`, `.mkv`) as input. Drop the file inside the `videos/` folder or specify its direct path:

```bash
# 1. Run video from videos/ folder
python main.py crash_test.mp4

# 2. Run video with direct path
python main.py C:/path/to/my_random_video.mp4

# 3. Run store/street robbery clip
python main.py robbery_sample.mp4
```

---

## 🌐 Web Dashboard

Once started, the Flask web server automatically launches alongside the computer vision engine.

- **Dashboard URL**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Features**:
  - **Live Camera Stream**: Real-time processed feed with emergency status overlays.
  - **Multi-Emergency Detection Badges**: Monitors `FALL`, `CAR ACCIDENT`, and `ROBBERY` simultaneously.
  - **Latest Event Card**: Displays severity score (0-100), severity level (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), motionless duration, and confidence.
  - **Event History**: Live table of recorded emergency events pulled from SQLite database.

---

## 📱 Twilio SMS Configuration (Optional)

To enable SMS emergency alerts:
1. Open `config/secrets.py`.
2. Add your Twilio Account SID, API Key/Secret, Twilio Phone Number, and Responder Phone Number:

```python
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_API_KEY = "your_api_key"
TWILIO_API_SECRET = "your_api_secret"
TWILIO_PHONE_NUMBER = "+1234567890"
RESPONDER_PHONE_NUMBER = "+10987654321"
```

*Note: If credentials are not provided, SMS alerting will be safely skipped while all other features continue to function normally.*

---

## 🧪 Running Unit Tests

Run the full test suite:

```bash
python -m unittest discover -s tests
```

---

## 📁 Project Architecture

```
visionSOS/
├── main.py                    # System entry point (runs Flask + Multi-Emergency CV loop)
├── pose_landmarker.task       # MediaPipe Pose Landmarker model asset
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── alerts/                    # SMS dispatch & alert logic
│   └── twilio_sms.py
├── backend/                   # Flask server, event handler, and risk scoring engine
│   ├── app.py
│   ├── event_handler.py
│   ├── fake_events.py
│   ├── situation_engine.py
│   └── system_state.py
├── config/                    # Configuration & secret handling
│   ├── config.py
│   └── secrets.py
├── cv/                        # Computer Vision modules
│   ├── camera.py
│   ├── car_accident_detection.py # Optical flow & vehicular crash vector detector
│   ├── emergency_detector.py     # Unified multi-emergency manager
│   ├── fall_detection.py         # Human fall & collapse detector
│   ├── immobility.py
│   ├── movement.py
│   ├── pose.py                   # MediaPipe Pose Landmarker wrapper
│   ├── rhythmic_motion.py
│   └── robbery_detection.py      # Hostage/surrender gesture & robbery detector
├── database/                  # SQLite database storage for emergency history
│   └── database.py
├── frontend/                  # Web dashboard UI (HTML, CSS, JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                     # Unit test suite
│   ├── test_alert.py
│   ├── test_engine.py
│   └── test_risk_score.py
└── videos/                    # Directory for test input videos
```
