import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import time
from flask import (
    Flask,
    jsonify,
    send_file,
    send_from_directory,
    request,
    Response
)

from database.database import init_db, get_events, update_event_category
from backend.system_state import (
    read_state,
    request_new_test,
    request_stop,
    set_active_category,
    get_latest_frame_bytes,
    wait_for_next_frame
)
from cv import emergency_detector

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

EVENTS_DIR = os.path.join(
    BASE_DIR,
    "events"
)
os.makedirs(EVENTS_DIR, exist_ok=True)

LIVE_FRAME_PATH = os.path.join(
    EVENTS_DIR,
    "live_frame.jpg"
)

UPLOADS_DIR = os.path.join(
    BASE_DIR,
    "videos",
    "uploads"
)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

# Initialize database
init_db()


@app.route("/")
def home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/api/start_test", methods=["POST"])
def start_test():
    """
    Start Endpoint:
    Runs the selected detection category (e.g. Fall, Car Accident, Robbery, Fire, or Unified)
    to execute only the required pipeline and avoid cross-detector clash.
    """
    source_type = request.form.get("source_type", "video")
    category = request.form.get("category", "Unified").strip() or "Unified"

    if source_type == "camera":
        request_new_test(0, category)
        return jsonify({
            "success": True,
            "category": category,
            "message": f"Live webcam started with {category} detection active."
        })

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOADS_DIR, video_file.filename)
    video_file.save(file_path)

    request_new_test(file_path, category)
    return jsonify({
        "success": True,
        "category": category,
        "message": f"Processing '{video_file.filename}' with {category} detection active."
    })


@app.route("/api/set_category", methods=["POST"])
def set_category_route():
    """Switch active emergency detection category on the fly."""
    data = request.get_json() or {}
    category = data.get("category", "Unified").strip() or "Unified"
    set_active_category(category)
    if hasattr(emergency_detector, "set_active_category"):
        emergency_detector.set_active_category(category)
    return jsonify({
        "success": True,
        "active_category": category,
        "message": f"Active detection category set to: {category}"
    })


@app.route("/api/stop_test", methods=["POST"])
def stop_test():
    """Stop active camera / video stream and return to standby."""
    request_stop()
    return jsonify({
        "success": True,
        "message": "Stream stopped. System returned to Standby."
    })


@app.route("/api/reset", methods=["POST"])
def reset_detector():
    """Reset emergency detector states back to normal."""
    emergency_detector.reset()
    return jsonify({
        "success": True,
        "message": "Detection states reset to NORMAL."
    })


@app.route("/api/status")
def status():
    return jsonify(read_state())


@app.route("/api/events")
def events():
    raw_events = get_events()
    # Format snapshot urls for client
    formatted = []
    for ev in raw_events:
        item = dict(ev)
        if item.get("snapshot_path"):
            item["snapshot_url"] = f"/api/snapshot/{os.path.basename(item['snapshot_path'])}"
        else:
            item["snapshot_url"] = None
        formatted.append(item)
    return jsonify(formatted)


@app.route("/api/snapshot/<path:filename>")
def snapshot(filename):
    file_path = os.path.join(EVENTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Snapshot not found"}), 404
    return send_file(file_path, mimetype="image/jpeg")


@app.route("/api/events/<int:event_id>/category", methods=["POST"])
def update_category(event_id):
    data = request.get_json()
    if not data or "category" not in data:
        return jsonify({"error": "Missing category"}), 400

    success = update_event_category(event_id, data["category"])
    if success:
        return jsonify({"success": True, "event_id": event_id, "category": data["category"]})
    return jsonify({"error": "Failed to update category"}), 500


@app.route("/api/frame")
def frame():
    frame_bytes = get_latest_frame_bytes()
    if frame_bytes is not None:
        response = Response(frame_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    if os.path.exists(LIVE_FRAME_PATH):
        response = send_file(
            LIVE_FRAME_PATH,
            mimetype="image/jpeg",
            max_age=0
        )
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return jsonify({
        "error": "Camera frame not available"
    }), 404


@app.route("/api/video_feed")
def video_feed():
    """High-speed MJPEG video stream direct from RAM buffer."""
    def generate():
        while True:
            frame_bytes = wait_for_next_frame(timeout=0.5)
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            else:
                time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "system": "VisionSOS",
        "unified_detectors": ["fall", "car_accident", "robbery", "fire"]
    })


if __name__ == "__main__":
    from main import main
    main()