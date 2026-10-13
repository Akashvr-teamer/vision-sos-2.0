import os
import sys
import time
import threading
import signal

import cv2

from cv import camera
from cv import pose
from cv import emergency_detector

from backend.app import app
from backend.event_handler import handle_event

from backend.system_state import (
    write_status,
    write_event,
    read_state,
    set_system_active,
    set_processing_status,
    check_for_new_test,
    is_stop_requested,
    set_latest_frame_bytes
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LIVE_FRAME_PATH = os.path.join(
    BASE_DIR,
    "events",
    "live_frame.jpg"
)

os.makedirs(
    os.path.dirname(LIVE_FRAME_PATH),
    exist_ok=True
)


# ============================================================
# VIDEO SOURCE CLI PARSER
# ============================================================

def get_cli_video_source():
    """Returns video source if passed in CLI, otherwise None (idle standby)."""
    if len(sys.argv) <= 1:
        return None

    source_arg = sys.argv[1]

    if source_arg in ("--cam", "--camera", "0"):
        return 0

    if os.path.isfile(source_arg):
        return source_arg

    videos_path = os.path.join(BASE_DIR, "videos", source_arg)
    if os.path.isfile(videos_path):
        return videos_path

    videos_mp4 = os.path.join(BASE_DIR, "videos", source_arg + ".mp4")
    if os.path.isfile(videos_mp4):
        return videos_mp4

    print(f"\n[WARN] Source file '{source_arg}' not found. Starting in Dashboard Standby mode.")
    return None


# ============================================================
# FLASK SERVER
# ============================================================

def run_flask():
    print()
    print("=" * 60)
    print("      VisionSOS - Multi-Emergency AI Monitoring System      ")
    print("=" * 60)
    print(" Dashboard: http://127.0.0.1:5000")
    print(" Status   : Server online. Camera is on standby.")
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )


# ============================================================
# LIVE FRAME BUFFER DISPATCHER
# ============================================================

def save_live_frame(frame, persist_disk=False):
    """
    Encodes frame to JPEG in memory for ultra-fast, zero-latency streaming.
    Only writes to disk occasionally to avoid OneDrive sync locks.
    """
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if success:
        frame_bytes = buffer.tobytes()
        set_latest_frame_bytes(frame_bytes)

        if persist_disk:
            temp_path = os.path.join(
                os.path.dirname(LIVE_FRAME_PATH),
                "live_frame_tmp.jpg"
            )
            try:
                with open(temp_path, "wb") as f:
                    f.write(frame_bytes)
                os.replace(temp_path, LIVE_FRAME_PATH)
            except Exception:
                pass


# ============================================================
# COMPUTER VISION WORKER LOOP
# ============================================================

def run_detection_worker(initial_source=None):
    """
    Main detection worker loop. Runs continuously in background.
    Stays in standby until a source (live webcam or uploaded video) is triggered.
    Supports isolated single-hazard pipelines or unified multi-hazard mode.
    """
    current_source = initial_source
    current_category = "Unified"
    running = True

    show_desktop_window = "--show-window" in sys.argv

    while running:
        if current_source is None:
            set_processing_status(False, "Standby")
            write_status("STANDBY", None, {
                "fall_state": "NORMAL",
                "car_state": "NORMAL",
                "robbery_state": "NORMAL",
                "fire_state": "NORMAL"
            })

            # Wait for test request from UI
            while current_source is None and running:
                new_source, new_cat = check_for_new_test()
                if new_source is not None:
                    current_source = new_source
                    current_category = new_cat or "Unified"
                    break
                time.sleep(0.2)

            if not running:
                break

        # Open source
        source_name = "Live Webcam" if current_source == 0 else os.path.basename(str(current_source))
        print(f"\n[CV] Starting detection on: {source_name} (Category: {current_category})")
        set_processing_status(True, source_name)
        emergency_detector.reset()
        emergency_detector.set_active_category(current_category)
        if hasattr(pose, "reset_tracking"):
            pose.reset_tracking()

        cap = camera.open_source(current_source)
        if cap is None:
            print(f"[ERROR] Could not open video source: {current_source}")
            current_source = None
            continue

        frame_count = 0
        last_log_state = "NORMAL"

        try:
            while running:
                # Check for stop request
                if is_stop_requested():
                    print("\n[CV] Stop requested. Releasing video source.")
                    current_source = None
                    break

                # Check for new source or category switch
                new_source, new_cat = check_for_new_test()
                if new_source is not None:
                    print(f"\n[CV] Switching source to: {new_source} ({new_cat})")
                    current_source = new_source
                    current_category = new_cat or "Unified"
                    break

                # Support dynamic category change during active stream without restart
                active_cat = read_state().get("active_category")
                if active_cat and active_cat != current_category:
                    print(f"\n[CV] Switched active pipeline category to: {active_cat}")
                    current_category = active_cat
                    emergency_detector.set_active_category(current_category)

                # Read Frame
                raw_frame = camera.read_frame(cap)
                if raw_frame is None:
                    print(f"\n[CV] Stream ended for {source_name}. Returning to Standby.")
                    current_source = None
                    break

                # Resize early to standard processing size (width 640) for high FPS
                frame = camera.resize_frame(raw_frame, width=640)

                frame_count += 1

                # Normalize active category
                cat_lower = current_category.strip().lower()
                is_car_mode = any(k in cat_lower for k in ["car", "accident"])
                is_fall_mode = any(k in cat_lower for k in ["fall", "collapse"])
                is_fire_mode = any(k in cat_lower for k in ["fire", "smoke", "combustion"])
                is_robbery_mode = any(k in cat_lower for k in ["robbery", "threat"])
                is_unified_mode = (not is_car_mode and not is_fall_mode and not is_fire_mode and not is_robbery_mode) or cat_lower == "unified"

                # 1. Pose landmarks:
                # In Car Accident or Fire mode, SKIP pose landmarks completely (landmarks = None).
                # This eliminates MediaPipe pose calculation overhead and prevents stick figures or false human poses.
                if is_car_mode or is_fire_mode:
                    landmarks = None
                else:
                    landmarks = pose.get_landmarks(frame)

                # 2. Emergency detection isolated to requested category
                event, status = emergency_detector.update(landmarks, frame, category=current_category)
                current_state = status["state"]
                countdown = status["countdown"]

                # Expose confirmed robbery signals for overlay and state API
                robbery_sub = status.get("robbery_status", {})
                status["robbery_signals"] = robbery_sub.get("active_signals", {})

                # 3. Update backend state for API
                write_status(current_state, countdown, {
                    "fall_state": status.get("fall_state", "NORMAL"),
                    "car_state": status.get("car_state", "NORMAL"),
                    "robbery_state": status.get("robbery_state", "NORMAL"),
                    "fire_state": status.get("fire_state", "NORMAL"),
                    "robbery_signals": status["robbery_signals"],
                })

                # 4. Visual Scene Overlays
                display_frame = frame.copy()

                # A. Render Overlays (Vehicle boxes and Fire boxes)
                if is_car_mode or is_fire_mode or is_unified_mode:
                    display_frame = emergency_detector.draw(display_frame)

                # B. Render Pose Skeleton only if genuine human detected and in Fall, Robbery, or Unified mode
                if landmarks and (is_fall_mode or is_robbery_mode or is_unified_mode) and current_state not in ("ACCIDENT_SUSPECTED", "ACCIDENT_EMERGENCY"):
                    display_frame = pose.draw(display_frame, landmarks)

                # C. Robbery confirmed-signal overlay — show WHAT movements are detected
                if is_robbery_mode or is_unified_mode:
                    robbery_signals = status.get("robbery_signals", {})
                    if robbery_signals:
                        rh, rw_dim = display_frame.shape[:2]
                        badge_y = rh - 20 - (len(robbery_signals) * 28)
                        badge_y = max(badge_y, 60)
                        for sig_label in robbery_signals.values():
                            badge_x = 12
                            badge_w = min(len(sig_label) * 11 + 20, rw_dim - 24)
                            # Draw pill background
                            cv2.rectangle(display_frame,
                                          (badge_x, badge_y - 20),
                                          (badge_x + badge_w, badge_y + 4),
                                          (20, 60, 180), -1)
                            cv2.rectangle(display_frame,
                                          (badge_x, badge_y - 20),
                                          (badge_x + badge_w, badge_y + 4),
                                          (80, 140, 255), 1)
                            cv2.putText(display_frame,
                                        f"DETECTED: {sig_label}",
                                        (badge_x + 8, badge_y),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.42,
                                        (200, 225, 255), 1, cv2.LINE_AA)
                            badge_y += 28

                # 5. Draw UI Telemetry HUD on frame
                h, w = display_frame.shape[:2]

                # HUD Top banner
                color_map = {
                    "NORMAL": (30, 200, 30),
                    "FALL_SUSPECTED": (0, 190, 255),
                    "ACCIDENT_SUSPECTED": (0, 190, 255),
                    "SUSPICIOUS_ACTIVITY": (0, 190, 255),
                    "FIRE_SUSPECTED": (0, 140, 255),
                    "FALL_EMERGENCY": (30, 30, 240),
                    "ACCIDENT_EMERGENCY": (30, 30, 240),
                    "ROBBERY_EMERGENCY": (30, 30, 240),
                    "FIRE_EMERGENCY": (0, 60, 255)
                }
                status_color = color_map.get(current_state, (200, 200, 200))

                # Background top overlay
                cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 24, 30), -1)

                status_label = current_state.replace("_", " ")
                if countdown is not None and isinstance(countdown, (int, float)):
                    status_label += f" ({countdown:.1f}s)"

                # Category badge display string
                cat_tag = current_category.upper()
                if "CAR" in cat_tag or "ACCIDENT" in cat_tag:
                    cat_tag = "CAR ACCIDENT"
                elif "FALL" in cat_tag or "COLLAPSE" in cat_tag:
                    cat_tag = "FALL & COLLAPSE"
                elif "ROBBERY" in cat_tag:
                    cat_tag = "ROBBERY"
                elif "FIRE" in cat_tag:
                    cat_tag = "FIRE & SMOKE"
                else:
                    cat_tag = "UNIFIED ALL"

                cv2.putText(display_frame, f"STATUS: {status_label}", (20, 32),
                            cv2.FONT_HERSHEY_DUPLEX, 0.70, status_color, 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"MODE: {cat_tag}", (w - 240, 24),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, "VisionSOS AI", (w - 240, 44),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

                # Save frame in memory for ultra-smooth video stream (persist to disk only occasionally)
                save_live_frame(display_frame, persist_disk=(frame_count % 30 == 0))

                # Process triggered emergency event
                if event is not None:
                    print("\n" + "=" * 55)
                    print(f"🚨 [EMERGENCY DETECTED] Type: {event['type'].upper()}")
                    print(f"Confidence : {event['confidence'] * 100:.1f}%")
                    print(f"Trigger    : {event.get('trigger', 'anomaly')}")
                    print(f"Timestamp  : {event.get('timestamp')}")
                    print("=" * 55)

                    result = handle_event(event)
                    write_event(result)

                # Terminal telemetry log on state transition
                if current_state != last_log_state:
                    if "EMERGENCY" in current_state:
                        print(f"[CV] STATE -> {current_state}")
                    elif "SUSPECTED" in current_state or "SUSPICIOUS" in current_state:
                        print(f"[CV] STATE -> {current_state}")
                        # Print confirmed robbery causes if applicable
                        rob_sigs = status.get("robbery_signals", {})
                        if rob_sigs:
                            for label in rob_sigs.values():
                                print(f"   >> Confirmed: {label}")
                    elif current_state == "NORMAL":
                        print("[CV] STATE -> NORMAL (All Hazards Clear)")
                    last_log_state = current_state

                # Optional desktop OpenCV window only if explicitly requested
                if show_desktop_window:
                    cv2.imshow("VisionSOS AI Feed", display_frame)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        print("[CV] Quit key pressed.")
                        current_source = None
                        break
                else:
                    # Regulate frame rate for video files (live webcam regulates naturally)
                    if current_source != 0:
                        time.sleep(0.025)

        finally:
            camera.close(cap)
            if show_desktop_window:
                cv2.destroyAllWindows()

    set_system_active(False)
    print("[CV] Detection worker stopped.")


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    # Setup signal handler for clean graceful shutdown
    def signal_handler(sig, frame):
        print("\n[VisionSOS] Shutting down cleanly...")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, signal_handler)
    except Exception:
        pass

    cli_source = get_cli_video_source()

    # Start Flask server thread
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )
    flask_thread.start()

    time.sleep(0.6)
    set_system_active(True)

    # Run CV loop
    try:
        run_detection_worker(initial_source=cli_source)
    except KeyboardInterrupt:
        print("\n[VisionSOS] Graceful shutdown completed.")
        sys.exit(0)


if __name__ == "__main__":
    main()