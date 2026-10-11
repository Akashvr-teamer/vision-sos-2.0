import os
import sys
import time

import cv2
import numpy as np


# -----------------------------------------
# IMPORT MOVEMENT
# -----------------------------------------

try:

    from . import movement

except ImportError:

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    if current_dir not in sys.path:
        sys.path.insert(
            0,
            current_dir
        )

    import movement


# -----------------------------------------
# CONSTANTS
# -----------------------------------------

# -----------------------------------------
# CONSTANTS
# -----------------------------------------

ANGLE_THRESHOLD = 55         # Torso angle away from vertical indicating horizontal pose
GAP_THRESHOLD = 0.28           # Normalized head-ankle distance indicating lying down
MOTION_THRESHOLD = 0.025       # Motion energy threshold for stillness (resistant to landmark jitter)
STILLNESS_SECONDS = 20.0        # Immobility duration required to trigger emergency
ANGLE_JUMP = 20                # Minimum sudden angle delta in degrees
SUSTAINED_FRAMES = 25          # Consecutive horizontal frames to enter FALL_SUSPECTED without sudden drop
RECOVERY_FRAMES_REQ = 15       # Consecutive standing frames required to recover back to NORMAL


# -----------------------------------------
# STATE
# -----------------------------------------

state = "NORMAL"
prev_points = None
prev_angle = 0.0
fired = False
horizontal_frames = 0
trigger_reason = None

# Time-based tracking
last_frame_time = None
accumulated_stillness = 0.0
recovery_frames = 0


# -----------------------------------------
# UPDATE
# -----------------------------------------

def update(landmarks, frame):
    global state
    global prev_points
    global prev_angle
    global fired
    global horizontal_frames
    global trigger_reason
    global last_frame_time
    global accumulated_stillness
    global recovery_frames

    now = time.time()
    dt = 0.04  # default assuming ~25 FPS
    if last_frame_time is not None:
        elapsed = now - last_frame_time
        if 0.001 < elapsed < 1.0:
            dt = elapsed
    last_frame_time = now

    event = None

    # -------------------------------------------------------------
    # CASE 1: No landmarks detected (or occluded on floor)
    # -------------------------------------------------------------
    if landmarks is None or not hasattr(landmarks, "landmark"):
        if state == "FALL_SUSPECTED":
            # Person collapsed and is now occluded on the floor:
            # DO NOT drop countdown! Continue advancing stillness timer.
            accumulated_stillness += dt
            countdown = max(0.0, STILLNESS_SECONDS - accumulated_stillness)

            if accumulated_stillness >= STILLNESS_SECONDS and not fired:
                state = "EMERGENCY"
                fired = True
                event = {
                    "type": "collapse",
                    "confidence": 0.88,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "motionless_sec": round(accumulated_stillness, 1),
                    "torso_angle": round(prev_angle, 1),
                    "motion_energy": 0.0,
                    "fall_velocity": 45.0,
                    "snapshot_frame": frame,
                    "trigger": trigger_reason or "sudden_drop"
                }

            return event, {
                "state": state,
                "countdown": round(countdown, 1)
            }
        elif state == "EMERGENCY":
            return None, {
                "state": "EMERGENCY",
                "countdown": 0.0
            }
        else:
            return None, {
                "state": "NORMAL",
                "countdown": None
            }

    # -------------------------------------------------------------
    # CASE 2: Landmarks present - calculate biomechanics
    # -------------------------------------------------------------
    angle = movement.torso_angle(landmarks)
    gap = movement.head_ankle_gap(landmarks)
    motion, prev_points = movement.motion_energy(landmarks, prev_points)

    is_horizontal = (angle > ANGLE_THRESHOLD and gap < GAP_THRESHOLD)
    is_standing = (angle < 45 and gap > 0.30)

    if is_horizontal:
        horizontal_frames += 1
    else:
        horizontal_frames = 0

    # -------------------------------------------------------------
    # STATE MACHINE
    # -------------------------------------------------------------
    if state == "NORMAL":
        # Scenario A: Sudden rapid drop/fall
        if is_horizontal and (angle - prev_angle) > ANGLE_JUMP:
            state = "FALL_SUSPECTED"
            accumulated_stillness = 0.0
            recovery_frames = 0
            trigger_reason = "sudden_drop"

        # Scenario B: Sustained lying horizontal
        elif horizontal_frames >= SUSTAINED_FRAMES:
            state = "FALL_SUSPECTED"
            accumulated_stillness = 0.0
            recovery_frames = 0
            trigger_reason = "sustained_horizontal"

    elif state == "FALL_SUSPECTED":
        # Check for genuine standing recovery
        if is_standing:
            recovery_frames += 1
            if recovery_frames >= RECOVERY_FRAMES_REQ:
                state = "NORMAL"
                accumulated_stillness = 0.0
                recovery_frames = 0
                fired = False
                trigger_reason = None
        else:
            recovery_frames = 0

            # Immobility evaluation with jitter tolerance
            if motion < MOTION_THRESHOLD:
                accumulated_stillness += dt
            else:
                # Gradual decay rather than hard reset to 0 on 1-frame jitter
                if motion > 0.05:
                    accumulated_stillness = max(0.0, accumulated_stillness - dt * 2.0)

            # Trigger emergency after stillness duration
            if accumulated_stillness >= STILLNESS_SECONDS and not fired:
                state = "EMERGENCY"
                fired = True
                event = {
                    "type": "collapse",
                    "confidence": 0.92,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "motionless_sec": round(accumulated_stillness, 1),
                    "torso_angle": round(angle, 1),
                    "motion_energy": round(motion, 4),
                    "fall_velocity": round(abs(angle - prev_angle) / max(dt, 0.001), 1),
                    "snapshot_frame": frame,
                    "trigger": trigger_reason or "sudden_drop"
                }

    elif state == "EMERGENCY":
        # Allow recovery only after sustained standing
        if is_standing:
            recovery_frames += 1
            if recovery_frames >= (RECOVERY_FRAMES_REQ * 2):
                state = "NORMAL"
                fired = False
                accumulated_stillness = 0.0
                recovery_frames = 0
                trigger_reason = None
        else:
            recovery_frames = 0

    prev_angle = angle

    # Countdown calculation
    countdown = None
    if state == "FALL_SUSPECTED":
        countdown = round(max(0.0, STILLNESS_SECONDS - accumulated_stillness), 1)

    return event, {
        "state": state,
        "countdown": countdown
    }


# -----------------------------------------
# RESET
# -----------------------------------------

def reset():
    global state
    global prev_points
    global prev_angle
    global fired
    global horizontal_frames
    global trigger_reason
    global last_frame_time
    global accumulated_stillness
    global recovery_frames

    state = "NORMAL"
    prev_points = None
    prev_angle = 0.0
    fired = False
    horizontal_frames = 0
    trigger_reason = None
    last_frame_time = None
    accumulated_stillness = 0.0
    recovery_frames = 0


# -----------------------------------------
# TEST RUNNER
# -----------------------------------------

if __name__ == "__main__":

    try:

        from . import camera, pose

    except ImportError:

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        if current_dir not in sys.path:

            sys.path.insert(
                0,
                current_dir
            )

        import camera
        import pose


    # -------------------------------------
    # Video source
    # -------------------------------------

    if len(sys.argv) > 1:

        source_arg = sys.argv[1]


        if os.path.isfile(source_arg):

            source = source_arg


        elif os.path.isfile(
            os.path.join(
                "videos",
                source_arg
            )
        ):

            source = os.path.join(
                "videos",
                source_arg
            )


        elif os.path.isfile(
            os.path.join(
                "videos",
                source_arg + ".mp4"
            )
        ):

            source = os.path.join(
                "videos",
                source_arg + ".mp4"
            )


        else:

            source = os.path.join(
                "videos",
                source_arg
            )


    else:

        source = 0


    print(
        f"Source: {source}"
    )

    print(
        "Press Q to quit"
    )


    cap = camera.open_source(
        source
    )


    if cap is None:

        print(
            f"ERROR: could not open {source}"
        )

        sys.exit(1)


    cv2.namedWindow(
        "VisionSOS - Fall Detection",
        cv2.WINDOW_NORMAL
    )


    cv2.resizeWindow(
        "VisionSOS - Fall Detection",
        960,
        540
    )


    frame_count = 0


    while True:

        frame = camera.read_frame(
            cap
        )


        if frame is None:

            break


        frame_count += 1


        landmarks = pose.get_landmarks(
            frame
        )


        event, status = update(
            landmarks,
            frame
        )


        if (
            status["state"]
            == "FALL_SUSPECTED"
        ):

            print(
                f"[CV] FALL SUSPECTED | "
                f"Countdown: "
                f"{status['countdown']:.1f}s"
            )


        elif (
            status["state"]
            == "EMERGENCY"
        ):

            print(
                "[CV] 🚨 EMERGENCY"
            )


        else:

            print(
                f"[CV] NORMAL | "
                f"Angle: "
                f"{prev_angle:.1f}"
            )


        if event is not None:

            print()
            print(
                "=" * 50
            )

            print(
                "       🚨 VISION SOS EMERGENCY"
            )

            print(
                "=" * 50
            )

            print(
                f"Event: "
                f"{event['type']}"
            )

            print(
                f"Confidence: "
                f"{event['confidence']}"
            )

            print(
                f"Motionless: "
                f"{event['motionless_sec']:.1f}s"
            )

            print(
                f"Torso angle: "
                f"{event['torso_angle']:.1f}°"
            )

            print(
                f"Trigger: "
                f"{event.get('trigger')}"
            )

            print(
                "=" * 50
            )


        frame = pose.draw(
            frame,
            landmarks
        )


        color_map = {

            "NORMAL":
                (0, 255, 0),

            "FALL_SUSPECTED":
                (0, 255, 255),

            "EMERGENCY":
                (0, 0, 255)
        }


        color = color_map.get(
            status["state"],
            (255, 255, 255)
        )


        state_text = status[
            "state"
        ]


        if status["countdown"] is not None:

            state_text += (
                f" "
                f"({status['countdown']:.1f}s)"
            )


        cv2.putText(
            frame,
            state_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )


        h, w = frame.shape[:2]


        if w > 960:

            new_height = int(
                h * (960 / w)
            )

            frame = cv2.resize(
                frame,
                (960, new_height)
            )


        cv2.imshow(
            "VisionSOS - Fall Detection",
            frame
        )


        if (
            cv2.waitKey(30) & 0xFF
            == ord("q")
        ):

            break


    camera.close(
        cap
    )