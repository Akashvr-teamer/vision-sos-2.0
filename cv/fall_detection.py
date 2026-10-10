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

ANGLE_THRESHOLD = 60

GAP_THRESHOLD = 0.20

MOTION_THRESHOLD = 0.005

STILLNESS_SECONDS = 3.5

FPS = 25

ANGLE_JUMP = 25

SUSTAINED_FRAMES = 45


# -----------------------------------------
# STATE
# -----------------------------------------

state = "NORMAL"

prev_points = None

still_frames = 0

prev_angle = 0.0

fired = False

horizontal_frames = 0

trigger_reason = None


# -----------------------------------------
# UPDATE
# -----------------------------------------

def update(landmarks, frame):

    global state
    global prev_points
    global still_frames
    global prev_angle
    global fired
    global horizontal_frames
    global trigger_reason


    # -------------------------------------
    # No person detected
    # -------------------------------------

    if landmarks is None:

        return None, {
            "state": state,
            "countdown": None
        }


    # -------------------------------------
    # Calculate features
    # -------------------------------------

    angle = movement.torso_angle(
        landmarks
    )

    gap = movement.head_ankle_gap(
        landmarks
    )

    motion, prev_points = movement.motion_energy(
        landmarks,
        prev_points
    )


    # -------------------------------------
    # Horizontal detection
    # -------------------------------------

    if (
        angle > ANGLE_THRESHOLD
        and gap < GAP_THRESHOLD
    ):

        horizontal_frames += 1

    else:

        horizontal_frames = 0


    event = None


    # =====================================
    # NORMAL
    # =====================================

    if state == "NORMAL":

        # Sudden fall
        if (
            angle > ANGLE_THRESHOLD
            and gap < GAP_THRESHOLD
            and (angle - prev_angle) > ANGLE_JUMP
        ):

            state = "FALL_SUSPECTED"

            still_frames = 0

            trigger_reason = "sudden_drop"


        # Person stays horizontal
        elif horizontal_frames >= SUSTAINED_FRAMES:

            state = "FALL_SUSPECTED"

            still_frames = 0

            trigger_reason = "sustained_horizontal"


    # =====================================
    # FALL SUSPECTED
    # =====================================

    elif state == "FALL_SUSPECTED":

        # Person recovered
        if angle < ANGLE_THRESHOLD:

            state = "NORMAL"

            still_frames = 0

            fired = False

            trigger_reason = None


        else:

            # Check motion
            if motion < MOTION_THRESHOLD:

                still_frames += 1

            else:

                still_frames = 0


            # ---------------------------------
            # Emergency confirmation
            # ---------------------------------

            if (
                still_frames / FPS
            ) >= STILLNESS_SECONDS:

                if not fired:

                    state = "EMERGENCY"

                    fired = True


                    event = {

                        "type": "collapse",

                        "confidence": 0.9,

                        "timestamp":
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "motionless_sec":
                            still_frames / FPS,

                        "torso_angle":
                            angle,

                        "motion_energy":
                            motion,

                        "fall_velocity":
                            abs(
                                angle -
                                prev_angle
                            ) * FPS,

                        "snapshot_frame":
                            frame,

                        "trigger":
                            trigger_reason
                    }


    # =====================================
    # EMERGENCY
    # =====================================

    elif state == "EMERGENCY":

        # Allow recovery
        if angle < ANGLE_THRESHOLD:

            state = "NORMAL"

            still_frames = 0

            fired = False

            horizontal_frames = 0

            trigger_reason = None


    # -------------------------------------
    # Update previous angle
    # -------------------------------------

    prev_angle = angle


    # -------------------------------------
    # Countdown
    # -------------------------------------

    countdown = None


    if state == "FALL_SUSPECTED":

        countdown = max(
            0.0,
            STILLNESS_SECONDS
            - (still_frames / FPS)
        )


    # -------------------------------------
    # Status
    # -------------------------------------

    status = {

        "state": state,

        "countdown": countdown
    }


    return event, status


# -----------------------------------------
# RESET
# -----------------------------------------

def reset():

    global state
    global prev_points
    global still_frames
    global prev_angle
    global fired
    global horizontal_frames
    global trigger_reason


    state = "NORMAL"

    prev_points = None

    still_frames = 0

    prev_angle = 0.0

    fired = False

    horizontal_frames = 0

    trigger_reason = None


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